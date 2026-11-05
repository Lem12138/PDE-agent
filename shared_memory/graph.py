# graph_memory.py — 有向图记忆系统，追踪操作间的数据依赖，支持错误溯源和子图裁剪
import json
import os
import re
from typing import Dict, List, Optional, Set

# 边关系类型
EDGE_PRODUCES = "PRODUCES"       # 节点A产出了某个产物 → 节点B
EDGE_CONSUMES = "CONSUMES"       # 节点B消费了节点A的产物
EDGE_TRIGGERED = "TRIGGERED"     # 错误触发诊断/修复


class GraphMemory:
    """
    有向图记忆：建模科研流程中每一步操作之间的数据依赖关系。

    节点 = 一次操作（探测数据、加载ckpt、训练、评估…）
    边   = 产物流向（A的输出→B的输入）

    核心用途:
      - 出错时沿边回溯 → 精准定位根因
      - 向下游追溯 → 找出哪些步骤被污染需要重跑
      - 子图裁剪 → 只给 LLM 注入相关上下文
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.nodes: Dict[str, dict] = {}   # node_id → {type, content, metadata}
        self.edges: List[dict] = []         # [{source, target, relation, artifact}]
        if filepath and os.path.exists(filepath):
            self._load()

    # ============================================================
    # 写入
    # ============================================================
    def add_node(self, node_id: str, node_type: str, content: str = "",
                 metadata: Optional[dict] = None) -> None:
        """添加/更新节点。node_type 如 'probe', 'load_ckpt', 'train', 'eval', 'code_gen', 'error'"""
        self.nodes[node_id] = {
            "type": node_type,
            "content": content,
            "metadata": metadata or {},
        }
        self._save()

    def add_edge(self, source: str, target: str, relation: str,
                 artifact: str = "") -> None:
        """添加有向边，artifact 描述流动的产物名称"""
        # 自动补全缺失的端点
        for nid in [source, target]:
            if nid not in self.nodes:
                self.add_node(nid, "unknown", f"auto-created for edge from {source}")

        edge = {"source": source, "target": target, "relation": relation, "artifact": artifact}
        if edge not in self.edges:
            self.edges.append(edge)
            self._save()

    # ============================================================
    # 查询 — 错误追溯（论文核心机制）
    # ============================================================
    def upstream(self, node_id: str) -> List[str]:
        """追溯上游：哪些节点的产出流入了我？用于根因分析"""
        return [e["source"] for e in self.edges if e["target"] == node_id]

    def downstream(self, node_id: str) -> List[str]:
        """追踪下游：我的产出流向了哪些节点？用于影响分析"""
        return [e["target"] for e in self.edges if e["source"] == node_id]

    def find_contaminated(self, error_node: str) -> Set[str]:
        """
        从出错节点出发，找所有被污染的节点（下游递归）。
        这些节点需要重新执行。
        """
        contaminated = set()
        stack = [error_node]
        while stack:
            nid = stack.pop()
            for child in self.downstream(nid):
                if child not in contaminated:
                    contaminated.add(child)
                    stack.append(child)
        return contaminated

    def trace_root_cause(self, error_node: str) -> List[str]:
        """
        从出错节点递归向上游追溯，找到所有可能根因。
        返回有序列表：最上游在前。
        """
        root_causes = []
        visited = set()
        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            parents = self.upstream(nid)
            if not parents:
                root_causes.append(nid)
            else:
                for p in parents:
                    dfs(p)
        dfs(error_node)
        return root_causes

    def get_node(self, node_id: str) -> Optional[dict]:
        return self.nodes.get(node_id)

    def find_by_type(self, node_type: str) -> List[str]:
        return [nid for nid, info in self.nodes.items() if info["type"] == node_type]

    # ============================================================
    # LLM 上下文格式化
    # ============================================================
    def format_for_llm(self) -> str:
        """输出完整图记忆的 Markdown，注入 LLM prompt。"""
        if not self.nodes:
            return "图记忆为空，尚无执行记录。"

        lines = ["### 执行图记忆 (Graph Memory)\n"]
        lines.append("#### 节点 (执行步骤):\n")
        for nid, info in self.nodes.items():
            meta = json.dumps(info["metadata"], ensure_ascii=False) if info["metadata"] else ""
            lines.append(f"- **[{info['type']}]** `{nid}`: {info['content'][:200]} {meta}")
        lines.append("\n#### 边 (数据依赖):\n")
        if not self.edges:
            lines.append("暂无。")
        for e in self.edges:
            lines.append(f"- `{e['source']}` --[{e['relation']}: {e['artifact']}]--> `{e['target']}`")
        return "\n".join(lines)

    def format_subgraph(self, node_ids: List[str]) -> str:
        """只输出相关节点的子图，避免全量日志撑爆 context。"""
        nid_set = set(node_ids)
        relevant_edges = [e for e in self.edges
                          if e["source"] in nid_set or e["target"] in nid_set]
        # 把边涉及的所有节点也加进来
        for e in relevant_edges:
            nid_set.add(e["source"])
            nid_set.add(e["target"])

        lines = [f"### 子图 (聚焦 {', '.join(node_ids)})\n"]
        for nid in nid_set:
            if nid in self.nodes:
                info = self.nodes[nid]
                lines.append(f"- [{info['type']}] `{nid}`: {info['content'][:150]}")
        lines.append("")
        for e in relevant_edges:
            lines.append(f"- `{e['source']}` --[{e['relation']}]--> `{e['target']}`")
        return "\n".join(lines)

    def lineage_path(self, node_id: str, max_depth: int = 32) -> List[str]:
        """沿上游一路回溯到根，返回正序路径（最上游在前）。"""
        path, cur, seen = [], node_id, set()
        while cur and cur not in seen and len(path) < max_depth:
            seen.add(cur)
            path.append(cur)
            parents = self.upstream(cur)
            cur = parents[0] if parents else None
        path.reverse()
        return path

    # ============================================================
    # 演化轨迹压缩 — 每轮一行，保住跨轮记忆
    # ============================================================
    def _rounds(self) -> Dict[int, Dict[str, dict]]:
        """把节点按轮次聚合：{轮次: {后缀: 节点}}。"""
        rounds: Dict[int, Dict[str, dict]] = {}
        for nid, info in self.nodes.items():
            m = re.match(r"^R(\d+)_(\w+)$", nid)
            if m:
                rounds.setdefault(int(m.group(1)), {})[m.group(2)] = info
        return rounds

    def _one_line(self, rnd: int, parts: Dict[str, dict]) -> str:
        """把一轮的所有信息压成一行。"""
        hyp = parts.get("hypothesis", {})
        title = (hyp.get("content") or "").strip().splitlines()
        title = (title[0] if title else "")[:38]

        seg = []
        edit = parts.get("edit", {})
        if edit:
            seg.append(f"改 {edit.get('metadata', {}).get('applied', '?')} 处")

        jud = parts.get("judgement", {})
        meta = jud.get("metadata", {}) or {}
        verdict = meta.get("verdict") or jud.get("content", "") or ""
        ov = meta.get("overall", {}) or {}
        if ov.get("baseline") is not None and ov.get("current") is not None:
            seg.append(f"{ov['baseline']:.4g}→{ov['current']:.4g}")
        segs = meta.get("segments", {}) or {}
        seg_txt = " ".join(
            f"{k}{v['r']*100:+.1f}%" for k, v in segs.items()
            if isinstance(v, dict) and v.get("r") is not None
        )
        if seg_txt:
            seg.append(seg_txt)

        outcome = parts.get("outcome", {})
        action = (outcome.get("content") or "").strip()
        if not action and verdict:
            action = verdict
        if action:
            seg.append(action)

        head = f"R{rnd}"
        if title:
            head += f": {title}"
        return head + (" | " + " | ".join(seg) if seg else "")

    def format_trajectory(self) -> str:
        """全量轮次摘要：每轮一行，永远注入，保证跨轮记忆不丢。"""
        rounds = self._rounds()
        if not rounds:
            return "尚无演化记录。"
        lines = ["### 演化轨迹（每轮一行，全量保留）"]
        for rnd in sorted(rounds):
            lines.append("- " + self._one_line(rnd, rounds[rnd]))
        return "\n".join(lines)

    def relevant_rounds(self, task_text: str, recent: int = 2) -> List[int]:
        """按任务文本里的关键词挑出相关轮次；再并上最近几轮。

        只看「方案描述 / 判定理由 / 处置结果」，不看原始指标 ——
        否则每轮都含 seg1/seg2 字段，等于全部命中。
        """
        rounds = self._rounds()
        if not rounds:
            return []
        keys = set(re.findall(r"seg[123]|fno|mgno|unet|deeponet", (task_text or "").lower()))
        picked = set(sorted(rounds)[-recent:])
        if not keys:
            return sorted(picked)
        for rnd, parts in rounds.items():
            blob = " ".join(
                str(parts.get(k, {}).get("content", ""))
                for k in ("hypothesis", "outcome")
            ).lower()
            reasons = (parts.get("judgement", {}).get("metadata", {}) or {}).get("reasons") or []
            blob += " " + " ".join(map(str, reasons)).lower()
            if any(k in blob for k in keys):
                picked.add(rnd)
        return sorted(picked)

    def format_for_task(self, task_text: str, recent: int = 2) -> str:
        """两层注入：全量轨迹（压缩） + 相关轮次细节（展开）。"""
        out = [self.format_trajectory()]
        rnds = self.relevant_rounds(task_text, recent=recent)
        detail = []
        for rnd in rnds:
            for suffix in ("hypothesis", "edit", "eval", "judgement", "outcome"):
                nid = f"R{rnd}_{suffix}"
                info = self.nodes.get(nid)
                if not info:
                    continue
                meta = info.get("metadata") or {}
                if suffix == "edit":
                    snippet = f"改 {meta.get('applied', '?')} 处"
                elif suffix == "eval":
                    snippet = f"overall={meta.get('overall')}"
                elif suffix == "judgement":
                    snippet = "；".join(map(str, meta.get("reasons") or []))[:160]
                else:
                    snippet = ""
                content = (info.get("content") or "")[:120]
                line = f"- [{info['type']}] R{rnd} {suffix}: {content}"
                if snippet:
                    line += f"（{snippet}）"
                detail.append(line)
        if detail:
            out.append("\n### 相关轮次细节（按当前任务选取）\n" + "\n".join(detail))
        return "\n".join(out)

    # ============================================================
    # 持久化
    # ============================================================
    def _save(self) -> None:
        if self.filepath:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"nodes": self.nodes, "edges": self.edges}, f,
                          ensure_ascii=False, indent=2)

    def _load(self) -> None:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", [])
        except Exception as e:
            print(f"[GraphMemory] 加载失败: {e}")
