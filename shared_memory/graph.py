# graph_memory.py — 有向图记忆系统，追踪操作间的数据依赖，支持错误溯源和子图裁剪
import json
import os
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
