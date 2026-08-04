# captain.py — PDE 科研闭环主调度器，Prog-Act 双循环，编排四个 subagent 完成自动演进
import sys, os, json, re, time, shutil, subprocess, argparse, textwrap, traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure workspace on path
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from llm_client import get_client
from shared_memory import GraphMemory, ResourcePool
from data_probe import probe_hdf5, format_probe_report

# ====================================================================
# 路径常量
# ====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(WORKSPACE, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Official Data and Sample", "data_and_sample_submission", "train_val_test_init")

TEMPLATES_DIR = os.path.join(WORKSPACE, "templates")
PROMPTS_DIR = os.path.join(WORKSPACE, "prompts")
OUTPUT_DIR = os.path.join(WORKSPACE, "output")
REPLIES_DIR = os.path.join(OUTPUT_DIR, "replies")
STATE_DIR = os.path.join(OUTPUT_DIR, "state")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
SNAPS_DIR = os.path.join(OUTPUT_DIR, "snapshots")
SUBS_DIR = os.path.join(OUTPUT_DIR, "submissions")

# ====================================================================
# 辅助：时间戳和日志
# ====================================================================
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ====================================================================
# Captain
# ====================================================================
class Captain:
    def __init__(self, task_id: int = 1,
                 train_data: str = "", val_data: str = "", test_data: str = "",
                 max_rounds: int = 5, inner_loop_limit: int = 3,
                 force_model: str = "", skip_phase0: bool = False,
                 scout_samples: int = 500, scout_topk: int = 2,
                 test_has_gt: bool = True):
        self.task_id = task_id
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.max_rounds = max_rounds
        self.inner_loop_limit = inner_loop_limit
        self.force_model = force_model
        self.skip_phase0 = skip_phase0  # 跳过训练，用已有结果
        self.scout_samples = scout_samples  # 0=跳过 scout
        self.scout_topk = scout_topk        # scout 后保留几个进全量训练
        self.test_has_gt = test_has_gt      # 测试数据是否包含 ground truth

        # LLM
        self.llm = get_client()

        # Shared memory
        os.makedirs(STATE_DIR, exist_ok=True)
        self.graph = GraphMemory(os.path.join(STATE_DIR, f"graph_t{task_id}.json"))
        self.pool = ResourcePool(os.path.join(STATE_DIR, f"pool_t{task_id}.json"))

        # Trace + Log
        self.trace: list[dict] = []
        self.log_path = os.path.join(OUTPUT_DIR, "captain.log")
        self._reply_seq: int = 0  # LLM 回复序号

        # State
        self.best_metrics: dict = {}
        self.best_model_path: str = ""
        self.best_code_dir: str = ""
        self.snapshot_dir: str = SNAPS_DIR
        self.current_round: int = 0
        self.inner_loop_count: int = 0

        # Run directory
        self.run_dir: str = ""

        # Ensure all output dirs
        for d in [OUTPUT_DIR, REPLIES_DIR, STATE_DIR, RUNS_DIR, SNAPS_DIR, SUBS_DIR]:
            os.makedirs(d, exist_ok=True)

        # Load registry
        with open(os.path.join(TEMPLATES_DIR, "registry.json"), "r") as f:
            self.registry = json.load(f)

        # Load prompts
        self.prompts = {}
        for name in ["captain", "proposal", "dev", "diagnosis", "eval_reflection"]:
            path = os.path.join(PROMPTS_DIR, f"{name}.md")
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.prompts[name] = f.read()

    # ================================================================
    # Trace
    # ================================================================
    def _trace(self, step_type: str, summary: str, detail: str = ""):
        entry = {
            "time": _now(),
            "type": step_type,   # "probe" | "llm_reasoning" | "subagent_call" | "training" | "eval" | "action"
            "summary": summary,
            "detail": detail,
        }
        self.trace.append(entry)
        prefix = {"probe": "[探测]", "llm_reasoning": "[LLM推理]", "subagent_call": "[Subagent]",
                  "training": "[训练]", "eval": "[评估]", "action": "[Captain]"}.get(step_type, "")
        line = f"[{_ts()}] {prefix} {summary}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
            if detail:
                lf.write(f"  {detail}\n")

    # ================================================================
    # Snapshot / Rollback
    # ================================================================
    def _save_snapshot(self, label: str):
        """备份当前 model.py 和最优 checkpoint 路径。"""
        snap_dir = os.path.join(self.snapshot_dir, label)
        os.makedirs(snap_dir, exist_ok=True)
        model_py = os.path.join(WORKSPACE, "model.py")
        if os.path.exists(model_py):
            shutil.copy2(model_py, os.path.join(snap_dir, "model.py"))
        # 记录 ckpt 路径
        ckpt_path = self.pool.get("best_checkpoint", "")
        with open(os.path.join(snap_dir, "checkpoint_path.txt"), "w") as f:
            f.write(ckpt_path)
        with open(os.path.join(snap_dir, "metrics.json"), "w") as f:
            json.dump(self.best_metrics, f, ensure_ascii=False, indent=2)
        self._trace("action", f"保存快照: {label}", f"ckpt={ckpt_path}")

    def _rollback(self) -> bool:
        """回退 model.py 和 checkpoint 到最优版本。"""
        if not self.best_code_dir or not os.path.exists(self.best_code_dir):
            self._trace("action", "回退失败：无快照", "")
            return False
        # 恢复 model.py
        src_model = os.path.join(self.best_code_dir, "model.py")
        if os.path.exists(src_model):
            shutil.copy2(src_model, os.path.join(WORKSPACE, "model.py"))
        # 恢复 ckpt 路径
        ckpt_txt = os.path.join(self.best_code_dir, "checkpoint_path.txt")
        if os.path.exists(ckpt_txt):
            with open(ckpt_txt, "r") as f:
                self.pool.put("best_checkpoint", f.read().strip())
        self._trace("action", "回退到最优版本", f"snapshot={self.best_code_dir}")
        return True

    # ================================================================
    # Subagent call
    # ================================================================
    def _call_subagent(self, agent_name: str, task_instruction: str) -> str:
        """调用一个 subagent，返回其文本回复。"""
        if agent_name not in self.prompts:
            return f"ERROR: no prompt for {agent_name}"

        system = self.prompts[agent_name]
        user = (
            f"=== 共享记忆 (Graph) ===\n{self.graph.format_for_llm()}\n\n"
            f"=== 资源池 (Pool) ===\n{self.pool.format_for_llm()}\n\n"
            f"=== 当前任务 ===\n{task_instruction}"
        )
        self._trace("subagent_call", f"调用 {agent_name}", task_instruction[:200])
        reply = self.llm.call(agent_name, system, user)
        if not reply:
            reply = "ERROR: empty reply"
        # 立即保存原始回复用于调试（序号命名，方便管理）
        self._reply_seq += 1
        debug_path = os.path.join(REPLIES_DIR, f"{self._reply_seq:03d}_{agent_name}.txt")
        with open(debug_path, "w", encoding="utf-8") as df:
            df.write(reply)
        return reply

    # ================================================================
    # Training
    # ================================================================
    def _run_training(self, run_label: str, extra_args: dict | None = None) -> tuple[bool, float]:
        """subprocess 执行 train.py，实时流式输出到 captain 日志。返回 (success, train_time_seconds)。"""
        train_script = os.path.join(WORKSPACE, "train.py")
        if not os.path.exists(train_script):
            self._trace("training", "train.py 不存在", "")
            return False, 0

        save_dir = os.path.join(RUNS_DIR, run_label)
        os.makedirs(save_dir, exist_ok=True)
        self._write_data_config(save_dir)
        self.run_dir = save_dir

        cmd = [
            sys.executable, train_script,
            "--train_data", self.train_data,
            "--val_data", self.val_data,
            "--save_dir", save_dir,
        ]
        if extra_args:
            for k, v in extra_args.items():
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"--{k}")
                else:
                    cmd.append(f"--{k}")
                    cmd.append(str(v))

        # 提取训练超参用于日志
        phase1_eps = extra_args.get("phase1_epochs", 50) if extra_args else 50
        phase2_eps = extra_args.get("phase2_epochs", 100) if extra_args else 100
        self._trace("training", f"开始训练: {run_label}",
                    f"Phase1={phase1_eps}eps + Phase2={phase2_eps}eps | cmd={' '.join(cmd)}")

        stdout_path = os.path.join(save_dir, "train_stdout.log")
        stderr_path = os.path.join(save_dir, "train_stderr.log")

        t0 = time.time()
        with open(stdout_path, "w") as fo, open(stderr_path, "w") as fe:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, cwd=WORKSPACE)
            # 读 stdout 和 stderr 行，实时转发
            import threading as _th
            stderr_lines: list[str] = []
            def _read_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line)
                    fe.write(line)
                    fe.flush()
            t = _th.Thread(target=_read_stderr, daemon=True)
            t.start()

            for line in proc.stdout:
                fo.write(line)
                fo.flush()
                line = line.rstrip()
                if line:
                    # 过滤出关键行：Phase 切换 + 训练指标
                    is_key = any(kw in line for kw in [
                        "Starting Phase", "Phase 1", "Phase 2",
                        "Training completed", "Best Phase",
                        "Epoch", "val_rel", "seg1", "seg2", "seg3",
                    ])
                    if is_key:
                        self._trace("training", line, "")
            proc.wait()
            t.join(timeout=5)
            for line in stderr_lines:
                fe.write(line)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            self._trace("training", f"训练失败 ({elapsed:.0f}s)", f"code={proc.returncode}")
            err = "".join(stderr_lines[-100:]) if stderr_lines else ""
            self.pool.put("last_error", err)
            return False, elapsed

        self._trace("training", f"训练完成 ({elapsed:.0f}s)", f"output={save_dir}")
        self.pool.put(f"train_{run_label}", {"time_s": elapsed, "save_dir": save_dir})
        return True, elapsed

    def _launch_training(self, run_label: str, extra_args: dict | None = None) -> dict:
        """非阻塞启动训练子进程，返回 handle 供 _wait_training 使用。"""
        train_script = os.path.join(WORKSPACE, "train.py")
        if not os.path.exists(train_script):
            self._trace("training", "train.py 不存在", "")
            return {"label": run_label, "proc": None, "error": "no train.py"}

        save_dir = os.path.join(RUNS_DIR, run_label)
        os.makedirs(save_dir, exist_ok=True)
        self._write_data_config(save_dir)

        cmd = [
            sys.executable, train_script,
            "--train_data", self.train_data,
            "--val_data", self.val_data,
            "--save_dir", save_dir,
        ]
        if extra_args:
            for k, v in extra_args.items():
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"--{k}")
                else:
                    cmd.append(f"--{k}")
                    cmd.append(str(v))

        phase1_eps = extra_args.get("phase1_epochs", 50) if extra_args else 50
        phase2_eps = extra_args.get("phase2_epochs", 100) if extra_args else 100
        self._trace("training", f"启动训练: {run_label}",
                    f"Phase1={phase1_eps}eps + Phase2={phase2_eps}eps | cmd={' '.join(cmd)}")

        stdout_path = os.path.join(save_dir, "train_stdout.log")
        stderr_path = os.path.join(save_dir, "train_stderr.log")
        fout = open(stdout_path, "w")
        ferr = open(stderr_path, "w")
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, text=True, cwd=WORKSPACE)

        return {
            "label": run_label,
            "proc": proc,
            "save_dir": save_dir,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "start_time": t0,
            "_fout": fout,
            "_ferr": ferr,
        }

    def _wait_training(self, handle: dict) -> tuple[bool, float]:
        """等待单个训练进程完成，返回 (success, elapsed)。"""
        if handle.get("error"):
            return False, 0
        proc = handle["proc"]
        proc.wait()
        handle["_fout"].close()
        handle["_ferr"].close()
        elapsed = time.time() - handle["start_time"]

        if proc.returncode != 0:
            self._trace("training", f"训练失败: {handle['label']} ({elapsed:.0f}s)",
                        f"code={proc.returncode}")
            try:
                with open(handle["stderr_path"], "r") as f:
                    err = f.read()[-2000:]
            except Exception:
                err = ""
            self.pool.put("last_error", err)
            return False, elapsed

        self._trace("training", f"训练完成: {handle['label']} ({elapsed:.0f}s)",
                    f"output={handle['save_dir']}")
        self.pool.put(f"train_{handle['label']}", {"time_s": elapsed, "save_dir": handle["save_dir"]})
        return True, elapsed

    # ================================================================
    # Phase 0: Baseline Establishment
    # ================================================================
    def _copy_template(self, model_name: str) -> bool:
        """从模板目录复制模型代码到 workspace/model.py。模板已验证正确。"""
        # 查找模板文件
        candidates = [
            os.path.join(TEMPLATES_DIR, model_name, f"{model_name}.py"),
            os.path.join(TEMPLATES_DIR, model_name, "model.py"),
        ]
        # 也尝试 registry 中记录的 file 路径
        info = self.registry.get(model_name, {})
        reg_file = info.get("file", "")
        if reg_file:
            candidates.insert(0, os.path.join(TEMPLATES_DIR, reg_file))

        for src in candidates:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(WORKSPACE, "model.py"))
                self._trace("action", f"从模板复制: {os.path.basename(src)} → model.py")
                # 如果模板有独立 utils.py，也复制
                utils_src = os.path.join(os.path.dirname(src), "utils.py")
                if os.path.exists(utils_src):
                    shutil.copy2(utils_src, os.path.join(WORKSPACE, "utils.py"))
                return True
        return False

    def _phase0_scout(self, candidates: list) -> list:
        """Scout 快速探路：逐模型生成代码→校验→训练，按 overall 排名保留 top-K。"""
        if not candidates or self.scout_samples <= 0:
            return candidates

        print(f"\n--- Phase 0.25: Scout 探路 ({self.scout_samples} samples, top-{self.scout_topk}) ---")
        scout_extra = {
            "max_samples": self.scout_samples,
            "phase1_epochs": 25,
            "phase2_epochs": 20,
            "batch_size": 8,
            "no_plateau": True,
        }

        scout_results = []
        for cand in candidates:
            model_name = cand["model"]
            info = self.registry.get(model_name, {})
            if info.get("status") == "unavailable":
                continue

            # 1) 获取模型代码（优先用模板，失败才让 Dev 生成）
            self._trace("action", f"Scout: 获取代码 — {model_name}")
            if not self._copy_template(model_name):
                task = (
                    f"Phase 0 baseline 生成。模型: {model_name}。"
                    + "从零训练，不加载任何 checkpoint。"
                    + f"模板代码在 templates/{model_name}/。"
                    + "生成 model.py，输出 mode=generate JSON。"
                )
                dev_reply = self._call_subagent("dev", task)
                self._extract_code(dev_reply)

            # 2) 语法校验 + 实例化检查（带快速修复 + 模板回退）
            model_py = os.path.join(WORKSPACE, "model.py")
            if not self._validate_python(model_py):
                self._trace("action", f"Scout: {model_name} 语法失败，尝试修复", "")
                if not self._scout_quick_fix("语法错误", model_name):
                    # 回退：重新从模板复制，再修一次
                    self._trace("action", f"Scout: {model_name} 语法修复失败，重拷模板再试", "")
                    self._copy_template(model_name)
                    if not self._scout_quick_fix("语法错误 (重新修复)", model_name):
                        continue
            if not self._instantiation_check():
                self._trace("action", f"Scout: {model_name} 实例化失败，尝试修复", "")
                if not self._scout_quick_fix("实例化/forward 失败", model_name):
                    # 回退：重新从模板复制，再修一次
                    self._trace("action", f"Scout: {model_name} 实例化修复失败，重拷模板再试", "")
                    self._copy_template(model_name)
                    if not self._scout_quick_fix("实例化/forward 失败 (重新修复)", model_name):
                        self._trace("action", f"Scout: {model_name} 重拷后仍失败，淘汰", "")
                        self.graph.add_node(f"scout_{model_name}_fail", "error",
                                            f"Scout {model_name} 实例化仍失败")
                        continue
                if not self._instantiation_check():
                    self._trace("action", f"Scout: {model_name} 修复后仍失败，淘汰", "")
                    self.graph.add_node(f"scout_{model_name}_fail", "error",
                                        f"Scout {model_name} 实例化仍失败")
                    continue

            # 3) 立即训练（串行，避免 model.py 被后续模型覆盖）
            self._trace("action", f"Scout: 训练 — {model_name}")
            ok, elapsed = self._run_training(f"scout_{model_name}", scout_extra)
            if not ok:
                self._trace("action", f"Scout: {model_name} 训练失败，淘汰", "")
                self.graph.add_node(f"scout_{model_name}_fail", "error",
                                    f"Scout {model_name} 训练失败")
                continue

            metrics = self._read_metrics(f"scout_{model_name}")
            if metrics:
                scout_results.append({"model": model_name, "metrics": metrics,
                                      "priority": cand.get("priority", 99)})
                self._trace("eval", f"Scout {model_name}: "
                            f"overall={metrics['overall']:.4f} "
                            f"seg1={metrics['seg1']:.4f} seg2={metrics['seg2']:.4f} seg3={metrics['seg3']:.4f}")
                self.pool.put(f"scout_{model_name}", metrics)
            else:
                self._trace("action", f"Scout: {model_name} 训练完成但无指标", "")

        if not scout_results:
            self._trace("action", "Scout: 全部训练失败，回退全量候选", "")
            return candidates

        # 按比赛加权分排名（seg1 exp(-20×) 极度敏感，整体平均会被 seg3 主导）
        import math as _math
        def _scout_score(m):
            s1 = 100 * _math.exp(-20 * m.get('seg1', 999))
            s2 = 100 * _math.exp(-10 * m.get('seg2', 999))
            s3 = 100 / (1 + 10 * m.get('seg3', 999))
            return 0.25 * s1 + 0.25 * s2 + 0.50 * s3
        ranked = sorted(scout_results, key=lambda r: _scout_score(r["metrics"]), reverse=True)
        topk = ranked[:self.scout_topk]

        print(f"\n📊 Scout 排名 (比赛加权分):")
        for i, r in enumerate(ranked):
            m = r["metrics"]
            flag = "✅" if i < self.scout_topk else "❌"
            score = _scout_score(m)
            print(f"  {flag} {i+1}. {r['model']}: score={score:.1f} "
                  f"(seg1={m['seg1']:.4f} seg2={m['seg2']:.4f} seg3={m['seg3']:.4f})")

        # 返回 top-K 候选（保持原始 priority 信息）
        topk_models = {r["model"] for r in topk}
        filtered = [c for c in candidates if c["model"] in topk_models]
        self._trace("action", f"Scout 完成: 保留 {[c['model'] for c in filtered]}，进入全量训练")
        return filtered

    def _scout_quick_fix(self, error_summary: str, model_name: str) -> bool:
        """Scout 中的快速修复：Diagnosis → Dev fix，最多 3 次。"""
        for attempt in range(3):
            self._trace("action", f"Scout 修复 {model_name} ({attempt+1}/3): {error_summary}")
            err_log = self.pool.get("last_error", "")
            current_model = self._read_current_model()
            diag_task = (
                f"Scout 阶段快速修复。错误: {error_summary}\n"
                f"错误日志:\n```\n{err_log[-1500:]}\n```\n"
                f"当前 model.py:\n```python\n{current_model}\n```\n"
            )
            diag_reply = self._call_subagent("diagnosis", diag_task)
            diag = self._parse_json_block(diag_reply)
            if diag and diag.get("needs_proposal"):
                return False
            fix = diag.get("fix", {}) if diag else {}
            fix_task = (
                f"Scout 快速修复。分析: {json.dumps(diag, ensure_ascii=False) if diag else 'N/A'}\n"
                f"精确修复: {json.dumps(fix, ensure_ascii=False)}\n"
                f"当前 model.py:\n```python\n{current_model}\n```\n"
                "输出 mode=edit JSON。"
            )
            dev_reply = self._call_subagent("dev", fix_task)
            self._apply_edits(dev_reply)
            if self._validate_python(os.path.join(WORKSPACE, "model.py")):
                # 语法通过还不够，必须能实例化 + forward
                if self._instantiation_check():
                    return True
                self._trace("action", f"Scout 修复 {model_name} ({attempt+1}/3): 语法通过但实例化失败，继续修复")
        return False

    def _build_data_config(self) -> dict:
        """从数据探测结果构建 data_config.json 内容。"""
        probe = self.pool.get("data_probe", {})
        if not probe:
            return {}
        dims = probe.get("dimensions", {})
        # 用 shape 最后一维作为空间分辨率（PDE 数据标准 N×T×X 格式）
        probe_shape = probe.get("shape", [])
        n_spatial = probe_shape[-1] if len(probe_shape) >= 3 else dims.get("n_spatial", 256)
        n_timesteps = probe_shape[1] if len(probe_shape) >= 2 else dims.get("n_timesteps", 200)

        # 确定空间降采样（目标分辨率 ~128-256）
        if n_spatial <= 256:
            subsample = 1
        elif n_spatial <= 512:
            subsample = 2
        else:
            subsample = 4

        # 检测训练数据中的额外字段
        condition_fields = []
        h5_key = "tensor"
        if os.path.exists(self.train_data):
            with __import__("h5py").File(self.train_data, "r") as f:
                all_keys = list(f.keys())
                # 找数据键
                for candidate in ["tensor", "data", "u", "sol"]:
                    if candidate in all_keys:
                        h5_key = candidate
                        break
                # 检测条件字段（可变物理参数）
                for field in ["nu", "mu", "alpha", "beta", "gamma", "Re"]:
                    if field in all_keys and isinstance(f[field], __import__("h5py").Dataset):
                        # 检查是否每个样本值不同（可变参数）
                        vals = f[field][:min(100, f[field].shape[0])]
                        if vals.ndim == 1 and vals.std() > 1e-8:
                            condition_fields.append(field)

        # 检测多文件训练数据（part0, part1, ...）
        extra_train_files = []
        train_dir = os.path.dirname(self.train_data)
        train_base = os.path.basename(self.train_data)
        if "part0" in train_base:
            for i in range(1, 10):
                extra_path = os.path.join(train_dir, train_base.replace("part0", f"part{i}"))
                if os.path.exists(extra_path):
                    extra_train_files.append(extra_path)
                else:
                    break

        config = {
            "h5_key": h5_key,
            "spatial_subsample": subsample,
            "t_in": 10,
            "n_spatial": n_spatial // subsample if subsample > 1 else n_spatial,
            "total_timesteps": n_timesteps,
            "extra_train_files": extra_train_files,
            "condition_fields": condition_fields,
            "test_has_gt": getattr(self, "test_has_gt", True),
        }
        self._trace("action", f"数据配置: h5_key={h5_key}, subsample={subsample}, "
                    f"n_spatial={config['n_spatial']}, t_in={config['t_in']}, "
                    f"extra_files={len(extra_train_files)}, conditions={condition_fields}")
        return config

    def _write_data_config(self, save_dir: str):
        """在训练目录写入 data_config.json，供 train.py 读取。"""
        cfg = getattr(self, "data_config", None)
        if not cfg:
            return
        cfg_path = os.path.join(save_dir, "data_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def _phase0_baseline(self) -> bool:
        """Step 0.1-0.5: 探测数据 → 匹配模型 → 跑 baseline → 选最优。"""
        print(f"\n{'='*60}\nPhase 0: Baseline Establishment\n{'='*60}")

        # --- Step 0.1: 探测数据 ---
        self._trace("action", "Phase 0.1: 数据探测")
        data_reports = []
        for fpath in [self.train_data, self.val_data]:
            if fpath and os.path.exists(fpath):
                r = probe_hdf5(fpath)
                data_reports.append(r)
                self._trace("probe", f"数据: {r['file']}",
                            f"shape={r['shape']}, freq={r.get('frequency',{}).get('interpretation','?')}, "
                            f"bc={r.get('boundary',{}).get('type','?')}")
        # 写入 Pool
        if data_reports:
            self.pool.put("data_probe", data_reports[0])
        # 构建数据自适应配置（train.py 启动时读取）
        self.data_config = self._build_data_config()

        # --- Step 0.2: Captain LLM 推理 — 匹配模型 ---
        self._trace("action", "Phase 0.2: Captain LLM 推理匹配模型")
        context = (
            f"### 数据探测报告\n{format_probe_report(data_reports, [])}\n\n"
            f"### Registry\n```json\n{json.dumps(self.registry, ensure_ascii=False, indent=2)}\n```\n\n"
            f"### 任务\n"
            f"🔴 本次是 Phase 0 模型匹配，不是标准决策流程。不要调用 subagent，不要输出 action/target。\n"
            f"根据以上探测结果和 registry 中的模型物理特征，列出候选模型及其优先级。所有模型从零训练，不依赖 checkpoint。\n"
            '严格输出以下 JSON（不要嵌套在 memory_update 里）:\n'
            '```json\n{"candidates": [{"model": "fno", "priority": 1, "reason": "..."}]}\n```'
        )
        reply = self._call_subagent("captain", context)
        candidates = self._parse_candidates(reply)

        if not candidates:
            # Fallback: all available models
            candidates = [{"model": m, "priority": i+1, "reason": "fallback"}
                          for i, m in enumerate(self.registry)
                          if isinstance(self.registry[m], dict) and self.registry[m].get("status") != "unavailable"]

        self._trace("llm_reasoning", f"候选模型: {[c['model'] for c in candidates]}")

        # --- Step 0.25: Scout 快速探路（如果启用）---
        if self.scout_samples > 0:
            candidates = self._phase0_scout(candidates)
            if not candidates:
                self._trace("action", "Scout 后无剩余候选，终止", "")
                return False

        # --- Step 0.3: 为每个候选生成代码并训练 ---
        results = []
        for cand in candidates:
            model_name = cand["model"]
            info = self.registry.get(model_name, {})
            if info.get("status") == "unavailable":
                continue

            self._trace("action", f"Phase 0.3: 跑 baseline — {model_name}")

            # 添加图节点
            self.graph.add_node(f"phase0_{model_name}", "baseline",
                                f"Baseline: {model_name}, priority={cand['priority']}")

            # 调 Dev 生成代码 — 统一从零训练
            # 获取模型代码（优先模板，失败才 Dev 生成）
            if not self._copy_template(model_name):
                task = (
                    f"Phase 0 baseline 生成。模型: {model_name}。"
                    + "从零训练，不加载任何 checkpoint。"
                    + f"模板代码在 templates/{model_name}/。"
                    + "生成 model.py，输出 mode=generate JSON。"
                )
                dev_reply = self._call_subagent("dev", task)
                self._extract_code(dev_reply)

            # 语法校验
            if not self._validate_python(os.path.join(WORKSPACE, "model.py")):
                self._trace("action", f"{model_name} 语法校验失败，跳过", "")
                results.append({"model": model_name, "error": "syntax"})
                continue
            # 训练（最多重试 2 次）
            ok, train_time = False, 0
            for retry in range(3):
                ok, train_time = self._run_training(f"phase0_{model_name}")
                if ok:
                    break
                self._trace("action", f"{model_name} 训练失败 (尝试 {retry+1}/3)，调用 Diagnosis 修复")
                err_log = self.pool.get("last_error", "")
                diag_task = (
                    f"训练崩溃。错误日志:\n```\n{err_log[-1500:]}\n```\n"
                    f"当前 model.py:\n```python\n{self._read_current_model()}\n```\n"
                )
                diag_reply = self._call_subagent("diagnosis", diag_task)
                diag = self._parse_json_block(diag_reply)
                fix = diag.get("fix", {}) if diag else {}
                fix_task = (
                    f"修复训练崩溃。错误分析: {json.dumps(diag, ensure_ascii=False) if diag else 'N/A'}\n"
                    f"精确修复: {json.dumps(fix, ensure_ascii=False)}\n"
                    "输出 mode=edit JSON。"
                )
                dev_reply = self._call_subagent("dev", fix_task)
                self._apply_edits(dev_reply)
                # 清理上次失败的 run 目录
                import shutil as _sh
                run_dir = os.path.join(RUNS_DIR, f"phase0_{model_name}")
                if os.path.exists(run_dir):
                    _sh.rmtree(run_dir)
            if not ok:
                results.append({"model": model_name, "error": "training_failed",
                                "train_time": train_time})
                self.graph.add_node(f"phase0_{model_name}_fail", "error",
                                    f"{model_name} 训练失败 (3次重试后)")
                continue

            # 读取指标
            metrics = self._read_metrics(f"phase0_{model_name}")
            results.append({"model": model_name, "metrics": metrics, "train_time": train_time})
            self.graph.add_node(f"phase0_{model_name}_done", "eval",
                                f"{model_name}: {json.dumps(metrics)}")
            if metrics:
                self.pool.put(f"baseline_{model_name}", metrics)

        # --- Step 0.4: 选最优 ---
        self._trace("action", "Phase 0.4: 汇总 baseline 排行")
        valid_results = [r for r in results if "metrics" in r and r["metrics"]]

        # 手动指定 baseline
        if self.force_model:
            forced = [r for r in valid_results if r["model"] == self.force_model]
            if forced:
                ranked = forced + [r for r in valid_results if r["model"] != self.force_model]
                self._trace("action", f"手动指定 baseline: {self.force_model}", "")
            else:
                self._trace("action", f"警告: --model {self.force_model} 未训练成功，回退自动排行", "")
                ranked = sorted(valid_results, key=lambda r: r["metrics"].get("overall", 999))
        else:
            ranked = sorted(valid_results, key=lambda r: r["metrics"].get("overall", 999))
        if not ranked:
            print("[Captain] Phase 0 失败：无一模型成功训练")
            return False

        best = ranked[0]
        self.best_metrics = best["metrics"]
        self.best_model_path = self.pool.get(f"train_phase0_{best['model']}.save_dir", "")
        self.pool.put("best_model", best["model"])
        self.pool.put("best_metrics", self.best_metrics)
        self.pool.put("best_checkpoint",
                      os.path.join(self.best_model_path, "best_phase2.pt") if self.best_model_path else "")

        # 保存快照作为初始最优
        self.best_code_dir = os.path.join(self.snapshot_dir, f"phase0_{best['model']}")
        self._save_snapshot(f"phase0_{best['model']}")

        self._trace("action",
                    f"🏆 Baseline 最优: {best['model']}, "
                    f"seg1={self.best_metrics.get('seg1','?')}, "
                    f"seg2={self.best_metrics.get('seg2','?')}, "
                    f"seg3={self.best_metrics.get('seg3','?')}")

        # 打印排行榜
        print("\n📊 Baseline 排行榜:")
        for i, r in enumerate(ranked):
            m = r["metrics"]
            flag = "🏆" if i == 0 else "  "
            print(f"  {flag} {r['model']}: seg1={m.get('seg1','?'):.4f} "
                  f"seg2={m.get('seg2','?'):.4f} seg3={m.get('seg3','?'):.4f} "
                  f"time={r.get('train_time',0):.0f}s")

        return True

    # ================================================================
    # Phase 0 (skip): 从已有训练结果恢复 baseline
    # ================================================================
    def _phase0_from_existing(self) -> bool:
        """跳过训练，直接从 output/runs/ 读取已有指标，排行选最优。"""
        print(f"\n{'='*60}\nPhase 0: 从已有结果恢复 (--skip-phase0)\n{'='*60}")
        results = []
        for model_name in self.registry:
            info = self.registry.get(model_name, {})
            if not isinstance(info, dict) or info.get("status") == "unavailable":
                continue
            run_dir = os.path.join(RUNS_DIR, f"phase0_{model_name}")
            if not os.path.isdir(run_dir):
                continue
            metrics = self._read_metrics(f"phase0_{model_name}")
            if metrics:
                results.append({"model": model_name, "metrics": metrics})
                self._trace("action", f"恢复 {model_name}: RelMSE={metrics['overall']:.4f} seg3={metrics['seg3']:.4f}", "")
                self.pool.put(f"baseline_{model_name}", metrics)
                self.graph.add_node(f"phase0_{model_name}_done", "eval",
                                    f"{model_name}: {json.dumps(metrics)}")

        if not results:
            print("[Captain] 无已有训练结果，请先跑 Phase 0")
            return False

        # 排行
        self._trace("action", "Phase 0.4: 汇总 baseline 排行")
        if self.force_model:
            forced = [r for r in results if r["model"] == self.force_model]
            ranked = forced + [r for r in results if r["model"] != self.force_model] if forced else sorted(results, key=lambda r: r["metrics"].get("overall", 999))
        else:
            ranked = sorted(results, key=lambda r: r["metrics"].get("overall", 999))

        best = ranked[0]
        self.best_metrics = best["metrics"]
        self.pool.put("best_model", best["model"])
        self.pool.put("best_metrics", self.best_metrics)
        self.pool.put("best_checkpoint",
                      os.path.join(RUNS_DIR, f"phase0_{best['model']}", "best_phase2.pt"))
        self.best_code_dir = os.path.join(SNAPS_DIR, f"phase0_{best['model']}")

        for i, r in enumerate(ranked):
            m = r["metrics"]
            flag = "🏆" if i == 0 else "  "
            line = f"  {flag} {i+1}. {r['model']}: RelMSE={m['overall']:.4f} seg1={m['seg1']:.4f} seg2={m['seg2']:.4f} seg3={m['seg3']:.4f}"
            self._trace("action", line, "")
            if i == 0:
                self._trace("action", "     → 选为 baseline，进入 Phase 1+ evolution", "")
                self._save_snapshot(f"phase0_{best['model']}")

        return True

    # ================================================================
    # Phase 1+: Evolution Loop
    # ================================================================
    def _phase1_evolution(self):
        """Prog-Act 双循环进化。"""
        print(f"\n{'='*60}\nPhase 1+: Evolution Loop (max {self.max_rounds} rounds)\n{'='*60}")

        for rnd in range(1, self.max_rounds + 1):
            self.current_round = rnd
            self.inner_loop_count = 0
            print(f"\n--- Round {rnd}/{self.max_rounds} ---")

            # 1) Proposal
            self._trace("action", f"Round {rnd}: 调度 Proposal")
            proposal_task = (
                f"当前最优模型: {self.pool.get('best_model')}, "
                f"指标: {json.dumps(self.best_metrics)}。"
                f"瓶颈: 分析 seg1/seg2/seg3 哪个是短板，提出 2-3 个改进假说。"
            )
            prop_reply = self._call_subagent("proposal", proposal_task)
            hypotheses = self._parse_json_block(prop_reply)
            if not hypotheses or "hypotheses" not in hypotheses:
                self._trace("action", "Proposal 无有效输出，跳过本轮", "")
                continue

            # 取 P0 假说
            hyps = sorted(hypotheses.get("hypotheses", []),
                          key=lambda h: 0 if h.get("priority") == "P0" else 1)
            if not hyps:
                continue
            selected = hyps[0]
            self._trace("llm_reasoning", f"选中假说: [{selected.get('priority','')}] {selected.get('title','')}")

            self.graph.add_node(f"R{rnd}_hypothesis", "hypothesis",
                                f"[{selected.get('priority','')}] {selected.get('title','')}",
                                selected)

            # 2) 保存快照
            self._save_snapshot(f"round{rnd}_pre")

            # 3) Dev — 传完整 model.py，确保 old 字符串精确匹配
            dev_task = (
                f"假说: {selected.get('title')}\n"
                f"修改目标: {json.dumps(selected.get('target', {}), ensure_ascii=False)}\n"
                f"伪代码: {selected.get('target', {}).get('pseudo_code', 'N/A')}\n"
                f"当前 model.py:\n```python\n{self._read_current_model()}\n```\n"
                "🔴 edit 的 old 字符串必须从以上代码中逐字复制（含缩进），不能自己编造。"
                "输出 mode=edit JSON，仅修改假说指定的行。"
            )
            dev_reply = self._call_subagent("dev", dev_task)
            self._apply_edits(dev_reply)

            # 语法校验 + 实例化检查
            if not self._validate_python(os.path.join(WORKSPACE, "model.py")):
                self._trace("action", "语法校验失败，触发内循环修复", "")
                if not self._inner_loop_fix("语法错误"):
                    continue
            if not self._instantiation_check():
                self._trace("action", "实例化检查失败，触发内循环修复", "")
                if not self._inner_loop_fix("实例化/forward 失败"):
                    continue

            # 4) 训练（带内循环修复 + 回退保护）
            ok, train_time = self._run_training(f"round{rnd}")
            if not ok:
                self.graph.add_node(f"R{rnd}_fail", "error", f"Round {rnd} 训练失败")
                if self._inner_loop_fix("训练失败"):
                    # 修复成功 → 重新训练，验证修复是否真的解决了问题
                    self._trace("action", f"Round {rnd} 内循环修复完成，重新训练验证")
                    ok, train_time = self._run_training(f"round{rnd}")
                if not ok:
                    self._trace("action", f"Round {rnd} 训练失败无法修复，回退 model.py 到最优版本")
                    self._rollback()
                    continue

            # 5) EvalReflection
            current_metrics = self._read_metrics(f"round{rnd}")
            if not current_metrics:
                self._trace("action", "无法读取指标，回退 model.py 到最优版本")
                self._rollback()
                continue

            self.graph.add_node(f"R{rnd}_eval", "eval", json.dumps(current_metrics))
            eval_task = (
                f"Baseline指标: {json.dumps(self.best_metrics)}\n"
                f"本轮指标: {json.dumps(current_metrics)}\n"
                f"假说: {selected.get('title')}\n"
                f"训练时间: {train_time:.0f}s\n"
                "判定 IMPROVED/DEGRADED/MIXED/STAGNANT，输出 JSON。"
            )
            eval_reply = self._call_subagent("eval_reflection", eval_task)
            verdict = self._parse_json_block(eval_reply)

            # 6) 根据判定行动
            v = verdict.get("verdict", "STAGNANT") if verdict else "STAGNANT"
            self._trace("eval", f"Round {rnd} 判定: {v}",
                        json.dumps(verdict, ensure_ascii=False) if verdict else "")

            if v == "IMPROVED":
                self.best_metrics = current_metrics
                self.best_code_dir = os.path.join(self.snapshot_dir, f"round{rnd}_best")
                self._save_snapshot(f"round{rnd}_best")
                self.graph.add_node(f"R{rnd}_improved", "milestone",
                                    f"Round {rnd} 改进！新最优。")
                self.pool.put("best_metrics", self.best_metrics)
                print(f"  ✅ IMPROVED — 更新 baseline")

            elif v == "DEGRADED":
                self.graph.add_node(f"R{rnd}_degraded", "milestone",
                                    f"Round {rnd} 退化，回退。")
                print(f"  ❌ DEGRADED — 回退到最优版本")
                self._rollback()
                # 外循环：Proposal 重新出方案
                self._trace("action", "触发外循环 — Proposal 基于最优版本重新出方案", "")

            elif v == "MIXED":
                print(f"  ⚠️ MIXED — 回退 model.py，保留指标供后续 Proposal 参考")
                self._rollback()
                self.graph.add_node(f"R{rnd}_mixed", "milestone",
                                    f"Round {rnd} 混合 — 部分改善部分退化，已回退到最优代码")

            else:  # STAGNANT
                print(f"  ⏸️ STAGNANT — 无显著变化")
                self.graph.add_node(f"R{rnd}_stagnant", "milestone", f"Round {rnd} 无变化")

        print(f"\n{'='*60}\nEvolution complete.\n{'='*60}")

    # ================================================================
    # 内循环修复
    # ================================================================
    def _inner_loop_fix(self, error_summary: str) -> bool:
        """内循环：Diagnosis → Dev 修复，最多 inner_loop_limit 次。"""
        err_log = self.pool.get("last_error", "")
        for attempt in range(self.inner_loop_limit):
            self.inner_loop_count += 1
            self._trace("action", f"内循环修复 ({attempt+1}/{self.inner_loop_limit})")

            # Diagnosis
            diag_task = (
                f"错误: {error_summary}\n"
                f"错误日志:\n```\n{err_log[-2000:]}\n```\n"
                f"当前 model.py:\n```python\n{self._read_current_model()}\n```\n"
            )
            diag_reply = self._call_subagent("diagnosis", diag_task)
            diag = self._parse_json_block(diag_reply)

            if diag and diag.get("needs_proposal"):
                self._trace("action", "Diagnosis 判定需要 Proposal 重出方案", "")
                return False  # 升级到外循环

            # Dev 修复
            fix = diag.get("fix", {}) if diag else {}
            fix_task = (
                f"修复模式。错误分析: {json.dumps(diag, ensure_ascii=False) if diag else 'N/A'}\n"
                f"精确修复: {json.dumps(fix, ensure_ascii=False)}\n"
                f"当前 model.py:\n```python\n{self._read_current_model()}\n```\n"
                "🔴 old 字符串必须从以上代码逐字复制。输出 mode=edit JSON。"
            )
            dev_reply = self._call_subagent("dev", fix_task)
            self._apply_edits(dev_reply)

            if self._validate_python(os.path.join(WORKSPACE, "model.py")):
                # 语法通过还不够，必须能实例化 + forward 才算修复成功
                if self._instantiation_check():
                    return True
                self._trace("action", f"内循环修复 ({attempt+1}): 语法通过但实例化/forward 失败，继续修复")

        self._trace("action", f"内循环耗尽 ({self.inner_loop_limit}次)，升级外循环", "")
        return False

    # ================================================================
    # 入口
    # ================================================================
    def run(self):
        print(f"🚀 Captain 启动 — Task {self.task_id}")
        self._trace("action", f"Task {self.task_id} 启动", f"train={self.train_data}, val={self.val_data}")

        # Phase 0
        if self.skip_phase0:
            ok = self._phase0_from_existing()
        else:
            ok = self._phase0_baseline()
        if not ok:
            self._generate_logs()
            return

        # Phase 1+
        if self.max_rounds > 0:
            self._phase1_evolution()

        self._generate_logs()
        print(f"\n📄 日志已保存: output/task{self.task_id}_report.log")

    # ================================================================
    # 辅助方法
    # ================================================================
    def _read_current_model(self) -> str:
        mp = os.path.join(WORKSPACE, "model.py")
        if os.path.exists(mp):
            with open(mp, "r") as f:
                return f.read()
        return ""

    def _validate_python(self, fpath: str) -> bool:
        try:
            with open(fpath, "r") as f:
                __import__("ast").parse(f.read())
            return True
        except SyntaxError as e:
            self._trace("action", f"语法错误: {fpath}:{e.lineno}: {e.msg}", "")
            return False

    def _instantiation_check(self) -> bool:
        """快速检验 model.py：import + 实例化 + dummy forward。耗时 <2s。"""
        check_script = """
import sys; sys.path.insert(0, '.')
import torch
import model as m
classes = [v for v in vars(m).values()
           if isinstance(v, type) and issubclass(v, torch.nn.Module)
           and v.__module__ == 'model']
for name in ['UNet1d', 'FNO1d', 'DeepONet1d', 'MGNO1d', 'Model']:
    obj = getattr(m, name, None)
    if obj is not None and isinstance(obj, type) and obj not in classes:
        classes.append(obj)
if not classes:
    print('FAIL: no nn.Module class found')
    exit(1)
# Try each class until one instantiates
model = None
ctor_patterns = [
    lambda c: c(),
    lambda c: c(modes=12, width=32, in_channels=10, out_channels=1),
    lambda c: c(modes=12, width=32, in_channels=10, out_channels=1, n_layers=4),
    lambda c: c(modes=12, width=32, in_channels=10, out_channels=1, n_layers=4, n_scales=3),
    lambda c: c(in_channels=10, out_channels=1, base_width=32, depth=4),
    lambda c: c(in_channels=10, out_channels=1, init_features=32),
    lambda c: c(num_channels=1, modes=12, width=32, initial_step=10),
    lambda c: c(in_channels=10, width=32, n_layers=4, n_basis=128, nx=256),
]
for cls in classes:
    for pat in ctor_patterns:
        try:
            model = pat(cls)
            break
        except (TypeError, ValueError):
            continue
    if model is not None:
        break
if model is None:
    print('FAIL: could not instantiate any class')
    exit(1)
x = torch.randn(2, 256, 10)
try:
    y = model(x)
except TypeError:
    grid = torch.linspace(0, 1, 256).view(1, 256, 1).repeat(2, 1, 1)
    y = model(x, grid)
if isinstance(y, tuple):
    y = y[0]
if y.dim() < 2 or y.shape[0] != 2:
    print(f'FAIL: bad output shape {list(y.shape)}')
    exit(1)
print(f'OK: {cls.__name__} input={list(x.shape)} output={list(y.shape)}')
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True, text=True, timeout=30, cwd=WORKSPACE,
            )
            ok = proc.returncode == 0 and "OK:" in proc.stdout
            self._trace("action",
                        f"实例化检查: {'PASS' if ok else 'FAIL'}",
                        proc.stdout.strip()[:200] if ok else proc.stderr.strip()[:200])
            return ok
        except subprocess.TimeoutExpired:
            self._trace("action", "实例化检查: TIMEOUT", "")
            return False
        except Exception as e:
            self._trace("action", f"实例化检查: ERROR {e}", "")
            return False

    def _read_metrics(self, run_label: str) -> dict | None:
        """从 train_stdout.log 提取最后的指标行。"""
        log_path = os.path.join(RUNS_DIR, run_label, "train_stdout.log")
        if not os.path.exists(log_path):
            return None
        with open(log_path, "r") as f:
            lines = f.readlines()
        # 找最后一行含指标的行（支持 seg1=/S1: 两种格式）
        import re as _re
        for line in reversed(lines):
            # 格式1: RelMSE=0.45 [seg1=0.004 seg2=0.15 seg3=0.45]
            # 格式2: Val Rel-MSE: 0.7415 | S1:0.5499 S2:0.6635 S3:0.8757
            m_overall = _re.search(r'(?:RelMSE|Val Rel-MSE|val_rel)\s*[=:]?\s*([\d.]+|inf|nan)', line)
            m1 = _re.search(r'(?:seg1|S1)\s*[=:]?\s*([\d.]+|inf|nan)', line)
            m2 = _re.search(r'(?:seg2|S2)\s*[=:]?\s*([\d.]+|inf|nan)', line)
            m3 = _re.search(r'(?:seg3|S3)\s*[=:]?\s*([\d.]+|inf|nan)', line)
            if m_overall and m1 and m2 and m3:
                try:
                    return {
                        "overall": float(m_overall.group(1)) if m_overall.group(1) not in ("inf", "nan") else 999,
                        "seg1": float(m1.group(1)) if m1.group(1) not in ("inf", "nan") else 999,
                        "seg2": float(m2.group(1)) if m2.group(1) not in ("inf", "nan") else 999,
                        "seg3": float(m3.group(1)) if m3.group(1) not in ("inf", "nan") else 999,
                    }
                except ValueError:
                    pass
        return None

    def _extract_code(self, reply: str):
        """从 LLM 回复中提取代码并写入 model.py / train.py。

        策略优先级:
        1. Markdown 代码块 (```python ... ```) — 优先，最可靠
        2. JSON 的 mode=generate → 从 files 字段提取（代码必须已正确转义）
        3. 裸代码回退
        """
        saved = []
        model_py = os.path.join(WORKSPACE, "model.py")
        # train.py is a permanent framework file — never overwrite from LLM output
        train_py = os.devnull

        # 防御：空/错误回复
        if not reply or reply.startswith("ERROR:"):
            self._trace("action", f"代码提取失败: 无效回复", f"reply={reply[:200] if reply else '(empty)'}")
            return

        # --- 策略 1: Markdown 代码块（优先 — LLM 最可靠地输出）---
        # 注意：必须要求语言标记，否则会把 ```json 的闭合 ``` 误当代码块开头
        blocks = re.findall(r'```(?:python|python3)\s*\n(.*?)\n\s*```', reply, re.DOTALL)
        for block in blocks:
            content = block.strip()
            if len(content) < 100:
                continue
            first_line = content.split('\n')[0].strip()
            clean = content  # default: use as-is
            # 检测文件标记
            if re.search(r'(?:^#\s*|^//\s*)?model\.py|=== model\.py|文件[：:]\s*model', first_line, re.IGNORECASE):
                clean = '\n'.join(content.split('\n')[1:]) if first_line.startswith('#') else content
                with open(model_py, "w") as f:
                    f.write(clean.strip() + "\n")
                saved.append("model.py")
            elif re.search(r'(?:^#\s*|^//\s*)?train\.py|=== train\.py|文件[：:]\s*train', first_line, re.IGNORECASE):
                clean = '\n'.join(content.split('\n')[1:]) if first_line.startswith('#') else content
                with open(train_py, "w") as f:
                    f.write(clean.strip() + "\n")
                saved.append("train.py")
            elif 'import torch' in content or 'import argparse' in content or 'from torch' in content:
                # 按内容推断
                if any(kw in content for kw in ['DataLoader', 'BurgersDataset', 'argparse', 'train_data']):
                    if 'train.py' not in saved:
                        with open(train_py, "w") as f:
                            f.write(clean.strip() + "\n")
                        saved.append("train.py")
                else:
                    if 'model.py' not in saved:
                        with open(model_py, "w") as f:
                            f.write(clean.strip() + "\n")
                        saved.append("model.py")

        # --- 策略 2: 按内容推断未标记的代码块 ---
        if 'model.py' not in saved or 'train.py' not in saved:
            for blk in blocks:
                content = blk.strip()
                if len(content) < 100:
                    continue
                if 'class FNO' in content or 'class UNet' in content or 'SpectralConv' in content:
                    if 'model.py' not in saved:
                        with open(model_py, "w") as f:
                            f.write(content + "\n")
                        saved.append("model.py")
                if ('argparse' in content and 'train_data' in content) or 'BurgersDataset' in content:
                    if 'train.py' not in saved:
                        with open(train_py, "w") as f:
                            f.write(content + "\n")
                        saved.append("train.py")

        # --- 策略 3: 长度回退（没识别出文件名时按长度分配）---
        if not saved and len(blocks) >= 2:
            blocks_sorted = sorted(blocks, key=len, reverse=True)
            with open(model_py, "w") as f:
                f.write(blocks_sorted[0].strip() + "\n")
            saved.append("model.py")
            with open(train_py, "w") as f:
                f.write(blocks_sorted[1].strip() + "\n")
            saved.append("train.py")

        # --- 策略 4: JSON mode=generate ---
        if not saved:
            parsed = self._parse_json_block(reply)
            if parsed and parsed.get("mode") == "generate":
                files = parsed.get("files", {})
                for fname, code in files.items():
                    fpath = os.path.join(WORKSPACE, fname)
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    # JSON 里的代码可能把真换行存成了 \\n 字符串
                    if isinstance(code, str) and '\\n' in code and code.count('\n') < 5:
                        code = code.replace('\\n', '\n')
                    with open(fpath, "w") as f:
                        f.write(code.strip() + "\n")
                    saved.append(fname)

        self._trace("action", f"代码提取: {saved}", f"reply={len(reply)} chars, blocks={len(blocks)}")
        if not saved:
            # 保存失败的回复供手动检查
            fail_path = os.path.join(REPLIES_DIR, "_extract_failed.txt")
            with open(fail_path, "w", encoding="utf-8") as ff:
                ff.write(reply)
            self._trace("action", f"代码提取完全失败! 原始回复已保存到 {fail_path}", f"reply前200字: {reply[:200]}")

    def _apply_edits(self, reply: str):
        """应用 Dev 的 edit 模式输出。"""
        parsed = self._parse_json_block(reply)
        if not parsed or parsed.get("mode") != "edit":
            # 可能是 generate 模式或纯文本
            self._extract_code(reply)
            return

        for edit in parsed.get("edits", []):
            fname = edit.get("file", "model.py")
            fpath = os.path.join(WORKSPACE, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r") as f:
                code = f.read()
            for change in edit.get("changes", []):
                old = change.get("old", "")
                new = change.get("new", "")
                if old and old in code:
                    code = code.replace(old, new, 1)
                    self._trace("action", f"编辑 {fname}: ✓ {old[:40].strip()} → {new[:40].strip()}",
                                change.get("reason", ""))
                elif old and old.strip() in code:
                    # 尝试去掉首尾空白匹配
                    code = code.replace(old.strip(), new.strip(), 1)
                    self._trace("action", f"编辑 {fname}: ~ (strip) {old[:40].strip()}", "")
                else:
                    # 显示附近代码帮助调试
                    self._trace("action", f"编辑 {fname}: ✗ 未找到 — {old[:60].strip()}",
                                f"文件不存在该字符串，已跳过")
            with open(fpath, "w") as f:
                f.write(code)

    def _parse_json_block(self, text: str) -> dict | None:
        """从 LLM 回复中提取第一个 JSON 块。"""
        import re as _re
        m = _re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        json_str = m.group(1) if m else text
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试找 { ... } 块
            m2 = _re.search(r'\{.*\}', text, re.DOTALL)
            if m2:
                try:
                    return json.loads(m2.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    def _parse_candidates(self, text: str) -> list:
        parsed = self._parse_json_block(text)
        if parsed:
            cands = parsed.get("candidates", [])
            if cands:
                return cands
            # Fallback 1: captain standard format may embed candidates in pool or memory_update
            pool = parsed.get("pool", parsed.get("memory_update", {}).get("pool", {}))
            cands = pool.get("phase0_candidates", pool.get("candidates", []))
            if isinstance(cands, list) and cands:
                return [{"model": m, "priority": i+1, "reason": "from_pool"}
                        for i, m in enumerate(cands) if isinstance(m, str)]
        return []

    # ================================================================
    # 日志生成
    # ================================================================
    def _generate_logs(self):
        """生成符合比赛格式的任务日志。"""
        log_path = os.path.join(OUTPUT_DIR, f"task{self.task_id}_report.log")
        lines = [
            f"{'='*60}",
            f"      Agent 科研实验日志 — Task {self.task_id}",
            f"{'='*60}\n",
            "# 一、思考链路 (Trace)\n",
            "记录了 Agent 每一步操作的原始记录：\n",
        ]
        for t in self.trace:
            lines.append(f"## [{t['time']}] {t['type']}")
            lines.append(f"- {t['summary']}")
            if t.get("detail"):
                lines.append(f"```\n{t['detail']}\n```")
            lines.append("")

        lines.append("\n# 二、实验轨迹 (Experiment Tracking)\n")
        lines.append("| 轮次 | 模型 | seg1 | seg2 | seg3 | 判定 |")
        lines.append("|------|------|------|------|------|------|")
        # 从 graph 提取实验节点
        for nid, info in self.graph.nodes.items():
            if info["type"] in ("eval", "milestone", "baseline"):
                try:
                    m = json.loads(info["content"]) if isinstance(info["content"], str) and '{' in info["content"] else {}
                except Exception:
                    m = {}
                if m:
                    lines.append(f"| {nid} | - | {m.get('seg1','?')} | {m.get('seg2','?')} | {m.get('seg3','?')} | {info['type']} |")
        lines.append("")

        lines.append("\n# 三、因果拓扑图 (Graph Memory)\n")
        lines.append(self.graph.format_for_llm())
        lines.append("")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Captain] 日志已写入: {log_path}")

# ====================================================================
# 数据目录自动发现
# ====================================================================
def _auto_discover_data(data_dir: str):
    """扫描目录，自动发现 train/val/test 文件。按空间分辨率分组隔离不同任务。
    返回 (train, val, test, extras, has_gt)。"""
    import glob as _glob, re as _re
    files = sorted(_glob.glob(os.path.join(data_dir, "*.h5")) +
                   _glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not files:
        return None, None, None, [], True

    # Probe each file
    probes = []
    for fpath in files:
        try:
            with __import__("h5py").File(fpath, "r") as f:
                for key in ["tensor", "data", "u", "sol"]:
                    if key in f:
                        s = f[key].shape
                        n_spatial = s[-1]  # last dim = spatial
                        probes.append({"path": fpath, "key": key,
                                       "shape": s, "name": os.path.basename(fpath),
                                       "n_samples": s[0], "n_timesteps": s[1] if len(s) >= 2 else 1,
                                       "n_spatial": n_spatial})
                        break
        except Exception:
            pass
    if not probes:
        return None, None, None, [], True

    # Group by spatial resolution — different tasks have different spatial dims
    spatial_groups = {}
    for p in probes:
        spatial_groups.setdefault(p["n_spatial"], []).append(p)

    # Select the group with the most complete (train+val+test) set
    def _score_group(pps):
        """Score a group by completeness: has train, val, test."""
        has_multi = any("part" in p["name"] for p in pps)
        names = {p["name"].lower() for p in pps}
        score = len(pps)  # more files = better
        if has_multi:
            score += 2  # prefer multi-file groups (likely training set)
        for kw in ["train", "val", "test"]:
            if any(kw in n for n in names):
                score += 1
        return score

    pps = max(spatial_groups.values(), key=_score_group)

    # Detect multi-file training sets within the selected spatial group
    part_groups = {}
    standalone = []
    for p in pps:
        m = _re.match(r"(.+)_part(\d+)", p["name"])
        if m:
            part_groups.setdefault(m.group(1), []).append(p)
        else:
            standalone.append(p)

    train_candidates = []
    for base, parts in part_groups.items():
        parts.sort(key=lambda x: x["name"])
        primary = parts[0]
        extras = [x["path"] for x in parts[1:]]
        train_candidates.append({**primary, "extra_files": extras, "is_multi": True})

    # Classify standalone files by name, then by timesteps
    for p in standalone:
        name = p["name"].lower()
        if "test" in name or p["n_timesteps"] <= 15:
            p["role"] = "test"
        elif "val" in name or "valid" in name:
            p["role"] = "val"
        elif "train" in name:
            train_candidates.append({**p, "extra_files": [], "is_multi": False})
        else:
            p["role"] = "unknown"

    unknown = [p for p in standalone if p.get("role") == "unknown"]
    unknown.sort(key=lambda x: x["n_samples"] * x["n_timesteps"], reverse=True)

    train_path, val_path, test_path, extra_files = None, None, None, []
    test_has_gt = True

    # Train: multi-file candidates have priority
    if train_candidates:
        best = max(train_candidates, key=lambda x: x["n_samples"] * x["n_timesteps"])
        train_path, extra_files = best["path"], best.get("extra_files", [])
    elif unknown:
        train_path = unknown.pop(0)["path"]

    # Extract prefix from train filename for matching val/test
    train_prefix = ""
    if train_path:
        train_prefix = _re.split(r"_(?:train|val|test|part\d+|Sols)",
                                 os.path.basename(train_path))[0]

    def _prefix_match(path):
        """How well a file prefix matches the train prefix."""
        if not train_prefix or not path:
            return 0
        p = _re.split(r"_(?:train|val|test|part\d+|Sols)", os.path.basename(path))[0]
        # Count common prefix chars
        for i, (a, b) in enumerate(zip(train_prefix, p)):
            if a != b:
                return i
        return min(len(train_prefix), len(p))

    # Val: prefer matching prefix
    val_candidates = [p for p in standalone if p.get("role") == "val"]
    if val_candidates:
        val_candidates.sort(key=lambda p: _prefix_match(p["path"]), reverse=True)
        val_path = val_candidates[0]["path"]
    elif unknown:
        val_path = unknown.pop(0)["path"]

    # Test: prefer matching prefix
    test_candidates = [p for p in standalone if p.get("role") == "test"]
    if test_candidates:
        test_candidates.sort(key=lambda p: _prefix_match(p["path"]), reverse=True)
        best_test = test_candidates[0]
        test_path = best_test["path"]
        test_has_gt = best_test["n_timesteps"] > 15
    elif unknown:
        test_path = unknown.pop(0)["path"]

    return train_path, val_path, test_path, extra_files, test_has_gt


# ====================================================================
# CLI 入口
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="Captain — PDE 神经算子自动演化")
    parser.add_argument("--task", type=int, default=1, help="Task ID (1 or 2)")
    parser.add_argument("--data_dir", type=str, default="",
                        help="数据目录路径（自动发现 train/val/test）")
    parser.add_argument("--train_data", type=str, default="")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--test_data", type=str, default="")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--inner_limit", type=int, default=3)
    parser.add_argument("--model", type=str, default="", help="手动指定 baseline 模型（跳过自动排行）")
    parser.add_argument("--skip-phase0", action="store_true", help="跳过 Phase 0 训练，直接用已有结果")
    parser.add_argument("--scout-samples", type=int, default=500,
                        help="Scout 探路样本数 (0=跳过 scout，直接全量训练)")
    parser.add_argument("--scout-topk", type=int, default=2,
                        help="Scout 后保留几个候选进入全量训练")
    args = parser.parse_args()

    # 自动数据发现
    test_has_gt = True
    if args.data_dir:
        train, val, test, extras, has_gt = _auto_discover_data(args.data_dir)
        if train:
            args.train_data = train
        if val:
            args.val_data = val
        if test:
            args.test_data = test
        if extras:
            # Store extra train files for data_config
            args.extra_train_files = extras
        test_has_gt = has_gt
        print(f"[自动发现] train={os.path.basename(train) if train else '?'}, "
              f"val={os.path.basename(val) if val else '?'}, "
              f"test={os.path.basename(test) if test else '?'}, "
              f"extra_train={len(extras) if extras else 0}, test_has_gt={test_has_gt}")
    else:
        args.extra_train_files = []

    # Fallback: 自动检测 Task 1 数据路径
    if not args.train_data:
        args.train_data = os.path.join(DATA_DIR, "1D_Burgers_Sols_Nu0.001.hdf5")
    if not args.val_data:
        args.val_data = os.path.join(DATA_DIR, "task1_val.hdf5")
    if not args.test_data:
        args.test_data = os.path.join(DATA_DIR, "task1_test.hdf5")

    captain = Captain(
        task_id=args.task,
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        max_rounds=args.max_rounds,
        inner_loop_limit=args.inner_limit,
        force_model=args.model,
        skip_phase0=args.skip_phase0,
        scout_samples=args.scout_samples,
        scout_topk=args.scout_topk,
        test_has_gt=test_has_gt,
    )
    captain.run()

if __name__ == "__main__":
    main()
