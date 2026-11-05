# captain.py — PDE 科研闭环主调度器，Prog-Act 双循环，编排四个 subagent 完成自动演进
import sys, os, json, re, time, shutil, subprocess, argparse, textwrap, traceback, hashlib, difflib
from datetime import datetime
from pathlib import Path
from typing import Optional

# Package and workspace paths
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(AGENT_DIR)
for _module_path in (WORKSPACE, AGENT_DIR):
    if _module_path not in sys.path:
        sys.path.insert(0, _module_path)

from agent.llm_client import get_client
from shared_memory import GraphMemory, ResourcePool
from tools.data_probe import probe_hdf5, format_probe_report
from tools.prepare_data import prepare_split, needs_split
from tools.judge import judge, format_judgement

# ====================================================================
# 路径常量
# ====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(WORKSPACE, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Official Data and Sample", "data_and_sample_submission", "train_val_test_init")

MODEL_PATH = os.path.join(AGENT_DIR, "model.py")
TRAIN_SCRIPT = os.path.join(AGENT_DIR, "train.py")
AGENT_UTILS_PATH = os.path.join(AGENT_DIR, "utils.py")
TEMPLATES_DIR = os.path.join(WORKSPACE, "templates")
PROMPTS_DIR = os.path.join(WORKSPACE, "prompts")
OUTPUT_DIR = os.path.join(WORKSPACE, "output")
REPLIES_DIR = os.path.join(OUTPUT_DIR, "replies")
STATE_DIR = os.path.join(OUTPUT_DIR, "state")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
SNAPS_DIR = os.path.join(OUTPUT_DIR, "snapshots")
SUBS_DIR = os.path.join(OUTPUT_DIR, "submissions")
DATA_SPLITS_DIR = os.path.join(OUTPUT_DIR, "data_splits")

# ====================================================================
# 辅助：时间戳和日志
# ====================================================================
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _sha256_file(path: str) -> str:
    """计算文件 sha256，失败返回空串。"""
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ====================================================================
# Captain
# ====================================================================
class Captain:
    # Dev 修改最多尝试次数（含首次），失败则本轮无效
    DEV_MAX_ATTEMPTS = 3

    def __init__(self, task_id: int = 1,
                 train_data: str = "", val_data: str = "", test_data: str = "",
                 data_source: str = "",
                 split_ratios: tuple = (0.8, 0.1, 0.1), split_seed: int = 42,
                 train_seed: int = 42,
                 max_rounds: int = 5, inner_loop_limit: int = 3,
                 force_model: str = "", skip_phase0: bool = False,
                 scout_samples: int = 500, scout_topk: int = 2,
                 skip_final_test: bool = False, eval_samples: int = 0,
                 sigma: float = 0.05, sigma_source: str = "assumed",
                 train_overrides: dict | None = None,
                 test_has_gt: bool = True):
        self.task_id = task_id
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.data_source = data_source      # 原始数据集（文件或目录），未划分时自动切
        self.split_ratios = tuple(split_ratios)
        self.split_seed = split_seed        # 决定哪些样本进 train/val/test
        self.train_seed = train_seed        # 决定训练随机性，两者应独立
        self.max_rounds = max_rounds
        self.inner_loop_limit = inner_loop_limit
        self.force_model = force_model
        self.skip_phase0 = skip_phase0  # 跳过训练，用已有结果
        self.scout_samples = scout_samples  # 0=跳过 scout
        self.scout_topk = scout_topk        # scout 后保留几个进全量训练
        self.skip_final_test = skip_final_test  # 跳过最终 test 评估
        self.eval_samples = eval_samples        # test 评估样本数上限（0=全部）
        self.sigma = sigma                      # 单次运行的相对噪声水平
        self.sigma_source = sigma_source        # calibrated / assumed
        self.train_overrides = dict(train_overrides or {})  # 覆盖所有训练的超参
        self.test_has_gt = test_has_gt      # 测试数据是否包含 ground truth

        # LLM
        self.llm = get_client()

        # Shared memory
        os.makedirs(STATE_DIR, exist_ok=True)
        # 每次运行使用全新的图/池，旧文件归档，避免上次运行的状态污染本次轨迹
        _stamp = time.strftime("%Y%m%d_%H%M%S")
        for _fname in (f"graph_t{task_id}.json", f"pool_t{task_id}.json"):
            _fpath = os.path.join(STATE_DIR, _fname)
            if os.path.exists(_fpath):
                _arch = os.path.join(STATE_DIR, "archive")
                os.makedirs(_arch, exist_ok=True)
                shutil.move(_fpath, os.path.join(_arch, f"{_fname[:-5]}_{_stamp}.json"))
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
        self.current_model: str = ""        # agent/model.py 里当前装的是哪个模型
        self._last_detected_class: str = "" # 最近一次实例化检查识别到的类名
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
    def _model_class_name(self, model_name: str) -> str:
        """从 registry 查模型名对应的类名（fno → FNO1d）。"""
        if not model_name:
            return ""
        info = self.registry.get(model_name, {}) if isinstance(self.registry, dict) else {}
        return info.get("class", "") if isinstance(info, dict) else ""

    _SIG_SCRIPT = r"""
import sys, json, hashlib
sys.path.insert(0, {root!r})
import torch
torch.manual_seed(0)
import model as m
from tools.model_factory import build_model

cls = getattr(m, {cls!r})
net = build_model(cls, hp={hp!r}, T_IN={t_in}, N_spatial={n})
net.eval()
torch.manual_seed(0)
x = torch.randn(2, {n}, {t_in})
with torch.no_grad():
    try:
        y = net(x)
    except TypeError:
        grid = torch.linspace(0, 1, {n}).view(1, {n}, 1).expand(2, {n}, 1)
        y = net(x, grid)
if isinstance(y, tuple):
    y = y[0]
h = hashlib.sha256(y.detach().cpu().numpy().tobytes()).hexdigest()
print(json.dumps({{
    "model_class": {cls!r},
    "params": sum(p.numel() for p in net.parameters()),
    "out_shape": list(y.shape),
    "out_hash": h,
}}))
"""

    def _behavior_signature(self, context: str = "") -> dict | None:
        """固定种子 + 固定输入跑一次前向，得到模型的行为指纹。

        用于识别「文件改了但模型行为没变」的空操作：
        参数量、每层形状、以及前向输出哈希三者全同 → 改动没有实际效果。
        """
        cls_name = self._model_class_name(self.current_model)
        if not cls_name:
            return None
        cfg = getattr(self, "data_config", None) or {}
        t_in = cfg.get("t_in", 10)
        n = cfg.get("n_spatial", 256)
        hp = {k: v for k, v in (self.train_overrides or {}).items()
              if k in ("modes", "width", "n_layers")}
        script = self._SIG_SCRIPT.format(root=WORKSPACE, cls=cls_name, hp=hp, t_in=t_in, n=n)
        try:
            proc = subprocess.run([sys.executable, "-B", "-c", script],
                                  capture_output=True, text=True,
                                  timeout=300, cwd=AGENT_DIR)
        except subprocess.TimeoutExpired:
            self._trace("action", f"行为签名超时（{context}）", "")
            return None
        lines = (proc.stdout or "").strip().splitlines()
        if not lines:
            self._trace("action", f"行为签名失败（{context}）", (proc.stderr or "")[-200:])
            return None
        try:
            return json.loads(lines[-1])
        except Exception:
            return None

    def _verify_identity(self, expected_model: str, context: str = "", recheck: bool = True) -> bool:
        """校验 agent/model.py 里实际能被实例化的类，是否就是 expected_model。

        recheck=False 时复用最近一次 _instantiation_check 的结果，避免重复起子进程。
        返回 True 表示身份一致。不一致时记录告警并返回 False。
        """
        expected_cls = self._model_class_name(expected_model)
        if recheck:
            ok = self._instantiation_check()
        else:
            ok = bool(self._last_detected_class)
        detected = self._last_detected_class
        if not ok:
            self._trace("action", f"身份校验失败（{context}）：model.py 无法实例化", "")
            return False
        if expected_cls and detected != expected_cls:
            self._trace("action",
                        f"❌ 模型身份错配（{context}）：期望 {expected_model}/{expected_cls}，"
                        f"实际 {detected}", "")
            return False
        self._trace("action", f"模型身份校验通过（{context}）：{detected}", "")
        return True

    _ESCALATE_PHRASES = ("需要重新提方案", "重新提方案", "需要换方案", "建议重新提案")

    def _wants_new_proposal(self, reply: str) -> bool:
        """判断 Diagnosis 是否要求退回 Proposal（兼容 JSON 与自然语言）。"""
        parsed = self._parse_json_block(reply) or {}
        if parsed.get("needs_proposal") is True:
            return True
        text = reply or ""
        return any(p in text for p in self._ESCALATE_PHRASES)

    def _add_outcome(self, rnd: int, text: str):
        """记录本轮最终处置，并接到 judgment / error 节点上。"""
        nid = f"R{rnd}_outcome"
        self.graph.add_node(nid, "outcome", text)
        for src in (f"R{rnd}_judgement", f"R{rnd}_no_edit", f"R{rnd}_fail"):
            if src in self.graph.nodes:
                self.graph.add_edge(src, nid, "TRIGGERED", "decision")

    def _save_snapshot(self, label: str, model_name: str = "",
                       checkpoint: str = "", metrics: dict | None = None,
                       parent_label: str = ""):
        """备份 model.py + checkpoint + 模型身份，写成自描述快照。

        与旧版的区别：旧版直接复制「当前」model.py 并用全局 best_checkpoint，
        而 baseline 循环会逐个覆盖 model.py，导致 phase0_fno 快照里存的其实是
        最后一个被复制进去的模型。现在必须显式指定存的是谁、对应哪个 checkpoint。
        """
        model_name = model_name or self.current_model
        if not model_name:
            self._trace("action", f"保存快照: {label} 失败 — 模型身份未知", "")
            return False

        expected_cls = self._model_class_name(model_name)
        snap_dir = os.path.join(self.snapshot_dir, label)
        os.makedirs(snap_dir, exist_ok=True)

        code_sha = ""
        if os.path.exists(MODEL_PATH):
            shutil.copy2(MODEL_PATH, os.path.join(snap_dir, "model.py"))
            code_sha = _sha256_file(MODEL_PATH)

        ok = self._instantiation_check()
        detected = self._last_detected_class
        match = bool(ok and detected and expected_cls and detected == expected_cls)

        ckpt_path = checkpoint or self.pool.get("best_checkpoint", "") or ""
        ckpt_sha = _sha256_file(ckpt_path) if ckpt_path else ""
        snap_metrics = metrics if metrics is not None else self.best_metrics

        manifest = {
            "label": label,
            "model_name": model_name,
            "model_class_expected": expected_cls,
            "model_class_detected": detected,
            "identity_ok": match,
            "code_sha256": code_sha,
            "checkpoint": ckpt_path,
            "checkpoint_sha256": ckpt_sha,
            "metrics": snap_metrics,
            "train_seed": self.train_seed,
            "split_seed": self.split_seed,
            "created_at": _now(),
        }
        with open(os.path.join(snap_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # 旧字段保留，兼容已有工具
        with open(os.path.join(snap_dir, "checkpoint_path.txt"), "w") as f:
            f.write(ckpt_path)
        with open(os.path.join(snap_dir, "metrics.json"), "w") as f:
            json.dump(snap_metrics, f, ensure_ascii=False, indent=2)

        # 快照血缘：记录这份快照从哪份派生，供 upstream/downstream 回溯
        self.graph.add_node(f"snap:{label}", "snapshot", model_name,
                            {"model_name": model_name, "label": label,
                             "checkpoint": ckpt_path, "identity_ok": match})
        if parent_label and f"snap:{parent_label}" in self.graph.nodes:
            self.graph.add_edge(f"snap:{parent_label}", f"snap:{label}", "PRODUCES", "model.py")

        self._trace("action",
                    f"保存快照: {label} ({model_name}/{detected or '?'})",
                    f"identity_ok={match} ckpt={ckpt_path} parent={parent_label or '-'}")
        return match

    def _rollback(self) -> bool:
        """回退 model.py、模型身份和 checkpoint 到最优版本，并在回退后复检。"""
        if not self.best_code_dir or not os.path.exists(self.best_code_dir):
            self._trace("action", "回退失败：无快照", "")
            return False

        src_model = os.path.join(self.best_code_dir, "model.py")
        if not os.path.exists(src_model):
            self._trace("action", "回退失败：快照缺少 model.py", self.best_code_dir)
            return False
        shutil.copy2(src_model, MODEL_PATH)

        # 从 manifest 恢复身份与 checkpoint
        man_path = os.path.join(self.best_code_dir, "manifest.json")
        if os.path.exists(man_path):
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            self.current_model = man.get("model_name", "") or self.current_model
            ckpt = man.get("checkpoint", "")
            if ckpt:
                self.pool.put("best_checkpoint", ckpt)
            want = man.get("code_sha256", "")
            if want and _sha256_file(MODEL_PATH) != want:
                self._trace("action", "⚠ 回退后 model.py 哈希与快照记录不一致", "")
        else:
            ckpt_txt = os.path.join(self.best_code_dir, "checkpoint_path.txt")
            if os.path.exists(ckpt_txt):
                with open(ckpt_txt, "r") as f:
                    self.pool.put("best_checkpoint", f.read().strip())
            self._trace("action", "⚠ 快照无 manifest，无法核对模型身份", self.best_code_dir)

        # 回退后复检：代码能实例化，且类名与记录一致
        ok = self._verify_identity(self.current_model, context="回退复检")
        self._trace("action", f"回退到最优版本: {self.current_model}",
                    f"snapshot={self.best_code_dir} identity_ok={ok}")
        return ok

    # ================================================================
    # Subagent call
    # ================================================================
    def _call_subagent(self, agent_name: str, task_instruction: str) -> str:
        """调用一个 subagent，返回其文本回复。"""
        if agent_name not in self.prompts:
            return f"ERROR: no prompt for {agent_name}"

        system = self.prompts[agent_name]
        user = (
            f"=== 演化轨迹 ===\n{self.graph.format_for_task(task_instruction)}\n\n"
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
        train_script = TRAIN_SCRIPT
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
        # 显式指定要训练的模型类，禁止训练器自己猜
        expected_cls = self._model_class_name(self.current_model)
        if expected_cls:
            cmd += ["--model_class", expected_cls]
        else:
            self._trace("training", f"⚠ {run_label}: 模型身份未知，训练器将自行推测", "")
        # 训练随机种子：与数据切分种子相互独立
        cmd += ["--seed", str(self.train_seed)]
        # 全局训练预算覆盖（冒烟/限时用），显式覆盖优先于调用方传入
        merged = dict(extra_args or {})
        merged.update(self.train_overrides)
        extra_args = merged
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

    # ================================================================
    # Phase 0: Baseline Establishment
    # ================================================================
    def _copy_template(self, model_name: str) -> bool:
        """从模板目录复制模型代码到 agent/model.py。模板已验证正确。"""
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
                shutil.copy2(src, MODEL_PATH)
                self.current_model = model_name        # 记录身份，后续快照/train 都用它
                self._trace("action", f"从模板复制: {os.path.basename(src)} → model.py "
                                      f"(model={model_name})")
                # 如果模板有独立 utils.py，也复制
                utils_src = os.path.join(os.path.dirname(src), "utils.py")
                if os.path.exists(utils_src):
                    shutil.copy2(utils_src, AGENT_UTILS_PATH)
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
                    + "请用中文说明设计思路，然后给出完整的 model.py 代码块。"
                )
                dev_reply = self._call_subagent("dev", task)
                self._extract_code(dev_reply)

            # 2) 语法校验 + 实例化检查（带快速修复 + 模板回退）
            model_py = MODEL_PATH
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
            if self._wants_new_proposal(diag_reply):
                return False
            fix_task = (
                f"Scout 阶段训练失败，下面是诊断意见：\n{diag_reply}\n\n"
                f"当前 model.py：\n```python\n{current_model}\n```\n\n"
                "请用 SEARCH/REPLACE 块给出精确修复。"
            )
            dev_reply = self._call_subagent("dev", fix_task)
            self._apply_edits(dev_reply)
            if self._validate_python(MODEL_PATH):
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

    # ================================================================
    # Phase 0.0: 数据准备（自动切割）
    # ================================================================
    def _prepare_data(self) -> bool:
        """确保存在 train/val/test 三份数据，未划分的原始数据集自动切割。

        判定规则（纯确定性，不经过 LLM）：
          1. 显式给了 --train_data/--val_data  → 直接使用
          2. --data 指向目录：
             - 目录内已有 train/val/test 文件 → 直接用
             - 目录内只有一个未划分文件       → 自动切割
          3. --data 指向单个文件：
             - 未划分 → 自动切割
             - 已划分 → 需要同目录下配套的 val/test，否则报错
        """
        self._trace("action", "Phase 0.0: 数据准备")

        # --- 1) 显式指定 ---
        if self.train_data and self.val_data:
            self._trace("action", "使用显式指定的数据划分（不做切割）")
            self._trace("action", f"  train={self.train_data}")
            self._trace("action", f"  val={self.val_data}")
            self._trace("action", f"  test={self.test_data or '(未指定)'}")
            return True

        # --- 2) 没有数据源 ---
        if not self.data_source:
            self._trace("action", "未指定数据源，无法开始", "")
            print("\n请用 --data 指定数据集（文件或目录），例如：")
            print('  python agent/captain.py --data "/path/to/1D_diff-sorp_NA_NA.h5"')
            print('  python agent/captain.py --data "/path/to/某数据目录"')
            return False

        src = os.path.abspath(self.data_source)
        if not os.path.exists(src):
            self._trace("action", f"数据路径不存在: {src}", "")
            return False

        # --- 3) 定位待切分的源文件 ---
        if os.path.isdir(src):
            entries = sorted(
                os.path.join(src, n) for n in os.listdir(src)
                if n.lower().endswith((".h5", ".hdf5"))
            )
            if not entries:
                self._trace("action", f"目录中没有 h5/hdf5 文件: {src}", "")
                return False

            # 3a) 目录内已含 train/val/test
            by_role = {"train": [], "val": [], "test": []}
            for p in entries:
                low = os.path.basename(p).lower()
                if "test" in low:
                    by_role["test"].append(p)
                elif "val" in low or "valid" in low:
                    by_role["val"].append(p)
                elif "train" in low:
                    by_role["train"].append(p)

            if by_role["train"] and by_role["val"]:
                self.train_data = by_role["train"][0]
                self.val_data = by_role["val"][0]
                self.test_data = by_role["test"][0] if by_role["test"] else ""
                self._trace("action", "目录中已有划分，直接使用")
                self._trace("action", f"  train={os.path.basename(self.train_data)}")
                self._trace("action", f"  val={os.path.basename(self.val_data)}")
                self._trace("action", f"  test={os.path.basename(self.test_data) if self.test_data else '(无)'}")
                return True

            if len(entries) != 1:
                self._trace("action",
                            f"目录中有 {len(entries)} 个文件但无法判断划分，请用 --data 指定单个文件", "")
                return False
            raw = entries[0]
        else:
            raw = src

        # --- 4) 判断是否需要切分 ---
        need, reason = needs_split(raw)
        if not need:
            self._trace("action", f"无需切分：{reason}", raw)
            self._trace("action", "但缺少配套的 val/test，请用 --data_dir 指定目录或用 --train_data 显式指定")
            return False

        self._trace("action", f"检测到未划分数据集：{reason}", raw)

        # --- 5) 切分（带缓存）---
        stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", os.path.splitext(os.path.basename(raw))[0])
        out_dir = os.path.join(DATA_SPLITS_DIR, stem)
        try:
            manifest, reused = prepare_split(
                src=raw, out_dir=out_dir,
                ratios=self.split_ratios, seed=self.split_seed,
                verify=True, reuse=True,
                log=lambda m: self._trace("probe", m),
            )
        except Exception as e:
            self._trace("action", f"数据切分失败: {e}", "")
            return False

        self.train_data = os.path.join(out_dir, manifest["files"]["train"])
        self.val_data = os.path.join(out_dir, manifest["files"]["val"])
        self.test_data = os.path.join(out_dir, manifest["files"]["test"])
        self.test_has_gt = True

        self._trace("action", f"切分完成（{'复用缓存' if reused else '新建'}）→ {out_dir}")
        self._trace("action",
                    f"  train={manifest['counts']['train']} val={manifest['counts']['val']} "
                    f"test={manifest['counts']['test']} 种子={manifest['seed']} "
                    f"比例={manifest['ratios']}")
        self.pool.put("data_split_manifest", manifest)
        return True

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
            f"本次是 Phase 0 模型匹配。根据探测结果和 registry 里的模型物理特征，"
            f"用自然语言说明：哪些模型适合这份数据、为什么、建议按什么顺序尝试、哪些明确不推荐。\n"
            f"请务必在回答中直接写出推荐的模型名（可选："
            f"{', '.join(k for k in self.registry if isinstance(self.registry[k], dict))}），"
            f"按推荐顺序排列。不要输出 JSON。"
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
                    + "请用中文说明设计思路，然后给出完整的 model.py 代码块。"
                )
                dev_reply = self._call_subagent("dev", task)
                self._extract_code(dev_reply)

            # 语法校验
            if not self._validate_python(MODEL_PATH):
                self._trace("action", f"{model_name} 语法校验失败，跳过", "")
                results.append({"model": model_name, "error": "syntax"})
                continue
            # 身份校验：model.py 里必须确实是这个模型，否则不训
            if not self._verify_identity(model_name, context=f"baseline {model_name}"):
                results.append({"model": model_name, "error": "identity_mismatch"})
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
                fix_task = (
                    f"训练崩溃，下面是诊断意见：\n{diag_reply}\n\n"
                    f"当前 model.py：\n```python\n{self._read_current_model()}\n```\n\n"
                    "请用 SEARCH/REPLACE 块给出精确修复。"
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

            # 立刻为这个候选存快照：此刻 model.py 确实是它自己的代码
            cand_ckpt = os.path.join(RUNS_DIR, f"phase0_{model_name}", "best_phase2.pt")
            self._save_snapshot(f"phase0_{model_name}", model_name=model_name,
                                checkpoint=cand_ckpt, metrics=metrics)
            results[-1]["snapshot"] = f"phase0_{model_name}"
            results[-1]["checkpoint"] = cand_ckpt

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
        self.current_model = best["model"]
        self.pool.put("best_model", best["model"])
        self.pool.put("best_metrics", self.best_metrics)
        best_ckpt = best.get("checkpoint") or (
            os.path.join(self.best_model_path, "best_phase2.pt") if self.best_model_path else "")
        self.pool.put("best_checkpoint", best_ckpt)

        # 指向该候选自己的快照，并把工作区代码恢复成它（而不是最后一个被复制进来的模型）
        self.best_code_dir = os.path.join(self.snapshot_dir, f"phase0_{best['model']}")
        if not self._rollback():
            self._trace("action", "⚠ Baseline 恢复失败，工作区模型可能与最优不一致", "")

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
        self.current_model = best["model"]

        for i, r in enumerate(ranked):
            m = r["metrics"]
            flag = "🏆" if i == 0 else "  "
            line = f"  {flag} {i+1}. {r['model']}: RelMSE={m['overall']:.4f} seg1={m['seg1']:.4f} seg2={m['seg2']:.4f} seg3={m['seg3']:.4f}"
            self._trace("action", line, "")
            if i == 0:
                self._trace("action", "     → 选为 baseline，进入 Phase 1+ evolution", "")
                self._save_snapshot(f"phase0_{best['model']}", model_name=best["model"])

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

            # 不变量：每轮开始前，工作区 model.py 必须等于最优快照的代码。
            # 任何异常路径若漏了回滚，这里会兜底纠正，避免在坏代码上继续演化。
            self._ensure_workspace_matches_best()

            # 1) Proposal
            self._trace("action", f"Round {rnd}: 调度 Proposal")
            proposal_task = (
                f"当前最优模型: {self.pool.get('best_model')}, "
                f"指标: {json.dumps(self.best_metrics)}。"
                f"瓶颈: 分析 seg1/seg2/seg3 哪个是短板，提出改进方案。"
                f"注意：只能改 model.py，train.py 是框架文件不可修改。"
            )
            prop_reply = self._call_subagent("proposal", proposal_task)
            selected, plan_text = self._parse_proposal(prop_reply)
            if not plan_text:
                self._trace("action", "Proposal 无有效输出，跳过本轮", "")
                continue

            self._trace("llm_reasoning", f"本轮方案: {selected.get('title','')}")
            self.graph.add_node(f"R{rnd}_hypothesis", "hypothesis",
                                selected.get("title", ""), {"plan": plan_text})

            # 2) 保存快照
            _parent = os.path.basename(self.best_code_dir) if self.best_code_dir else ""
            self._save_snapshot(f"round{rnd}_pre", model_name=self.current_model,
                                parent_label=_parent)
            # 编辑前记录行为指纹（此刻 model.py 仍是修改前的代码）
            sig_before = self._behavior_signature(f"Round {rnd} 编辑前")

            # 3) Dev — 传完整 model.py，确保 old 字符串精确匹配
            dev_task = (
                f"本轮改进方案：\n{plan_text}\n\n"
                f"当前 model.py：\n```python\n{self._read_current_model()}\n```\n\n"
                "请用 SEARCH/REPLACE 块给出精确修改：SEARCH 的内容必须从上面代码原样复制"
                "（含缩进，且确保唯一），块外写清楚改了什么、为什么。"
            )
            dev_reply = self._call_subagent("dev", dev_task)
            edit_res = self._apply_edits(dev_reply)

            # 修改没落地 → 重试；仍失败则本轮无效，绝不带病训练
            attempt = 1
            while not edit_res["ok"] and attempt < self.DEV_MAX_ATTEMPTS:
                attempt += 1
                self._trace("action",
                            f"Round {rnd} 第 {attempt-1} 次修改无效（{edit_res['reason']}），重试", "")
                self._rollback()          # 回到干净状态，避免半套修改叠加
                retry_task = dev_task + (
                    f"\n\n🔴 上一次输出没有产生任何有效修改：{edit_res['reason']}。"
                    f"请重新输出 SEARCH/REPLACE 块，SEARCH 必须与当前代码完全一致且只匹配一处。"
                )
                dev_reply = self._call_subagent("dev", retry_task)
                edit_res = self._apply_edits(dev_reply)

            if not edit_res["ok"]:
                self._trace("action",
                            f"Round {rnd} Dev 修改未生效（{edit_res['reason']}），"
                            f"本轮判无效，不启动训练", "")
                self.graph.add_node(f"R{rnd}_no_edit", "error",
                                    f"Round {rnd} 无效修改: {edit_res['reason']}")
                self._add_outcome(rnd, f"无效（{edit_res['reason']}）")
                self._rollback()
                continue

            self._trace("action",
                        f"Round {rnd} 代码修改生效: {edit_res['applied']} 处 "
                        f"(尝试 {attempt} 次) "
                        f"hash {edit_res['hash_before'][:8]}→{edit_res['hash_after'][:8]}")
            self.graph.add_node(f"R{rnd}_edit", "code_edit",
                                f"{edit_res['applied']} 处修改",
                                {"applied": edit_res["applied"],
                                 "hash_before": edit_res["hash_before"],
                                 "hash_after": edit_res["hash_after"]})
            self.graph.add_edge(f"R{rnd}_hypothesis", f"R{rnd}_edit", "PRODUCES", "model.py")

            # 语法校验 + 实例化检查
            if not self._validate_python(MODEL_PATH):
                self._trace("action", "语法校验失败，触发内循环修复", "")
                if not self._inner_loop_fix("语法错误"):
                    self._trace("action", f"Round {rnd} 语法无法修复，本轮无效，回退", "")
                    self.graph.add_node(f"R{rnd}_syntax", "error", f"Round {rnd} 语法错误未修复")
                    self._add_outcome(rnd, "无效（语法错误未修复）")
                    self._rollback()
                    continue
            if not self._instantiation_check():
                self._trace("action", "实例化检查失败，触发内循环修复", "")
                if not self._inner_loop_fix("实例化/forward 失败"):
                    self._trace("action", f"Round {rnd} 实例化无法修复，本轮无效，回退", "")
                    self.graph.add_node(f"R{rnd}_instfail", "error",
                                        f"Round {rnd} 实例化失败未修复")
                    self._add_outcome(rnd, "无效（实例化/forward 失败）")
                    self._rollback()
                    continue
            # 身份校验：实际实例化的类必须仍是本轮要优化的模型，否则本轮无效
            if not self._verify_identity(self.current_model, context=f"Round {rnd}", recheck=False):
                self._trace("action", f"Round {rnd} 模型身份错配，本轮无效，回退", "")
                self._rollback()
                continue

            # 3.5) 行为级校验：文件改了不等于模型变了
            sig_after = self._behavior_signature(f"Round {rnd} 编辑后")
            if sig_before and sig_after:
                if sig_before == sig_after:
                    self._trace("action",
                                f"Round {rnd} 空操作：模型行为完全未变，本轮无效，不训练",
                                f"params={sig_after.get('params')} "
                                f"out_hash={str(sig_after.get('out_hash'))[:16]}")
                    self.graph.add_node(f"R{rnd}_noop", "error",
                                        f"Round {rnd} 空操作：改动未影响模型行为")
                    self._add_outcome(rnd, "无效（改动未改变模型行为）")
                    self._rollback()
                    continue
                self._trace("action",
                            f"Round {rnd} 行为已改变：params {sig_before.get('params')} → "
                            f"{sig_after.get('params')}, "
                            f"out_hash {str(sig_before.get('out_hash'))[:8]} → "
                            f"{str(sig_after.get('out_hash'))[:8]}")

            # 4) 训练（带内循环修复 + 回退保护）
            ok, train_time = self._run_training(f"round{rnd}")
            if not ok:
                self.graph.add_node(f"R{rnd}_fail", "error", f"Round {rnd} 训练失败")
                chain = self.graph.trace_root_cause(f"R{rnd}_fail")
                if chain:
                    self._trace("action", f"Round {rnd} 失败溯源",
                                " ← ".join(chain[-6:]))
                if self._inner_loop_fix("训练失败"):
                    # 修复成功 → 重新训练，验证修复是否真的解决了问题
                    self._trace("action", f"Round {rnd} 内循环修复完成，重新训练验证")
                    ok, train_time = self._run_training(f"round{rnd}")
                if not ok:
                    self._trace("action", f"Round {rnd} 训练失败无法修复，回退 model.py 到最优版本")
                    self._rollback()
                    continue

            self.graph.add_node(f"R{rnd}_train", "train", f"{train_time:.0f}s",
                                {"train_time": train_time, "run": f"round{rnd}"})
            self.graph.add_edge(f"R{rnd}_edit", f"R{rnd}_train", "PRODUCES", "checkpoint")

            # 5) EvalReflection
            current_metrics = self._read_metrics(f"round{rnd}")
            if not current_metrics:
                self._trace("action", "无法读取指标，回退 model.py 到最优版本")
                self._rollback()
                continue

            self.graph.add_node(f"R{rnd}_eval", "eval", "", current_metrics)
            self.graph.add_edge(f"R{rnd}_train", f"R{rnd}_eval", "PRODUCES", "metrics")

            # 5.1) 确定性判定：由代码算，LLM 无权修改
            judged = judge(self.best_metrics, current_metrics,
                           sigma=self.sigma, sigma_source=self.sigma_source)
            v = judged["verdict"]
            self._trace("eval", f"Round {rnd} 判定: {v}（程序）",
                        json.dumps(judged, ensure_ascii=False))

            # 5.2) LLM 只解释原因、给方向，不参与判定
            eval_task = (
                f"判定已由程序算出，你必须接受，不得修改。\n"
                f"程序判定: {v}\n"
                f"判定依据: {format_judgement(judged)}\n"
                f"阈值: 2*sigma = {judged['threshold']*100:.1f}%"
                f"（sigma={judged['sigma']}，来源={judged['sigma_source']}）\n"
                f"Baseline指标: {json.dumps(self.best_metrics)}\n"
                f"本轮指标: {json.dumps(current_metrics)}\n"
                f"假说: {selected.get('title')}\n"
                f"训练时间: {train_time:.0f}s\n"
                "请用自然语言回答两件事：第一，为什么会出现这个结果（对应什么物理/数值机制）；"
                "第二，下一轮建议改什么、为什么。不要输出判定，判定已经定了。"
            )
            eval_reply = self._call_subagent("eval_reflection", eval_task)
            judged["reflection"] = (eval_reply or "").strip()
            if judged["reflection"]:
                self._trace("eval", f"Round {rnd} 解释", judged["reflection"][:500])
            interp = self._parse_json_block(eval_reply) or {}
            if interp.get("verdict") and interp["verdict"] != v:
                judged["llm_disagreement"] = interp["verdict"]
                self._trace("eval",
                            f"⚠ LLM 给出的判定 {interp['verdict']} 与程序 {v} 不一致，以程序为准", "")
            judged_meta = {k: judged.get(k) for k in
                           ("verdict", "threshold", "sigma", "overall", "segments", "reasons")}
            self.graph.add_node(f"R{rnd}_judgement", "judgement", v, judged_meta)
            self.graph.add_edge(f"R{rnd}_eval", f"R{rnd}_judgement", "PRODUCES", "verdict")

            # 6) 根据判定行动

            if v == "IMPROVED":
                new_ckpt = os.path.join(RUNS_DIR, f"round{rnd}", "best_phase2.pt")
                if not os.path.exists(new_ckpt):
                    self._trace("action",
                                f"Round {rnd} 判 IMPROVED 但 checkpoint 不存在，不采纳", new_ckpt)
                    self._rollback()
                    continue

                # 先存快照（显式指定 checkpoint/metrics，不读 Pool），成功才更新指针
                snap_ok = self._save_snapshot(
                    f"round{rnd}_best", model_name=self.current_model,
                    checkpoint=new_ckpt, metrics=current_metrics,
                    parent_label=f"round{rnd}_pre")
                if not snap_ok:
                    self._trace("action",
                                f"Round {rnd} 改进快照身份校验失败，不采纳，回退", "")
                    self._rollback()
                    continue

                # 四项一起更新，避免「指标变了但模型/ckpt 没变」
                self.best_metrics = current_metrics
                self.best_model_path = os.path.join(RUNS_DIR, f"round{rnd}")
                self.best_code_dir = os.path.join(self.snapshot_dir, f"round{rnd}_best")
                self.pool.put("best_metrics", self.best_metrics)
                self.pool.put("best_checkpoint", new_ckpt)
                self.pool.put("best_model", self.current_model)
                self._add_outcome(rnd, "IMPROVED（采纳，更新最优）")
                self._trace("action",
                            f"✅ Round {rnd} 新最优已同步",
                            f"model={self.current_model} ckpt={new_ckpt} "
                            f"metrics={json.dumps(current_metrics)}")
                print(f"  ✅ IMPROVED — 更新 baseline（模型/指标/checkpoint/快照已同步）")

            elif v == "DEGRADED":
                self._add_outcome(rnd, "DEGRADED（回退）")
                print(f"  ❌ DEGRADED — 回退到最优版本")
                self._rollback()
                # 外循环：Proposal 重新出方案
                self._trace("action", "触发外循环 — Proposal 基于最优版本重新出方案", "")

            elif v == "MIXED":
                print(f"  ⚠️ MIXED — 回退 model.py，保留指标供后续 Proposal 参考")
                self._rollback()
                self._add_outcome(rnd, "MIXED（回退）")

            elif v == "INVALID":
                print(f"  ⚠️ INVALID — 指标不足以判定，回退")
                self._trace("action",
                            f"Round {rnd} 判定无效：指标不可用（分段 baseline 全为 0 或缺失）", "")
                self._rollback()
                self._add_outcome(rnd, "INVALID（指标不可用，回退）")

            else:  # STAGNANT
                print(f"  ⏸️ STAGNANT — 无显著变化，回退到最优版本")
                self._add_outcome(rnd, "STAGNANT（在噪声阈值内，回退）")
                self._rollback()

        print(f"\n{'='*60}\nEvolution complete.\n{'='*60}")

    # ================================================================
    # 内循环修复
    # ================================================================
    def _inner_loop_fix(self, error_summary: str) -> bool:
        """内循环：Diagnosis → Dev 修复，最多 inner_loop_limit 次。"""
        err_log = self.pool.get("last_error", "")
        self.pool.delete("last_error")   # 读过即清，不让旧错误污染后续轮次
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
            if self._wants_new_proposal(diag_reply):
                self._trace("action", "Diagnosis 判定需要 Proposal 重出方案", "")
                return False  # 升级到外循环

            # Dev 修复
            fix_task = (
                f"内循环修复。下面是诊断意见：\n{diag_reply}\n\n"
                f"当前 model.py：\n```python\n{self._read_current_model()}\n```\n\n"
                "请用 SEARCH/REPLACE 块给出精确修复。"
            )
            dev_reply = self._call_subagent("dev", fix_task)
            self._apply_edits(dev_reply)

            if self._validate_python(MODEL_PATH):
                # 语法通过还不够，必须能实例化 + forward 才算修复成功
                if self._instantiation_check():
                    return True
                self._trace("action", f"内循环修复 ({attempt+1}): 语法通过但实例化/forward 失败，继续修复")

        self._trace("action", f"内循环耗尽 ({self.inner_loop_limit}次)，升级外循环", "")
        return False

    # ================================================================
    # 入口
    # ================================================================
    # ================================================================
    # 血缘审计（用上游/下游遍历回答「最优从哪来」「谁被污染」）
    # ================================================================
    def _ensure_workspace_matches_best(self) -> bool:
        """确保 agent/model.py 与最优快照一致；不一致则恢复。

        这是「模型对应关系」的兜底：正常情况下各分支已经回滚，
        但任何遗漏的路径都会在这里被发现并纠正。
        """
        if not self.best_code_dir:
            return False
        best_model_py = os.path.join(self.best_code_dir, "model.py")
        if not os.path.exists(best_model_py):
            return False
        cur = _sha256_file(MODEL_PATH)
        want = _sha256_file(best_model_py)
        if cur == want:
            return True
        self._trace("action",
                    f"⚠ Round {self.current_round} 开始前发现工作区代码与最优快照不一致，自动恢复",
                    f"{str(cur)[:8]} → {str(want)[:8]}")
        return self._rollback()

    def _lineage_report(self) -> dict:
        """沿图边回溯：最优模型的血统链 + 受无效轮次影响的下游节点。"""
        best_snap = f"snap:{os.path.basename(self.best_code_dir)}" if self.best_code_dir else ""
        ancestors, roots = [], []
        if best_snap and best_snap in self.graph.nodes:
            ancestors = self.graph.lineage_path(best_snap)
            roots = self.graph.trace_root_cause(best_snap)

        contaminated = {}
        for nid, info in self.graph.nodes.items():
            if info.get("type") != "outcome":
                continue
            if "无效" not in (info.get("content") or ""):
                continue
            downstream = sorted(self.graph.find_contaminated(nid))
            if downstream:
                contaminated[nid] = downstream

        report = {
            "best_snapshot": best_snap,
            "lineage": ancestors,
            "roots": roots,
            "contaminated_outcomes": contaminated,
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
        }
        path = os.path.join(OUTPUT_DIR, "lineage_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self._trace("action", "血缘审计",
                    f"最优血统: {' → '.join(ancestors) if ancestors else '(无)'}")
        if contaminated:
            self._trace("action", "⚠ 存在受污染的下游轮次",
                        json.dumps(contaminated, ensure_ascii=False))
        return report

    # ================================================================
    # 最终 Test 评估（流程末尾执行一次，结果不参与任何决策）
    # ================================================================
    def _final_test_eval(self, max_samples: int = 0):
        """在 test 集上对比「原始 baseline」与「演化后模型」。

        纪律：
          - 只在所有轮次结束后调用一次；
          - 结果只写报告，不写回 Pool/Graph，不影响任何后续决策；
          - 两个模型必须同类，否则本次对比判为无效。
        """
        self._trace("action", "最终 Test 评估")
        if not self.test_data or not os.path.exists(self.test_data):
            self._trace("action", "跳过：没有 test 数据", "")
            return None

        baseline_model = self.pool.get("best_model", "") or ""
        targets = [
            ("baseline", os.path.join(SNAPS_DIR, f"phase0_{baseline_model}") if baseline_model else ""),
            ("evolved", getattr(self, "best_code_dir", "") or ""),
        ]

        # 如果一轮都没改进，evolved 与 baseline 是同一份快照，对比无意义
        baseline_snap = targets[0][1]
        evolved_snap = targets[1][1]
        if baseline_snap and evolved_snap and os.path.abspath(baseline_snap) == os.path.abspath(evolved_snap):
            self._trace("action", "最终 Test 评估：无演化模型（所有轮次均未改进），跳过对比", "")
            report = {"test_data": self.test_data, "skipped": "no_evolved_model",
                      "note": "所有轮次均未产生新最优，evolved 与 baseline 为同一模型"}
            with open(os.path.join(OUTPUT_DIR, "final_test_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            return report

        eval_root = os.path.join(OUTPUT_DIR, "final_eval")
        results = {}
        backup_model = MODEL_PATH + ".prefinal"
        shutil.copy2(MODEL_PATH, backup_model)

        try:
            for label, snap in targets:
                if not snap or not os.path.isdir(snap):
                    self._trace("action", f"  {label}: 无快照，跳过", snap or "(空)")
                    continue

                snap_model = os.path.join(snap, "model.py")
                ckpt_txt = os.path.join(snap, "checkpoint_path.txt")
                if not os.path.exists(snap_model) or not os.path.exists(ckpt_txt):
                    self._trace("action", f"  {label}: 快照不完整，跳过", snap)
                    continue
                with open(ckpt_txt, "r") as f:
                    ckpt = f.read().strip()
                if not ckpt or not os.path.exists(ckpt):
                    self._trace("action", f"  {label}: checkpoint 不存在，跳过", ckpt or "(空)")
                    continue

                # 换成该快照自己的 model.py（保证模型类与 checkpoint 配套）
                shutil.copy2(snap_model, MODEL_PATH)

                out_dir = os.path.join(eval_root, label)
                os.makedirs(out_dir, exist_ok=True)
                self._write_data_config(out_dir)

                cmd = [
                    sys.executable, os.path.join(WORKSPACE, "tools", "evaluate.py"),
                    "--checkpoint", ckpt,
                    "--test_data", self.test_data,
                    "--output_dir", out_dir,
                    "--task_id", str(self.task_id),
                ]
                if max_samples and max_samples > 0:
                    cmd += ["--max_samples", str(max_samples)]

                self._trace("action", f"  {label}: 在 test 上评估 ...", ckpt)
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          cwd=AGENT_DIR, timeout=7200)
                except subprocess.TimeoutExpired:
                    self._trace("action", f"  {label}: 评估超时", "")
                    continue

                m_cls = re.search(r"Model:\s*(\w+)", proc.stdout)
                model_class = m_cls.group(1) if m_cls else "?"
                score_path = os.path.join(out_dir, f"task{self.task_id}_score.json")
                if not os.path.exists(score_path):
                    self._trace("action", f"  {label}: 评估未产出结果",
                                (proc.stderr or proc.stdout)[-300:])
                    continue

                with open(score_path, "r") as f:
                    sc = json.load(f)
                results[label] = {
                    "model_class": model_class,
                    "checkpoint": ckpt,
                    "metrics": sc.get("metrics"),
                    "scores": sc.get("scores"),
                }
                mt = sc.get("metrics", {})
                self._trace("eval", f"  {label}({model_class}): "
                                    f"overall={mt.get('overall', 0):.6g} "
                                    f"seg1={mt.get('seg1', 0):.6g} "
                                    f"seg2={mt.get('seg2', 0):.6g} "
                                    f"seg3={mt.get('seg3', 0):.6g}")
        finally:
            shutil.move(backup_model, MODEL_PATH)   # 无论成败都还原 model.py

        # ---- 汇总 ----
        report = {"test_data": self.test_data, "results": results}
        if "baseline" in results and "evolved" in results:
            b, e = results["baseline"], results["evolved"]
            same_cls = b["model_class"] == e["model_class"]
            delta = {}
            for k in ("overall", "seg1", "seg2", "seg3"):
                bv = (b["metrics"] or {}).get(k, 0.0)
                ev = (e["metrics"] or {}).get(k, 0.0)
                delta[k] = ((ev - bv) / bv) if bv else None
            report["delta_rel"] = delta
            report["same_model_class"] = same_cls
            report["conclusion_valid"] = bool(same_cls)

            if not same_cls:
                self._trace("action",
                            f"⚠ 对比无效：baseline 是 {b['model_class']}，evolved 是 {e['model_class']}", "")
            else:
                d = delta.get("overall")
                self._trace("eval",
                            f"Test 对比（{b['model_class']}）：baseline={b['metrics']['overall']:.6g} "
                            f"→ evolved={e['metrics']['overall']:.6g} "
                            f"相对变化={'-' if d < 0 else '+'}{abs(d) * 100:.2f}%"
                            if d is not None else "Test 对比完成")

        report_path = os.path.join(OUTPUT_DIR, "final_test_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self._trace("action", f"最终 Test 报告: {report_path}", "")
        return report

    def run(self):
        print(f"🚀 Captain 启动 — Task {self.task_id}")

        # Phase 0.0: 数据准备（自动切割）
        if not self._prepare_data():
            self._generate_logs()
            return

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

        # 血缘审计：最优从哪来、谁被污染
        self._lineage_report()

        # 最终 Test 评估：只做一次，结果不回流
        if not self.skip_final_test:
            self._final_test_eval(self.eval_samples)

        self._generate_logs()
        print(f"\n📄 日志已保存: output/task{self.task_id}_report.log")

    # ================================================================
    # 辅助方法
    # ================================================================
    def _read_current_model(self) -> str:
        mp = MODEL_PATH
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
                [sys.executable, "-B", "-c", check_script],
                capture_output=True, text=True, timeout=30, cwd=AGENT_DIR,
            )
            ok = proc.returncode == 0 and "OK:" in proc.stdout
            m = re.search(r"OK:\s*([A-Za-z_][A-Za-z0-9_]*)", proc.stdout)
            self._last_detected_class = m.group(1) if m else ""
            self._trace("action",
                        f"实例化检查: {'PASS' if ok else 'FAIL'}",
                        proc.stdout.strip()[:200] if ok else proc.stderr.strip()[:200])
            return ok
        except subprocess.TimeoutExpired:
            self._last_detected_class = ""
            self._trace("action", "实例化检查: TIMEOUT", "")
            return False
        except Exception as e:
            self._last_detected_class = ""
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
            m_overall = _re.search(r'(?:RelMSE|Val Rel-MSE|val_rel)\s*[=:]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|inf|nan)', line)
            m1 = _re.search(r'(?:seg1|S1)\s*[=:]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|inf|nan)', line)
            m2 = _re.search(r'(?:seg2|S2)\s*[=:]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|inf|nan)', line)
            m3 = _re.search(r'(?:seg3|S3)\s*[=:]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|inf|nan)', line)
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
        model_py = MODEL_PATH
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
                    safe_name = os.path.basename(str(fname))
                    fpath = os.path.join(AGENT_DIR, safe_name)
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

    # 允许 Dev 修改的文件白名单（basename）
    # 只允许改模型文件；train.py 是框架，任何角色都不得修改
    EDIT_WHITELIST = {"model.py"}

    # ================================================================
    # Dev 输出解析：SEARCH/REPLACE 文本块（非 JSON）
    # ================================================================
    _EDIT_BLOCK_RE = re.compile(
        r"<{3,}\s*(?:SEARCH|OLD)\s*\n(.*?)\n\s*={3,}\s*\n(.*?)\n\s*>{3,}\s*(?:REPLACE|NEW)",
        re.DOTALL,
    )
    _FILE_HINT_RE = re.compile(r"^\s*(?:FILE|文件)\s*[:：]\s*(\S+)\s*$", re.MULTILINE)

    def _parse_edit_blocks(self, reply: str) -> list:
        """从自然语言回复里解析 SEARCH/REPLACE 块。

        格式（不涉及 JSON，代码原样贴）：
            FILE: model.py
            <<<<<<< SEARCH
            原代码
            =======
            新代码
            >>>>>>> REPLACE
        """
        blocks = []
        for m in self._EDIT_BLOCK_RE.finditer(reply):
            prefix = reply[:m.start()]
            hint = None
            for cand in self._FILE_HINT_RE.finditer(prefix):
                hint = cand                      # 取块前最近的一个 FILE: 提示
            blocks.append({
                "file": hint.group(1) if hint else "model.py",
                "old": m.group(1),
                "new": m.group(2),
            })
        return blocks

    def _find_unique(self, code: str, old: str):
        """在 code 中定位 old，三级匹配。返回 (start, length, method)。

        start 为 None 表示失败，此时 method 里带失败原因。
        三级：精确 → 空白归一 → 模糊（唯一最佳且相似度 >= 0.90）。
        """
        if not old.strip():
            return None, None, "empty"

        # 1) 精确匹配
        cnt = code.count(old)
        if cnt == 1:
            return code.index(old), len(old), "exact"
        if cnt > 1:
            return None, None, f"ambiguous({cnt})"

        # 2) 空白归一：按行比较，忽略行首缩进 / 行尾空白
        old_lines = [l.strip() for l in old.splitlines()]
        code_lines = code.splitlines(keepends=True)
        n = len(old_lines)
        hits = [i for i in range(len(code_lines) - n + 1)
                if [code_lines[i + j].strip() for j in range(n)] == old_lines]
        if len(hits) == 1:
            i = hits[0]
            start = sum(len(x) for x in code_lines[:i])
            length = sum(len(x) for x in code_lines[i:i + n])
            return start, length, "normalized"
        if len(hits) > 1:
            return None, None, f"ambiguous({len(hits)})"

        # 3) 模糊匹配
        best_i, best_s, second_s = None, 0.0, 0.0
        for i in range(len(code_lines) - n + 1):
            window = "".join(code_lines[i:i + n])
            s = difflib.SequenceMatcher(None, old.strip(), window.strip()).ratio()
            if s > best_s:
                best_i, second_s, best_s = i, best_s, s
            elif s > second_s:
                second_s = s
        if best_i is not None and best_s >= 0.90 and (best_s - second_s) >= 0.05:
            start = sum(len(x) for x in code_lines[:best_i])
            length = sum(len(x) for x in code_lines[best_i:best_i + n])
            return start, length, f"fuzzy({best_s:.2f})"
        return None, None, f"not_found(best={best_s:.2f})"

    def _apply_edits(self, reply: str) -> dict:
        """应用 Dev 的修改，返回结构化结果。

        优先解析 SEARCH/REPLACE 文本块（自然语言描述 + 原样代码，无 JSON 转义）；
        没有块时回退到旧的 JSON edits / generate 模式。

        调用方据 ok 决定是否允许训练：
          - 没有可应用的修改          → ok=False
          - old 未找到 / 不唯一        → ok=False
          - 修改后文件哈希没变         → ok=False
        """
        before = _sha256_file(MODEL_PATH)
        result = {
            "mode": "", "ok": False, "applied": 0, "failed": 0,
            "hash_before": before, "hash_after": before,
            "reason": "", "details": [],
        }

        # ---- 路径 A：SEARCH/REPLACE 文本块 ----
        blocks = self._parse_edit_blocks(reply)
        if blocks:
            result["mode"] = "blocks"
            for blk in blocks:
                fname = os.path.basename(str(blk.get("file") or "model.py"))
                if fname not in self.EDIT_WHITELIST:
                    result["failed"] += 1
                    result["details"].append(f"{fname}: 不在白名单")
                    self._trace("action", f"修改 {fname}: ✗ 不在允许修改的白名单内", "")
                    continue
                fpath = os.path.join(AGENT_DIR, fname)
                if not os.path.exists(fpath):
                    result["failed"] += 1
                    result["details"].append(f"{fname}: 文件不存在")
                    continue

                with open(fpath, "r") as f:
                    code = f.read()

                start, length, method = self._find_unique(code, blk["old"])
                if start is None:
                    result["failed"] += 1
                    result["details"].append(f"{fname}: {method}")
                    self._trace("action",
                                f"修改 {fname}: ✗ 无法定位（{method}）",
                                blk["old"][:80])
                    continue

                code = code[:start] + blk["new"] + code[start + length:]
                with open(fpath, "w") as f:
                    f.write(code)
                result["applied"] += 1
                self._trace("action",
                            f"修改 {fname}: ✓ 匹配方式={method}",
                            f"旧: {blk['old'][:60]!r} → 新: {blk['new'][:60]!r}")

            after = _sha256_file(MODEL_PATH)
            changed = after != before
            result["hash_after"] = after
            result["ok"] = bool(result["applied"] > 0 and result["failed"] == 0 and changed)
            if not result["ok"]:
                if result["failed"]:
                    result["reason"] = f"{result['failed']} 处修改失败"
                elif result["applied"] == 0:
                    result["reason"] = "没有应用任何修改"
                else:
                    result["reason"] = "修改后文件哈希未变化"
            self._trace("action",
                        f"修改结果: {'生效' if result['ok'] else '无效'} "
                        f"(成功 {result['applied']} / 失败 {result['failed']})",
                        f"hash {before[:8]}→{after[:8]}")
            return result

        # ---- 路径 B：旧的 JSON 格式（向后兼容） ----
        parsed = self._parse_json_block(reply)
        if not parsed or parsed.get("mode") != "edit":
            # generate 模式或纯文本：整体写入代码
            self._extract_code(reply)
            after = _sha256_file(MODEL_PATH)
            changed = after != before
            result.update(mode="generate", hash_after=after, ok=changed,
                          reason="" if changed else "generate 模式未产生代码变更")
            self._trace("action",
                        f"代码写入（generate）: {'生效' if changed else '无变更'}",
                        f"hash {before[:8]}→{after[:8]}")
            return result

        result["mode"] = "edit"
        edits = parsed.get("edits") or []
        if not edits:
            result["reason"] = "edits 为空（Dev 未提出任何修改）"
            self._trace("action", "Dev 输出 edits 为空，未修改任何文件", "")
            return result

        for edit in edits:
            fname = str(edit.get("file", "model.py"))
            safe_name = os.path.basename(fname)
            if safe_name not in self.EDIT_WHITELIST:
                result["failed"] += 1
                result["details"].append(f"{safe_name}: 不在白名单")
                self._trace("action", f"编辑 {safe_name}: ✗ 不在允许修改的白名单内", "")
                continue
            fpath = os.path.join(AGENT_DIR, safe_name)
            if not os.path.exists(fpath):
                result["failed"] += 1
                result["details"].append(f"{safe_name}: 文件不存在")
                self._trace("action", f"编辑 {safe_name}: ✗ 文件不存在", "")
                continue

            with open(fpath, "r") as f:
                code = f.read()
            original = code

            for change in edit.get("changes", []):
                old = change.get("old", "")
                new = change.get("new", "")
                if not old or old == new:
                    result["failed"] += 1
                    result["details"].append("old 为空或 old==new")
                    self._trace("action", f"编辑 {safe_name}: ✗ old 为空或与 new 相同", "")
                    continue

                cnt = code.count(old)
                if cnt == 1:
                    code = code.replace(old, new, 1)
                    result["applied"] += 1
                    self._trace("action",
                                f"编辑 {safe_name}: ✓ {old[:40].strip()} → {new[:40].strip()}",
                                change.get("reason", ""))
                    continue

                # 去首尾空白再试一次
                if cnt == 0 and old.strip() and code.count(old.strip()) == 1:
                    code = code.replace(old.strip(), new.strip(), 1)
                    result["applied"] += 1
                    self._trace("action", f"编辑 {safe_name}: ~ (strip) {old[:40].strip()}", "")
                    continue

                result["failed"] += 1
                if cnt == 0:
                    result["details"].append(f"old 未找到: {old[:50]}")
                    self._trace("action", f"编辑 {safe_name}: ✗ 未找到 — {old[:60].strip()}", "")
                else:
                    result["details"].append(f"old 出现 {cnt} 次，不唯一: {old[:50]}")
                    self._trace("action", f"编辑 {safe_name}: ✗ old 出现 {cnt} 次，不唯一", "")

            if code != original:
                with open(fpath, "w") as f:
                    f.write(code)

        after = _sha256_file(MODEL_PATH)
        changed = after != before
        result["hash_after"] = after
        result["ok"] = bool(result["applied"] > 0 and result["failed"] == 0 and changed)
        if not result["ok"]:
            if result["failed"]:
                result["reason"] = f"{result['failed']} 处修改失败"
            elif result["applied"] == 0:
                result["reason"] = "没有应用任何修改"
            else:
                result["reason"] = "修改后文件哈希未变化"

        self._trace("action",
                    f"修改结果: {'生效' if result['ok'] else '无效'} "
                    f"(成功 {result['applied']} / 失败 {result['failed']})",
                    f"hash {before[:8]}→{after[:8]}")
        return result

    def _parse_proposal(self, reply: str):
        """解析 Proposal 输出，返回 (selected, plan_text)。

        优先解析旧的 JSON hypotheses（向后兼容）；否则把整段自然语言当作方案。
        """
        parsed = self._parse_json_block(reply)
        if parsed and isinstance(parsed.get("hypotheses"), list) and parsed["hypotheses"]:
            hyps = sorted(parsed["hypotheses"],
                          key=lambda h: 0 if h.get("priority") == "P0" else 1)
            sel = hyps[0]
            return sel, json.dumps(sel, ensure_ascii=False, indent=2)

        text = (reply or "").strip()
        if len(text) < 20:
            return {}, ""
        title = ""
        for line in text.splitlines():
            line = line.strip().lstrip("#-*> 0123456789.")
            if line:
                title = line[:100]
                break
        return {"title": title or text[:80], "target": {}}, text

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

    _NEGATION_WORDS = ("不要", "不适合", "不推荐", "排除", "避免", "not ", "don't", "avoid")

    def _parse_candidates(self, text: str) -> list:
        """解析候选模型。优先 JSON（兼容旧格式），否则从自然语言里按出现顺序提取。"""
        parsed = self._parse_json_block(text)
        if parsed:
            cands = parsed.get("candidates", [])
            if cands:
                return cands
            pool = parsed.get("pool", parsed.get("memory_update", {}).get("pool", {}))
            cands = pool.get("phase0_candidates", pool.get("candidates", []))
            if isinstance(cands, list) and cands:
                return [{"model": m, "priority": i + 1, "reason": "from_pool"}
                        for i, m in enumerate(cands) if isinstance(m, str)]

        # 自然语言：按在文本中首次出现的位置排序，跳过被否定的提法
        t = (text or "").lower()
        hits = []
        for name in self.registry:
            if not isinstance(self.registry.get(name), dict):
                continue
            m = re.search(rf"\b{re.escape(name.lower())}\b", t)
            if not m:
                continue
            ctx = t[max(0, m.start() - 30):m.start()]
            if any(neg in ctx for neg in self._NEGATION_WORDS):
                continue
            hits.append((m.start(), name))
        hits.sort()
        return [{"model": name, "priority": i + 1, "reason": "nl_mention"}
                for i, (_, name) in enumerate(hits)]

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
    parser.add_argument("--data", type=str, default="",
                        help="数据集路径（单个 h5 文件或目录）；未划分时自动切割")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="自动切割比例，默认 0.8 0.1 0.1")
    parser.add_argument("--seed", type=int, default=42,
                        help="自动切割的打乱种子，默认 42")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="训练随机种子（权重初始化/数据打乱），与切分种子独立")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="单次运行的相对噪声水平，判定阈值=2*sigma（默认 0.05）")
    parser.add_argument("--sigma-calibrated", action="store_true",
                        help="声明 sigma 已由重复实验标定（否则标记为假定值）")
    parser.add_argument("--phase1-epochs", type=int, default=None,
                        help="覆盖所有训练的 Phase1 epoch 数（冒烟/限时用）")
    parser.add_argument("--phase2-epochs", type=int, default=None,
                        help="覆盖所有训练的 Phase2 epoch 数（冒烟/限时用）")
    parser.add_argument("--train-max-samples", type=int, default=None,
                        help="覆盖所有训练的最大样本数（冒烟用）")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="覆盖 DataLoader 进程数（无 /dev/shm 的环境设 0）")
    parser.add_argument("--modes", type=int, default=None,
                        help="覆盖模型 modes；不指定则用 model.py 里的默认值")
    parser.add_argument("--width", type=int, default=None,
                        help="覆盖模型 width；不指定则用 model.py 里的默认值")
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
    parser.add_argument("--no-final-test", action="store_true",
                        help="跳过流程末尾的 test 集评估")
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="test 评估只用前 N 条样本（0=全部），用于快速冒烟")
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

    # 数据源优先级：--data（显式）> --data_dir（自动发现失败则交给 Phase 0.0 处理）
    data_source = args.data or ""
    if not data_source and args.data_dir and not (args.train_data and args.val_data):
        data_source = args.data_dir

    captain = Captain(
        task_id=args.task,
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        data_source=data_source,
        split_ratios=tuple(args.split),
        split_seed=args.seed,
        train_seed=args.train_seed,
        sigma=args.sigma,
        sigma_source=("calibrated" if args.sigma_calibrated else "assumed"),
        train_overrides={
            k: v for k, v in {
                "phase1_epochs": args.phase1_epochs,
                "phase2_epochs": args.phase2_epochs,
                "max_samples": args.train_max_samples,
                "num_workers": args.num_workers,
                "modes": args.modes,
                "width": args.width,
            }.items() if v is not None
        },
        max_rounds=args.max_rounds,
        inner_loop_limit=args.inner_limit,
        force_model=args.model,
        skip_phase0=args.skip_phase0,
        scout_samples=args.scout_samples,
        scout_topk=args.scout_topk,
        skip_final_test=args.no_final_test,
        eval_samples=args.eval_samples,
        test_has_gt=test_has_gt,
    )
    captain.run()

if __name__ == "__main__":
    main()
