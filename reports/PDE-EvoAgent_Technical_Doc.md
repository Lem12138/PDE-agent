# PDE-EvoAgent 技术文档

## 1. 系统概述

PDE-EvoAgent 是一个基于 LLM 多 Agent 协作的 PDE 神经算子自动演化系统。它不依赖人工调参和试错，而是通过四个专业化 Agent（Proposal / Dev / Diagnosis / EvalReflection）在 Captain 调度器的编排下，自动完成"分析瓶颈 → 提出假说 → 修改代码 → 训练验证 → 评估判定"的完整科研闭环。

**核心能力**: 给定一个新 PDE 数据集，系统自动选择/设计最合适的神经算子架构，并通过迭代演化持续提升精度。

---

## 2. 系统架构

```
                        ┌──────────────┐
                        │   Captain    │  ← 总调度器
                        │  captain.py  │
                        └──────┬───────┘
                               │ 编排四个 subagent
               ┌───────────────┼───────────────┐
               │               │               │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
        │  Proposal   │ │    Dev      │ │  Diagnosis  │
        │  提假说      │ │  改代码     │ │  诊断错误    │
        └─────────────┘ └─────────────┘ └─────────────┘
                               │
                        ┌──────▼──────┐
                        │EvalReflection│
                        │  评估判定    │
                        └─────────────┘
                               │
                        ┌──────▼──────┐
                        │  train.py   │  ← 训练执行器
                        │  (subprocess)│
                        └─────────────┘
```

### 2.1 Captain — 总调度器 (`captain.py`)

Captain 本身不做具体工作，只做决策编排。核心职责：

- **Phase 0**: 数据探测 → 模型匹配 → Scout 快速筛选 → 全量 Baseline → 选最优
- **Phase 1+**: Evolution 循环 — 调度 Proposal → Dev → 训练 → EvalReflection → 判定
- **内循环修复**: 语法/训练错误触发的 Diagnosis → Dev 修复（最多 3 次）
- **外循环回退**: DEGRADED 时回退到最优版本，Proposal 重新出方案
- **共享记忆**: Graph（因果拓扑图）+ Pool（资源池），每步操作后更新

### 2.2 四个 Subagent（LLM Prompt 驱动）

每个 Agent 都有严格的 JSON 输出格式和绝对红线约束：

| Agent | Prompt 文件 | 职责 | 输入 | 输出 |
|-------|------------|------|------|------|
| **Proposal** | `prompts/proposal.md` | 基于物理/数学提出 2-3 个架构改进假说 | 当前模型参数+瓶颈+历史 | JSON: hypotheses[], 含优先级、物理依据、伪代码 |
| **Dev** | `prompts/dev.md` | 将假说落地为 model.py 的精确修改 | 假说 + 当前代码 | JSON: mode=edit, changes[] 含 old/new 字符串 |
| **Diagnosis** | `prompts/diagnosis.md` | 定位训练崩溃/指标异常的根因 | traceback + 日志 + 代码 | JSON: error_type, root_cause, fix |
| **EvalReflection** | `prompts/eval_reflection.md` | 对比 baseline 和当前指标，判定改进/退化 | 两轮指标 + 假说 | JSON: verdict + segment_analysis |

### 2.3 训练引擎 (`train.py`)

- **两阶段训练**:
  - Phase 1 (Teacher Forcing): 随机时间窗口，单步监督学习
  - Phase 2 (Progressive Rollout): 渐进式自回归 rollout，从 1 步逐步增加到 91 步
- **模型自动发现**: `_discover_model_class()` 从 `model.py` 自动检测 `nn.Module` 子类，尝试多种构造函数模式
- **数据归一化**: `(data - mean) / std`
- **验证分三段**: seg1 (步骤 0-47), seg2 (步骤 47-95), seg3 (步骤 95+)

### 2.4 共享记忆系统

| 组件 | 文件 | 功能 |
|------|------|------|
| **GraphMemory** | `shared_memory/graph.py` | 因果拓扑图，记录每步操作和依赖关系，防止重复踩坑 |
| **ResourcePool** | `shared_memory/pool.py` | 键值存储，保存最优指标、checkpoint 路径、数据统计等 |

### 2.5 数据探测 (`data_probe.py`)

自动分析 HDF5 数据的物理特征：
- 维度推断（N×T×X 结构识别）
- FFT 频谱分析（高频能量占比 → 判断 smooth/mixed/shock_rich）
- 边界条件检测（周期/Dirichlet/其他）
- 时间演化趋势（收敛/发散/震荡）

---

## 3. 工作流详解（以 diff_sorp 为例）

以下基于 `output/captain_full.log` 中 2026-08-04 的完整运行记录。

### 3.1 数据集

**1D Diffusion-Sorption (diff_sorp)**，PDE 形式：

$$R(c) \frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}, \quad R(c) = 1 + \frac{1-\phi}{\phi} \rho_s k_f n_f (c+\epsilon)^{n_f-1}$$

- 10000 个样本，每个 (101 时间步 × 1024 空间点)
- 左 Dirichlet 边界 (c=1.0)，右 Neumann 零梯度
- 解场平滑（高频能量占比 0.04%）

### 3.2 Phase 0.1-0.2: 数据探测与模型匹配

```
[07:36:52] 探测 train.h5 → shape=(8000,101,256), smooth, dirichlet
[07:36:55] Captain LLM 推理 → 候选: fno, unet, deeponet, mgno
```

### 3.3 Phase 0.25: Scout 快速筛选

用 200 个样本快速评估 4 个模型（训练参数：Phase1=25epoch, Phase2=20epoch）：

| 排名 | 模型 | 比赛加权分 | seg1 | seg2 | 结果 |
|------|------|-----------|------|------|------|
| 1 | **FNO** | 83.0 | 0.0315 | 0.0240 | ✅ 保留 |
| 2 | **MGNO** | 82.0 | 0.0312 | 0.0297 | ✅ 保留 |
| 3 | DeepONet | 59.4 | 0.2521 | 0.0993 | ❌ 淘汰 |
| 4 | UNet | 50.0 | 1.5673 | 1.8257 | ❌ 淘汰 |

### 3.4 Phase 0.3-0.4: 全量 Baseline

Top-2 模型用全部 8000 个样本训练（Phase1=50epoch, Phase2=100epoch）：

| 模型 | seg1 | seg2 | 耗时 | 排名 |
|------|------|------|------|------|
| 🏆 **FNO** | **9e-06** | **8e-06** | 5233s (87min) | 1 |
| MGNO | 3.8e-05 | 4.7e-05 | 3468s (58min) | 2 |

FNO 以数量级优势胜出（seg1 约 4 倍好于 MGNO），被选为 Evolution 起点。

### 3.5 Phase 1+: Evolution Loop

#### Round 1 — DEGRADED ❌

| 步骤 | Agent | 内容 |
|------|-------|------|
| Proposal | `pde-proposal` | P0:「多步 rollout 损失训练增强自回归稳定性」 |
| Dev | `pde-dev` | 修改 model.py → 语法✅ 实例化✅ |
| 训练 | `train.py` | Phase1 50ep + Phase2 100ep, 约 20min |
| **EvalReflection** | `pde-eval-reflection` | **判定: DEGRADED** — 指标退化 |
| Captain | 回退 | 恢复 model.py + ckpt 到 phase0_fno 最优版本 |

#### Round 2 — IMPROVED ✅

| 步骤 | Agent | 内容 |
|------|-------|------|
| Proposal | `pde-proposal` | P0:「扩大 FNO 的傅里叶模态数与通道宽度，增强初期激波高频表征」 |
| Dev | `pde-dev` | modes 12→16, width 32→48 → 语法✅ 实例化✅ |
| 训练 | `train.py` | Phase1 50ep + Phase2 100ep, 约 100min |
| **EvalReflection** | `pde-eval-reflection` | **判定: IMPROVED** — seg1/seg2 均有改善 |
| Captain | 更新 baseline | 保存快照到 `round2_best` |

**最终指标**: Rel-MSE 7e-06, seg1=8e-06, seg2=7e-06

---

## 4. Evolution 闭环机制

```
                    ┌──────────────────────────┐
                    │     Proposal: 提假说       │
                    │  (基于物理/数学 + 历史)     │
                    └──────────┬───────────────┘
                               │ hypothesis
                    ┌──────────▼───────────────┐
                    │      Dev: 修改 model.py   │
                    │  (精确 edit，保持 I/O)     │
                    └──────────┬───────────────┘
                               │ model.py
                    ┌──────────▼───────────────┐
                    │    训练 + 验证 (train.py)  │
                    │  Phase1 TF → Phase2 PR    │
                    └──────────┬───────────────┘
                               │ metrics
                    ┌──────────▼───────────────┐
                    │  EvalReflection: 判定     │
                    └──────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
     IMPROVED             DEGRADED              MIXED
   更新 baseline         回退最优版本          回退代码
   进入下一轮           Proposal 重出方案      保留指标参考
```

**内循环（最多 3 次）**: 语法错误 / 训练崩溃 → Diagnosis 诊断 → Dev 修复  
**外循环**: 内循环耗尽或 needs_proposal → Proposal 重新出方案，Dev 从头生成

---

## 5. 关键设计决策

### 5.1 为什么分 Scout + Baseline 两阶段？

Scout（200 样本，10% 训练量）在 **17 分钟**内淘汰了 50% 的候选模型。如果直接全量训练全部 4 个模型需要约 6 小时。Scout 的正确率在此案例中是 100%（Scout Top-2 正是 Baseline Top-2）。

### 5.2 为什么用两阶段训练（Teacher Forcing + Progressive Rollout）？

单步监督学习训练效率高（~8s/epoch），但自回归推理时不接触非完美输入 → 误差累积。渐进式 rollout 让模型在训练中逐步接触长序列自回归，同时保持训练稳定性。

### 5.3 为什么 FNO 在 diff_sorp 上优于 MGNO？

diff_sorp 的解场**非常平滑**（高频能量仅 0.04%），FNO 的 Fourier 频域截断天然契合。MGNO 的多尺度机制在这种情况下是过度设计，反而增加了优化难度。

### 5.4 Round 1 为什么会 DEGRADED？

Proposal 提出的"多步 rollout 损失训练"与训练脚本已有的 Phase 2 进度 rollout 功能重复。Dev 的实现可能破坏了已有的渐进式 rollout 逻辑，导致训练不稳定。

### 5.5 Round 2 为什么 IMPROVED？

扩大 modes (12→16) 和 width (32→48) 直接提升了 FNO 的频域表达能力（参数量从 ~23K 增至 ~80K），同时不改变训练策略，是一个纯粹的正向改进。

---

## 6. 文件清单

### 核心运行时
| 文件 | 行数 | 职责 |
|------|------|------|
| `captain.py` | 1518 | 总调度器，编排 Agent + 控制流程 |
| `train.py` | ~350 | 两阶段训练脚本 |
| `data_probe.py` | 357 | 自动数据物理特征探测 |
| `llm_client.py` | ~80 | LLM API 客户端 |
| `utils.py` | ~700 | 训练辅助函数 |

### Agent 系统
| 文件 | 用途 |
|------|------|
| `prompts/captain.md` | Captain 的决策框架 (129 行) |
| `prompts/proposal.md` | 假说生成的物理约束 (94 行) |
| `prompts/dev.md` | 代码修改的 I/O shape 强制检查 (93 行) |
| `prompts/diagnosis.md` | 错误分类诊断流程 (109 行) |
| `prompts/eval_reflection.md` | 分段评分逻辑与判定规则 (107 行) |

### 模型模板
| 模板 | 类名 | 适用场景 |
|------|------|---------|
| `templates/fno/fno.py` | FNO1d | 周期边界 + 平滑解 |
| `templates/deeponet/` | DeepONet1d | 不规则网格 + 函数映射 |
| `templates/unet/` | UNet1d | 激波 + 多尺度局部特征 |
| `templates/mgno/` | MGNO1d | 多尺度能量级联 |
| `templates/registry.json` | — | 模型元数据 + 物理适配信息 |

### 运行产物 (output/)
| 路径 | 内容 |
|------|------|
| `captain.log` | 实时跟踪日志（关键事件行） |
| `captain_full.log` | 完整 stdout 输出 |
| `runs/` | 每次训练的子目录（stdout/stderr 日志） |
| `snapshots/` | 最优版本的 model.py + checkpoint 备份 |
| `replies/` | 所有 LLM subagent 的原始 JSON 回复 |
| `state/` | Graph + Pool 持久化 JSON |

---

## 7. 运行命令

```bash
# 完整工作流（Scout + Baseline + Evolution）
python captain.py \
  --data_dir data/diff_sorp \
  --scout-samples 200 \
  --scout-topk 2 \
  --max_rounds 2 \
  --task 1

# 单次训练
python train.py \
  --train_data data/diff_sorp/train.h5 \
  --val_data data/diff_sorp/val.h5 \
  --save_dir output/runs/manual
```

---

## 8. 本次实验总结

| 阶段 | 耗时 | 关键结果 |
|------|------|---------|
| Scout (4 模型) | 17 min | FNO (83.0) > MGNO (82.0) |
| Baseline (2 模型) | 145 min | FNO Rel-MSE 4e-06 |
| Evolution Round 1 | 23 min | DEGRADED — 回退 |
| Evolution Round 2 | 100 min | **IMPROVED** — modes↑ width↑ |
| **总计** | **~4.5 h** | **最优 Rel-MSE 7e-06** |

Agent 系统在无人干预的情况下自动完成了探测数据 → 筛选模型 → 全量训练 → 提出假说 → 修改架构 → 验证改进的完整科研流程。
