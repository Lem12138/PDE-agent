# 论文资料库

本目录用于集中保存项目调研和报告引用的论文原文。

## 已收录

### 申请书参考文献

| 编号 | 论文/资料 | 本地文件 | 与本项目的关系 |
|---:|---|---|---|
| [1] | DeepONet: Learning nonlinear operators for identifying differential equations | [deeponet-1910.03193.pdf](./deeponet-1910.03193.pdf) | 算子学习与 DeepONet 基线理论 |
| [2] | Fourier Neural Operator for Parametric Partial Differential Equations | [fno-2010.08895.pdf](./fno-2010.08895.pdf) | FNO 核心架构与基线依据 |
| [3] | Learning the solution operator of parametric PDEs with physics-informed DeepONets | [pi-deeponet-2103.10974.pdf](./pi-deeponet-2103.10974.pdf) | PDE 物理残差与算子学习结合 |
| [4] | Physics Informed Deep Learning (Part I) | [pinn-1711.10561.pdf](./pinn-1711.10561.pdf) | PINN 物理约束与自动微分方法 |
| [5] | In-Context Operator Learning with Data Prompts for Differential Equation Problems | [icon-2304.07993.pdf](./icon-2304.07993.pdf) | 无需重新训练的上下文算子学习 |
| [6] | PDEBench: An Extensive Benchmark for Scientific Machine Learning | [pdebench-2210.07182.pdf](./pdebench-2210.07182.pdf) | 数据集格式、基线模型与评价规则 |
| [7] | FourCastNet: A Global Data-driven High-resolution Weather Model | [fourcastnet-2202.11214.pdf](./fourcastnet-2202.11214.pdf) | FNO 在大规模科学计算中的应用 |
| [8] | FuXi: A cascade machine learning forecasting system for 15-day global weather forecast | [fuxi-2306.12873.pdf](./fuxi-2306.12873.pdf) | 长时预测、误差累积与级联模型 |
| [9] | LangChain | 官方链接：https://github.com/langchain-ai/langchain | Agent 工作流、工具调用和状态管理 |
| [10] | Fourier Neural Operator with Learned Deformations for PDEs on General Geometries | [geo-fno-2207.05209.pdf](./geo-fno-2207.05209.pdf) | 非规则几何与 FNO 推广 |
| 正文补充 | DeepXDE: A Deep Learning Library for Solving Differential Equations | [deepxde-1907.04502.pdf](./deepxde-1907.04502.pdf) | PINN 工具链、自动化求解与工程实现 |

说明：

- 以上 PDF 主要从 arXiv 下载，部分为正式发表前的预印本版本。
- [5] 的 arXiv 页面明确说明预印本较旧，正式版本发表于 PNAS，DOI 为 `10.1073/pnas.2310142120`。
- [9] LangChain 是软件项目而非论文，因此保留官方 GitHub 地址。

### 中期补充文献

### 2512.16214

| 字段 | 内容 |
|---|---|
| 标题 | PDE-AGENT: A Toolchain-Augmented Multi-Agent Framework for PDE Solving |
| 作者 | Jianming Liu, Ren Zhu, Jian Xu, Kun Ding, Xu-Yao Zhang, Gaofeng Meng, Cheng-Lin Liu |
| arXiv | 2512.16214v2 |
| 版本日期 | 2025-12-22 |
| 本地文件 | `2512.16214.pdf` |
| 页数 | 12 |
| 文件大小 | 1,718,686 bytes |
| SHA-256 | `1b7d12ea18482efd07eca0281ccec2341d131ce7c812671ebd83526fb9e2f229` |
| 来源 | https://arxiv.org/pdf/2512.16214 |

**与本项目的关系：**

- 提出面向 PDE 求解的工具增强多 Agent 框架；
- 使用 Prog-Act 双循环进行动态规划与错误纠正；
- 使用图记忆组织多 Agent 协作过程；
- 使用 Resource-Pool 集中管理运行时产物，并采用工具与参数分离机制；
- 为本项目的 GraphMemory、ResourcePool、内循环修复和外循环重规划设计提供了直接参考。

## 文件命名约定

建议统一使用 arXiv 编号或论文简称命名，例如：

- `2512.16214.pdf`
- `FNO-2021.pdf`
- `DeepONet-2021.pdf`

新增论文时，请在本文件中补充标题、作者、版本日期、来源和与本项目的关系。
