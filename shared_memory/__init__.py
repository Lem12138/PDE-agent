# shared_memory/ — 共享记忆模块
#
# GraphMemory — 有向图记忆
#   追踪「做了什么」和「操作间的数据依赖」。
#   节点 = 一次操作（探测数据、加载ckpt、训练、评估…）
#   边   = 产物流向（A的输出→B的输入）
#   用途：出错时沿边回溯找根因，向下游追溯找被污染的步骤。
#
# ResourcePool — 资源池
#   集中管理运行时产物（路径、统计量、指标、Python对象…）。
#   解决 agent 间隐式参数无法通过命令行传递的问题。
#   用途：所有 agent 读写同一份键值池，不依赖口头传递。
#
from .graph import GraphMemory
from .pool import ResourcePool

__all__ = ["GraphMemory", "ResourcePool"]
