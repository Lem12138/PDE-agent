# pool.py — 资源池，集中管理运行时产物（路径、统计量、指标等），消除 agent 间隐式参数传递
import json
import os
from typing import Any, Optional


class ResourcePool:
    """
    资源池 — 所有 agent 可读写的共享键值存储。

    解决论文中「隐式参数传递」问题：
      - 工具 A 产出了 Python 对象/路径/统计量
      - 工具 B 需要这些值，但无法通过命令行序列化
      - → 全部写入 Pool，统一读写

    支持点号路径深读：pool.get("metrics.seg3")
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self._data: dict = {}
        if filepath and os.path.exists(filepath):
            self._load()

    # ----------------------------------------------------------
    # 基本读写
    # ----------------------------------------------------------
    def put(self, key: str, value: Any) -> None:
        """写入一个键值对。key 支持点号表示嵌套路径。"""
        keys = key.split(".")
        target = self._data
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        """读取键值。key 支持点号深读，如 'metrics.seg3'。"""
        keys = key.split(".")
        target = self._data
        for k in keys:
            if isinstance(target, dict) and k in target:
                target = target[k]
            else:
                return default
        return target

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        keys = key.split(".")
        target = self._data
        for k in keys[:-1]:
            if k not in target:
                return
            target = target[k]
        if keys[-1] in target:
            del target[keys[-1]]
            self._save()

    # ----------------------------------------------------------
    # 批量读写
    # ----------------------------------------------------------
    def update(self, mapping: dict) -> None:
        """批量写入顶层键值。"""
        for k, v in mapping.items():
            self._data[k] = v
        self._save()

    def all(self) -> dict:
        """返回全部数据的浅拷贝，给 LLM 用。"""
        return dict(self._data)

    # ----------------------------------------------------------
    # LLM 格式化
    # ----------------------------------------------------------
    # 注入 LLM 时只展示这些「当前状态」字段。
    # 历史与因果关系由图记忆负责，池里其余键仍可 get()，只是不塞进 prompt。
    CURATED_KEYS = (
        ("best_model", "当前最优模型"),
        ("best_metrics", "当前最优指标"),
        ("best_checkpoint", "当前最优 checkpoint"),
        ("data_split_manifest", "数据切分"),
        ("data_probe", "数据探测"),
    )

    def format_for_llm(self, max_chars: int = 320) -> str:
        """只输出决策相关的当前状态，长值截断。

        旧版把整个池 json.dumps 塞进每个 prompt，包括完整 traceback 和
        嵌套统计，既撑上下文也容易让 LLM 抓着旧信息不放。
        """
        lines = ["### 当前状态（资源池·精选）"]
        shown = False
        for key, label in self.CURATED_KEYS:
            v = self.get(key)
            if v is None:
                continue
            text = (json.dumps(v, ensure_ascii=False, default=str)
                    if isinstance(v, (dict, list)) else str(v))
            if len(text) > max_chars:
                text = text[:max_chars] + " …（已截断）"
            lines.append(f"- **{label}**: {text}")
            shown = True

        err = self.get("last_error")
        if err:
            lines.append(f"- **最近错误**: {str(err)[-max_chars:]}")
            shown = True

        return "\n".join(lines) if shown else "（暂无状态）"

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------
    def _save(self) -> None:
        if self.filepath:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2, default=str)

    def _load(self) -> None:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception as e:
            print(f"[ResourcePool] 加载失败: {e}")
