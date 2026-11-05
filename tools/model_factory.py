#!/usr/bin/env python3
# model_factory.py — 模型实例化的唯一入口
#
# train.py 和 Captain 的行为签名检查都必须走这里，否则会出现
# 「检查用的模型」和「训练用的模型」不是同一个东西的情况。
#
# 设计要点：只把**显式给出**的超参传给构造函数，其余交给模型自己的默认值。
# 这样 Dev 修改 model.py 里的默认值才会真正生效。

def _clean(hp):
    return {k: v for k, v in (hp or {}).items() if v is not None}


def build_model(ModelClass, hp=None, T_IN=10, N_spatial=256):
    """按统一策略实例化模型。

    ModelClass : 模型类
    hp         : 显式覆盖的超参（modes / width / n_layers），None 表示用模型默认值
    T_IN       : 输入时间步
    N_spatial  : 空间分辨率

    依次尝试多套构造签名，返回第一个成功的实例。
    """
    hp = _clean(hp)
    modes = hp.get("modes")
    width = hp.get("width")
    n_layers = hp.get("n_layers")

    def kw(**base):
        out = dict(base)
        for key, val in (("modes", modes), ("width", width), ("n_layers", n_layers)):
            if val is not None:
                out[key] = val
        return out

    patterns = [
        lambda: ModelClass(**kw(in_channels=T_IN, out_channels=1)),
        lambda: ModelClass(**kw(in_channels=T_IN, out_channels=1, n_layers=n_layers or 4)),
        lambda: ModelClass(**kw(in_channels=T_IN, out_channels=1,
                                n_layers=n_layers or 4, n_scales=3)),
        lambda: ModelClass(**kw(num_channels=1, initial_step=T_IN)),
        lambda: ModelClass(**kw(in_channels=T_IN, out_channels=1,
                                base_width=width or 32, depth=n_layers or 4)),
        lambda: ModelClass(**kw(in_channels=T_IN, out_channels=1, init_features=width or 32)),
        lambda: ModelClass(**kw(in_channels=T_IN, width=width or 32,
                                n_layers=n_layers or 4, n_basis=128, nx=N_spatial)),
        lambda: ModelClass(),
    ]

    last_err = None
    for pat in patterns:
        try:
            return pat()
        except (TypeError, ValueError) as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Cannot instantiate {getattr(ModelClass, '__name__', ModelClass)} "
        f"with any known pattern (last error: {last_err})"
    )
