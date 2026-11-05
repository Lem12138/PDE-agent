#!/usr/bin/env python3
# judge.py — 确定性实验判定
#
# 设计原则：判定权归代码，不归 LLM。
#   r  = (current - baseline) / baseline     某一段的相对变化
#   thr = 2 * sigma                          显著性阈值
#
# 规则：
#   - 只统计 baseline > 0 的段（seg3 恒为 0 时自动跳过，不让它干扰判定）
#   - 所有有效段 r <= -thr 且 overall r <= -thr  → IMPROVED
#   - 所有有效段 r >= +thr 且 overall r >= +thr  → DEGRADED
#   - 所有 |r| < thr                              → STAGNANT
#   - 其余                                        → MIXED
#   - 没有可判定的段                              → INVALID
#
# sigma 的来源有两种，必须显式标注：
#   - "calibrated": 由同配置重复实验实测得到（推荐）
#   - "assumed"   : 尚未标定，使用假定值（结论需谨慎）

SEGMENTS = ("seg1", "seg2", "seg3")

VERDICT_IMPROVED = "IMPROVED"
VERDICT_DEGRADED = "DEGRADED"
VERDICT_MIXED = "MIXED"
VERDICT_STAGNANT = "STAGNANT"
VERDICT_INVALID = "INVALID"


def _rel_change(baseline, current):
    """相对变化率；baseline 为 0 或缺失时返回 None（该段不可判定）。"""
    if baseline is None or current is None:
        return None
    try:
        b = float(baseline)
        c = float(current)
    except (TypeError, ValueError):
        return None
    if b == 0:
        return None
    return (c - b) / b


def judge(baseline: dict, current: dict, sigma: float = 0.05,
          sigma_source: str = "assumed", segments=SEGMENTS) -> dict:
    """对比 baseline 与 current 指标，返回确定性判定结果。

    参数
    ----
    baseline, current : dict   至少含 overall 和 seg1/seg2/seg3
    sigma             : float  单次运行的相对噪声水平（例如 0.03 表示 3%）
    sigma_source      : str    "calibrated" 或 "assumed"
    segments          : tuple  参与判定的分段字段名

    返回
    ----
    dict: verdict / threshold / overall / segments / reasons / sigma ...
    """
    if sigma is None or sigma < 0:
        raise ValueError(f"sigma 必须 >= 0，当前 {sigma}")
    threshold = 2.0 * float(sigma)

    # ---- overall ----
    r_overall = _rel_change((baseline or {}).get("overall"), (current or {}).get("overall"))

    # ---- 分段 ----
    seg_detail = {}
    valid_rs = []
    for seg in segments:
        b = (baseline or {}).get(seg)
        c = (current or {}).get(seg)
        r = _rel_change(b, c)
        entry = {"baseline": b, "current": c, "r": r, "usable": r is not None}
        if r is None:
            entry["judgment"] = "不可判定"
            # 说明原因，便于审计
            if b is None or c is None:
                entry["reason"] = "缺少该段指标"
            else:
                entry["reason"] = "baseline 为 0，该段无区分度"
        elif r <= -threshold:
            entry["judgment"] = "改善"
            valid_rs.append(r)
        elif r >= threshold:
            entry["judgment"] = "退化"
            valid_rs.append(r)
        else:
            entry["judgment"] = "持平"
            valid_rs.append(r)
        seg_detail[seg] = entry

    reasons = []
    result = {
        "verdict": VERDICT_INVALID,
        "sigma": float(sigma),
        "sigma_source": sigma_source,
        "threshold": threshold,
        "overall": {
            "baseline": (baseline or {}).get("overall"),
            "current": (current or {}).get("overall"),
            "r": r_overall,
        },
        "segments": seg_detail,
        "valid_segments": [s for s in segments if seg_detail[s]["usable"]],
        "reasons": reasons,
    }

    if not valid_rs or r_overall is None:
        reasons.append("没有可用于判定的指标（全部分段 baseline 为 0 或缺失）")
        return result

    thr_pct = threshold * 100
    all_improved = all(r <= -threshold for r in valid_rs)
    all_degraded = all(r >= threshold for r in valid_rs)
    all_flat = all(abs(r) < threshold for r in valid_rs)

    if all_improved and r_overall <= -threshold:
        verdict = VERDICT_IMPROVED
        reasons.append(f"所有有效段改善且均超过阈值 {thr_pct:.1f}%，overall 同向改善")
    elif all_degraded and r_overall >= threshold:
        verdict = VERDICT_DEGRADED
        reasons.append(f"所有有效段退化且均超过阈值 {thr_pct:.1f}%，overall 同向退化")
    elif all_flat:
        verdict = VERDICT_STAGNANT
        reasons.append(f"所有有效段变化均在阈值 {thr_pct:.1f}% 以内，视为噪声")
        if abs(r_overall) >= threshold:
            reasons.append(
                f"⚠ overall 变化 {r_overall*100:+.1f}% 超过阈值，但各分段均未超阈值，"
                f"指标来源可能不一致，建议核对")
    else:
        verdict = VERDICT_MIXED
        up = [s for s in segments if seg_detail[s]["usable"] and seg_detail[s]["r"] >= threshold]
        down = [s for s in segments if seg_detail[s]["usable"] and seg_detail[s]["r"] <= -threshold]
        if up:
            reasons.append(f"退化的段: {', '.join(up)}")
        if down:
            reasons.append(f"改善的段: {', '.join(down)}")
        if not up and not down:
            reasons.append("各段方向不一致但均未超过阈值")
        # overall 与分段方向相反时单独提示
        if r_overall <= -threshold and all(r >= -threshold for r in valid_rs):
            reasons.append("overall 改善但分段均未改善，需检查指标来源")

    skipped = [s for s in segments if not seg_detail[s]["usable"]]
    if skipped:
        reasons.append(f"跳过不可判定的段: {', '.join(skipped)}（baseline 为 0 或缺失）")

    if sigma_source == "assumed":
        reasons.append(f"⚠ sigma={sigma:.3g} 为假定值，未用重复实验标定")

    result["verdict"] = verdict
    return result


def format_judgement(j: dict) -> str:
    """把判定结果压成一行可读文本，用于日志。"""
    o = j.get("overall", {})
    parts = [f"verdict={j['verdict']}",
             f"overall {o.get('baseline')}→{o.get('current')}",
             f"sigma={j.get('sigma')}({j.get('sigma_source')})",
             f"thr={j.get('threshold')}"]
    for seg, d in j.get("segments", {}).items():
        if d.get("usable"):
            parts.append(f"{seg}:{d['r']*100:+.1f}%")
        else:
            parts.append(f"{seg}:n/a")
    return " | ".join(parts)


if __name__ == "__main__":
    # 自测
    base = {"overall": 8e-6, "seg1": 9e-6, "seg2": 8e-6, "seg3": 0.0}

    cases = {
        "明显改善": {"overall": 6e-6, "seg1": 7e-6, "seg2": 6e-6, "seg3": 0.0},
        "明显退化": {"overall": 1.2e-5, "seg1": 1.3e-5, "seg2": 1.1e-5, "seg3": 0.0},
        "噪声级抖动": {"overall": 8.2e-6, "seg1": 9.1e-6, "seg2": 8.1e-6, "seg3": 0.0},
        "一升一降": {"overall": 8e-6, "seg1": 6e-6, "seg2": 1.1e-5, "seg3": 0.0},
    }
    for name, cur in cases.items():
        r = judge(base, cur, sigma=0.05)
        print(f"{name:10s} -> {format_judgement(r)}")
