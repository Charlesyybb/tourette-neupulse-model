"""
个体化算法的实用产出 (scripts/individualize_demo.py)
=====================================================

四张表，对应四个决策：

  1. 逐个体处方样例          —— 算法长什么样
  2. 试验设计表 (s × W)      —— 个体化值不值得做
  3. 虚拟队列应答率          —— 能把 59% 推到多少
  4. 设备频率步进要求        —— 硬件规格
"""

from __future__ import annotations

import numpy as np

from fkm.individualize import (
    PeakEstimate,
    Prior,
    breakeven_detuning,
    h_factor,
    population_incremental_gain,
    prescribe,
    shrink,
)

PRIOR = Prior(mu_pop=10.5, sd_pop=1.0)
DEVICE_FIXED = 10.0


def _peak(f, s, W):
    return PeakEstimate(f, s, W, 0.1, 1.0, 5.0, 0.99, "lorentzian")


# --------------------------------------------------------------------------- #
def table1_worked_examples():
    print("=" * 78)
    print("1. 逐个体处方样例   (μ_pop = 10.5 Hz, σ_pop = 1.0 Hz, 设备固定 10 Hz)")
    print("=" * 78)
    cases = [
        ("窄峰、测得准 —— 最该个体化", 11.9, 0.15, 0.5),
        ("窄峰、测不准 —— 收缩救场",   11.9, 0.80, 0.5),
        ("宽峰、测得准 —— 收益有限",   11.9, 0.15, 2.0),
        ("峰频恰在人群均值",           10.5, 0.20, 1.0),
        ("低频个体",                   8.9,  0.20, 0.8),
    ]
    for label, f, s, W in cases:
        rx = prescribe(peak=_peak(f, s, W), prior=PRIOR,
                       device_fixed=DEVICE_FIXED, freq_step=0.5)
        print(f"\n  【{label}】")
        print("  " + rx.summary().replace("\n", "\n  "))
    print("\n  注意第 1、2 行的对比：同一个 ω̂ = 11.9 Hz，测量精度从 0.15 掉到 0.80 Hz，")
    print("  算法自动把处方从 11.9 拉回 11.0 —— 不追噪声。这是收缩项在干活。")
    print()


# --------------------------------------------------------------------------- #
def table2_trial_design():
    print("=" * 78)
    print("2. 试验设计表：个体化相对**最优固定频率**的人群期望增量")
    print("=" * 78)
    print("   （相对设备现用的 10 Hz 还要再加上换频率本身的收益，见 headline_numbers）")
    print()
    ss = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    Ws = [0.5, 1.0, 1.5, 2.0]
    print("        " + "".join(f"{'W=' + str(W):>12}" for W in Ws))
    print("  s(Hz) " + "-" * (12 * len(Ws)))
    for s in ss:
        row = "".join(f"{population_incremental_gain(s, PRIOR, W)['incremental']:>11.1%} "
                      for W in Ws)
        print(f"  {s:>5.1f} {row}")
    print()
    print("  读法：EEG 峰频精度 s 通常在 0.2–0.5 Hz。")
    print("  若 W ≈ 0.5 Hz（窄峰），个体化增量 33–53%，明确值得做。")
    print("  若 W ≈ 2.0 Hz（宽峰），增量降到 7–9%，多半淹没在噪声里。")
    print("  ⟹ **不该对所有人做个体化，该对窄峰人群做。** 这是式 (2.20) 的直接后果：")
    print("     W·ρ_max = F/2，窄峰个体天花板高且窗口窄，正是收益集中的地方。")
    print()
    print("  经验法则（小 u 渐近，见 test_breakeven_small_u_asymptote）：")
    print("    以精度 s 做个体化 ≈ 把固定频率恰好设在失谐 s 处。")
    print("    所以只要 s < 个体的 |ω̄ − 10 Hz|，个体化就在赚。")
    print()


# --------------------------------------------------------------------------- #
def _responder_rate(strategy_freq, ob, W, B, threshold=0.25):
    """式 (2.34)：E = 1 − exp(−B/sqrt(A²+4Δ²))，A = 2W。应答 = E ≥ threshold。"""
    A = 2.0 * W
    d = ob - strategy_freq
    E = 1.0 - np.exp(-B / np.sqrt(A**2 + 4.0 * d**2))
    return E, float(np.mean(E >= threshold))


def table3_virtual_cohort(n=20000, seed=7):
    print("=" * 78)
    print("3. 虚拟队列：三种策略下的应答率")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    ob = rng.normal(PRIOR.mu_pop, PRIOR.sd_pop, n)
    W = np.clip(rng.lognormal(np.log(0.9), 0.45, n), 0.3, 3.0)
    s = 0.3                                  # 假设的 EEG 峰频精度
    f_hat = ob + rng.normal(0.0, s, n)
    om_ind = np.array([shrink(fh, s, PRIOR)[0] for fh in f_hat])

    # 标定 B，使"设备固定 10 Hz"的应答率 = 59%（试验实测值）
    from scipy.optimize import brentq
    B = brentq(lambda b: _responder_rate(DEVICE_FIXED, ob, W, b)[1] - 0.59, 0.01, 20.0)
    print(f"  标定：B = βF = {B:.3f}，使固定 10 Hz 的应答率 = 59%（试验实测）")
    print(f"  假设 EEG 峰频精度 s = {s} Hz，W ~ lognormal(中位数 0.9 Hz)")
    print()
    print(f"  {'策略':<26}{'应答率':>10}{'平均疗效 E':>14}{'相对提升':>12}")
    print("  " + "-" * 62)
    base = None
    for label, freq in [("设备现用 固定 10 Hz", DEVICE_FIXED),
                        ("最优固定 (= μ_pop)", PRIOR.mu_pop),
                        ("收缩个体化", om_ind)]:
        E, rr = _responder_rate(freq, ob, W, B)
        base = rr if base is None else base
        print(f"  {label:<26}{rr:>10.1%}{E.mean():>14.3f}{rr / base - 1:>+12.1%}")
    print()
    print("  ⚠ 这是**演示性标定**，不是预测。B 是反解出来的，不是独立测得的；")
    print("    模型的 E 是抽动率降幅，试验的应答是 YGTSS 降幅，二者只是近似对应。")
    print("    这张表的用途是给出量级和样本量，不能当结果引用。")
    print()

    print("  对 W 分布的敏感性（W 中位数是全流程约束最弱的输入）：")
    print(f"    {'W 中位数':<12}{'固定 10 Hz':>12}{'个体化':>10}{'相对提升':>12}")
    for wmed in [0.6, 0.9, 1.5, 2.5]:
        Wx = np.clip(rng.lognormal(np.log(wmed), 0.45, n), 0.3, 4.0)
        Bx = brentq(lambda b: _responder_rate(DEVICE_FIXED, ob, Wx, b)[1] - 0.59, 0.01, 40.0)
        _, r1 = _responder_rate(DEVICE_FIXED, ob, Wx, Bx)
        _, r2 = _responder_rate(om_ind, ob, Wx, Bx)
        print(f"    {wmed:<12.1f}{r1:>12.1%}{r2:>10.1%}{r2 / r1 - 1:>+12.1%}")
    print("    ⟹ 结论的方向对 W 稳健，量级不稳健。窄峰假设下收益大得多。")
    print()

    # 头对头试验样本量
    _, p1 = _responder_rate(DEVICE_FIXED, ob, W, B)
    _, p2 = _responder_rate(om_ind, ob, W, B)
    za, zb = 1.959964, 0.8416212            # α=0.05 双侧, power=80%
    npg = (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
    print(f"  头对头试验（个体化 vs 固定 10 Hz，应答率 {p1:.1%} → {p2:.1%}）：")
    print(f"    α=0.05 双侧、power=80% ⟹ 每组约 {int(np.ceil(npg))} 人")
    print(f"    交叉设计可显著压低（每人自比），且 online 终点可在单次 session 内读出。")
    print()


# --------------------------------------------------------------------------- #
def table4_device_spec():
    print("=" * 78)
    print("4. 设备频率步进要求")
    print("=" * 78)
    print("   量化误差上界 = step/2，最坏情形损失 = 1 − 1/sqrt(1+(step/2W)²)")
    print()
    steps = [2.0, 1.0, 0.5, 0.25, 0.1]
    Ws = [0.5, 1.0, 2.0]
    print("            " + "".join(f"{'W=' + str(W):>12}" for W in Ws))
    print("  step(Hz)  " + "-" * (12 * len(Ws)))
    for st in steps:
        row = "".join(f"{1 - 1 / np.sqrt(1 + (st / 2 / W) ** 2):>11.1%} " for W in Ws)
        print(f"  {st:>7.2f}   {row}")
    print()
    print("  ⟹ 1.0 Hz 步进在窄峰个体 (W=0.5) 上最坏损失 29% —— 会吃掉个体化收益的大半。")
    print("     0.5 Hz 步进降到 11%，0.25 Hz 步进降到 3%。")
    print("     窄峰个体恰恰是收益最大的那群人（表 2），所以步进要按他们定。")
    print("     **建议硬件规格：频率步进 ≤ 0.25 Hz，覆盖 7–14 Hz。**")
    print()


if __name__ == "__main__":
    table1_worked_examples()
    table2_trial_design()
    table3_virtual_cohort()
    table4_device_spec()
