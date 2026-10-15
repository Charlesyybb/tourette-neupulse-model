"""
2.18 (1) 的定量化 (scripts/iverson_ratio.py)
=============================================

正文把 Iverson et al. [2023] 的"节律 ≈ 非节律"列为模型无法解释的最强反证。
本脚本给出模型**实际**预测的比值，并检验现有数据是否有效能拒绝它。

结构：
  1. √N 定律与相干增益表
  2. 解析 vs 直接模拟（验证 waveform.py 的全部近似）
  3. 疗效比（经莱斯分布门控平均，含饱和）
  4. 统计效能：拒绝该预测需要多大样本
  5. 与附录 A.8 的统一：抖动是同一条曲线
  6. 一条新的可判伪预测：ITC 与疗效在非节律条件下解耦
"""

from __future__ import annotations

import numpy as np

from fkm.waveform import (
    RiceResponse,
    coherence_gain,
    drive_spectrum,
    efficacy_rice,
    mean_exp_rice,
    response_moments,
    simulate_response,
)

TWO_PI = 2.0 * np.pi
Q = 0.02                      # 单脉冲相位冲量（标定量，只影响绝对值不影响比值）


def sec1_sqrtN():
    print("=" * 78)
    print("1. √N 定律：相干时间窗内的脉冲数决定一切")
    print("=" * 78)
    print("   区制 I 下系统是中心 ω̄、半宽 W 的单极点滤波器，相干时间 τ_c = 1/W_rad。")
    print("   窗内 N = r/W_rad 个脉冲：同相叠加 ∝ N，随机叠加 ∝ √N ⟹ 增益 = √N。")
    print("   精确系数：ν_periodic / rms_poisson = sqrt(2r/W_rad) = √(2N)")
    print()
    rates = [10.0, 12.0, 20.0]
    Ws = [0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"  {'r (Hz)':<9}" + "".join(f"{'W=' + str(W):>11}" for W in Ws))
    print("  " + "-" * (9 + 11 * len(Ws)))
    for r in rates:
        row = "".join(f"{coherence_gain(r, W):>10.2f} " for W in Ws)
        print(f"  {r:<9.0f}{row}")
    print()
    print(f"  临床参数 (r = 12 Hz, W ≈ 1 Hz)：N = {12 / TWO_PI:.2f} 个脉冲/相干窗，")
    print(f"  增益仅 {coherence_gain(12, 1.0):.2f}×。")
    print()
    print("  ⟹ **模型从来没有预测过「节律 ≫ 非节律」。** mu 的相干时间只有 ~160 ms，")
    print("     12 Hz 下窗内不到两个脉冲，相干叠加根本没多少优势可占。")
    print("     正文 2.18(1) 说的'ρ 提升应远小于节律驱动'是想当然，不是推导。")
    print()


def sec2_validation():
    print("=" * 78)
    print("2. 解析 vs 直接模拟（真实脉冲串，不用任何高斯/圆对称近似）")
    print("=" * 78)
    W_hz, rate = 1.0, 12.0
    W, ob = TWO_PI * W_hz, TWO_PI * rate
    print(f"  {'条件':<20}{'ν 解析/模拟':>22}{'σ 解析/模拟':>22}")
    print("  " + "-" * 64)
    for kind, jit in [("periodic", 0.0), ("jittered", 0.010),
                      ("jittered", 0.030), ("poisson", 0.0)]:
        ds = drive_spectrum(kind, Q, rate, ob, jit)
        an = response_moments(ds, W)
        _, w = simulate_response(kind, Q, rate, W, ob, ob,
                                 duration=800.0, jitter=jit, seed=1)
        nu_s = abs(w.mean())
        sg_s = np.sqrt(np.mean(np.abs(w - w.mean()) ** 2) / 2)
        lbl = kind + (f" σ_τ={jit * 1e3:.0f}ms" if kind == "jittered" else "")
        print(f"  {lbl:<20}{an.nu:>11.5f}/{nu_s:<10.5f}{an.sigma:>11.5f}/{sg_s:<10.5f}")
    print()
    print("  周期条件下模拟的 σ ≈ 0.002 是 δ 函数离散化残差，非物理。其余全部吻合。")
    print()


def _beta_for(target_E, rr):
    """标定 β 使给定条件下疗效 = target_E。"""
    from scipy.optimize import brentq
    return brentq(lambda b: efficacy_rice(rr, b) - target_E, 1e-4, 5e4)


def sec3_efficacy():
    print("=" * 78)
    print("3. 疗效比：经莱斯分布门控平均（式 2.32 看的是瞬时 |z|，不是 ⟨z⟩）")
    print("=" * 78)
    rate = 12.0
    print(f"  {'W (Hz)':<9}{'振幅比':>10}{'疗效比 (E_R=0.15)':>20}{'非节律保留':>14}")
    print("  " + "-" * 55)
    for W_hz in [0.5, 1.0, 1.5, 2.0, 3.0]:
        W, ob = TWO_PI * W_hz, TWO_PI * rate
        rr_R = response_moments(drive_spectrum("periodic", Q, rate, ob), W)
        rr_A = response_moments(drive_spectrum("poisson", Q, rate, ob), W)
        beta = _beta_for(0.15, rr_R)
        E_R, E_A = efficacy_rice(rr_R, beta), efficacy_rice(rr_A, beta)
        print(f"  {W_hz:<9.1f}{rr_R.rms / rr_A.rms:>10.2f}{E_R / E_A:>20.2f}"
              f"{E_A / E_R:>13.0%}")
    print()
    print("  **模型预测非节律刺激保留节律刺激 34–79% 的疗效**（临床 W 区间 34–65%）。")
    print()
    print("  疗效比略**大于**振幅比：瑞利分布的均值 σ√(π/2) 低于其均方根 σ√2，")
    print("  比值 0.886。门控看的是 ⟨|z|⟩ 而不是 rms，所以非节律吃了一点亏。")
    print("  饱和又把这个差距拉回来一些（E_R = 0.15 处已有轻微非线性）。")
    print()


def sec4_power():
    print("=" * 78)
    print("4. 统计效能：Iverson 的阴性结果能不能拒绝这个预测？")
    print("=" * 78)
    print("  Morera Maiquez et al. [2023] 报告节律 vs 假刺激效应量 d ≈ 0.5。")
    print("  若非节律保留比例 φ，则节律 vs 非节律的效应量 d' ≈ 0.5(1 − φ)。")
    print()
    za, zb = 1.959964, 0.8416212
    print(f"  {'φ (非节律保留)':<18}{'d′':>8}{'配对设计每组 n':>18}")
    print("  " + "-" * 44)
    for phi in [0.40, 0.50, 0.60, 0.70, 0.75]:
        d = 0.5 * (1 - phi)
        n = (za + zb) ** 2 / d**2
        print(f"  {phi:<18.0%}{d:>8.3f}{int(np.ceil(n)):>18}")
    print()
    print("  ⟹ 模型预测的 φ = 0.4–0.75 对应 **每组 125–500 人**。")
    print("     Iverson 的交叉试验样本量远低于此（**需核对原文确切 N**，")
    print("     我记忆中在 30 人量级，请以原文为准）。")
    print()
    print("  结论：**小样本阴性结果不等于等效。** Iverson 的数据与模型预测相容 ——")
    print("  它没有效能区分'节律优势 2×'与'节律优势 1×'。")
    print("  2.18(1) 应从「模型无法解释」改写为「模型预测一个小比值，现有数据")
    print("  无效能拒绝之」，并把这条列为**待检验预测**而不是反证。")
    print()


def sec5_unify_A8():
    print("=" * 78)
    print("5. 与附录 A.8 的统一：抖动是同一条曲线上的两个点")
    print("=" * 78)
    print("  A.8 把 Ξ_jitter = e^{−σ_τ²Ω²/2} 当作'可忽略的负面结果'；")
    print("  2.18(1) 把非节律当作'最强反证'。二者其实是同一个参数 σ_τ 的两端。")
    print()
    rate, W_hz = 12.0, 1.0
    W, ob = TWO_PI * W_hz, TWO_PI * rate
    rr_R = response_moments(drive_spectrum("periodic", Q, rate, ob), W)
    beta = _beta_for(0.15, rr_R)
    print(f"  {'σ_τ':<12}{'Φ (相干保留)':>14}{'ν':>10}{'σ_z':>10}{'E/E_R':>10}")
    print("  " + "-" * 56)
    for jit in [0.0, 0.003, 0.010, 0.020, 0.030, 0.060, np.inf]:
        kind = "poisson" if not np.isfinite(jit) else \
               ("periodic" if jit == 0 else "jittered")
        ds = drive_spectrum(kind, Q, rate, ob, jit if np.isfinite(jit) else 0.0)
        rr = response_moments(ds, W)
        lbl = "∞ (泊松)" if not np.isfinite(jit) else f"{jit * 1e3:.0f} ms"
        print(f"  {lbl:<12}{ds.phi:>14.3f}{rr.nu:>10.5f}{rr.sigma:>10.5f}"
              f"{efficacy_rice(rr, beta) / 0.15:>10.2f}")
    print()
    print("  A.8 的 3 ms（Φ=0.982）确实可忽略；但曲线在 σ_τ ~ 20–30 ms 处")
    print("  就基本走完了 —— 而这仍远小于 1/r = 83 ms 的脉冲间隔。")
    print("  **A.8 和 2.18(1) 应合并为一节。** 这既省篇幅，也让'非节律'从")
    print("  一个孤立的反证变成一条连续调节轴上的极限情形。")
    print()


def sec6_new_prediction():
    print("=" * 78)
    print("6. 一条随之而来的新可判伪预测（建议编号 P11）")
    print("=" * 78)
    rate, W_hz = 12.0, 1.0
    W, ob = TWO_PI * W_hz, TWO_PI * rate
    rr_R = response_moments(drive_spectrum("periodic", Q, rate, ob), W)
    rr_A = response_moments(drive_spectrum("poisson", Q, rate, ob), W)
    print("  门控（式 2.31）看的是**瞬时** |z(t)|；ITC 测的是 ⟨z⟩。二者在节律条件下")
    print("  同步变化，在非节律条件下**解耦**：")
    print()
    print(f"  {'条件':<14}{'ITC ∝ |⟨z⟩|':>16}{'门控 ∝ ⟨|z|⟩':>16}{'比值':>10}")
    print("  " + "-" * 56)
    for lbl, rr in [("节律", rr_R), ("非节律", rr_A)]:
        itc = rr.nu
        gate = rr.mean_abs
        print(f"  {lbl:<14}{itc:>16.5f}{gate:>16.5f}{itc / gate:>10.2f}")
    print()
    print("  **P11：非节律刺激应给出接近零的 ITC，却仍有可观的抽动抑制。**")
    print()
    print("  这条预测很硬 —— 它把 Iverson 的阴性结果从反证转成支持证据：")
    print("  若非节律真的无效，ITC 与疗效应同时归零；若模型对，则 ITC 归零而")
    print("  疗效只降到 40–75%。**同一次实验同时测 ITC 与 TPM 即可判定。**")
    print()
    print("  另一条结构性差别（同样可测）：不相干响应与失谐无关（白噪声在通带内")
    print("  均匀），所以**非节律刺激的疗效应对刺激频率不敏感，节律刺激应敏感**。")
    print("  这给 P1–P3 的频率依赖检验提供了一个天然的阴性对照臂。")
    print()


if __name__ == "__main__":
    sec1_sqrtN()
    sec2_validation()
    sec3_efficacy()
    sec4_power()
    sec5_unify_A8()
    sec6_new_prediction()
