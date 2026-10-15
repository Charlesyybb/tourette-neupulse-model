"""
个体化算法测试 (tests/test_individualize.py)
=============================================

分三层：
  A. 谱估计能否从含 1/f 背景的合成谱里恢复已知峰
  B. 收缩解的极限行为与最优性命题
  C. 端到端处方的降级行为
"""

from __future__ import annotations

import numpy as np
import pytest

from fkm.individualize import (
    MU_BAND,
    Prior,
    breakeven_detuning,
    estimate_mu_peak,
    expected_gain,
    fit_aperiodic,
    h_factor,
    population_incremental_gain,
    prescribe,
    quantise,
    shrink,
)


# --------------------------------------------------------------------------- #
# 合成静息谱
# --------------------------------------------------------------------------- #
def synth_psd(f_peak=10.8, W=0.9, amp=1.0, exponent=1.6, offset=0.0,
              noise=0.0, fmin=2.0, fmax=40.0, df=0.25, seed=0, shape="lorentzian"):
    """1/f 背景 + 洛伦兹(或高斯) mu 峰 + 乘性对数噪声。"""
    rng = np.random.default_rng(seed)
    f = np.arange(fmin, fmax + df, df)
    bg = 10.0 ** (offset - exponent * np.log10(f))
    if shape == "lorentzian":
        pk = amp * bg[np.argmin(abs(f - f_peak))] / (1.0 + ((f - f_peak) / W) ** 2)
    else:
        sig = W / np.sqrt(2 * np.log(2))
        pk = amp * bg[np.argmin(abs(f - f_peak))] * np.exp(-((f - f_peak) ** 2) / (2 * sig**2))
    psd = bg + pk
    if noise > 0:
        psd = psd * np.exp(rng.normal(0.0, noise, f.size))
    return f, psd


# =========================================================================== #
# A. 谱估计
# =========================================================================== #
class TestSpectralEstimation:
    def test_aperiodic_recovers_exponent(self):
        """1/f 拟合应恢复已知指数，且不被 mu 峰拉偏。"""
        f, p = synth_psd(exponent=1.6, amp=3.0)
        (off, chi), _ = fit_aperiodic(f, p)
        assert chi == pytest.approx(1.6, abs=0.1)

    @pytest.mark.parametrize("f_true,W_true", [(9.4, 0.6), (10.8, 0.9), (12.1, 1.5)])
    def test_recovers_peak_noiseless(self, f_true, W_true):
        """无噪声：峰位与半宽应精确恢复。"""
        f, p = synth_psd(f_peak=f_true, W=W_true, amp=4.0)
        est = estimate_mu_peak(f, p)
        assert est.ok, est.reasons
        assert est.f_peak == pytest.approx(f_true, abs=0.08)
        assert est.hwhm == pytest.approx(W_true, abs=0.15)
        assert est.r2 > 0.98

    def test_recovers_peak_with_noise(self):
        """含噪声：峰位仍在 EEG 典型精度 (±0.2–0.5 Hz) 内，且标准误量级合理。"""
        errs, ses = [], []
        for seed in range(30):
            f, p = synth_psd(f_peak=10.8, W=0.9, amp=4.0, noise=0.25, seed=seed)
            est = estimate_mu_peak(f, p)
            if np.isfinite(est.f_peak):
                errs.append(est.f_peak - 10.8)
                ses.append(est.se_f_peak)
        errs, ses = np.array(errs), np.array(ses)
        assert len(errs) >= 25
        assert abs(errs.mean()) < 0.15                     # 无系统偏差
        assert errs.std() < 0.5                            # 在 EEG 典型精度内
        assert np.median(ses) == pytest.approx(errs.std(), rel=0.8)  # SE 标定合理

    def test_A_is_twice_hwhm(self):
        """易感率恒等式：A = 2W。静息谱峰宽直接给出调谐曲线宽度。"""
        f, p = synth_psd(W=0.8, amp=4.0)
        est = estimate_mu_peak(f, p)
        assert est.A == pytest.approx(2 * est.hwhm, abs=1e-12)

    def test_shape_comparison_is_a_model_test(self):
        """
        式 (2.17) 预测洛伦兹。用洛伦兹生成的数据上，洛伦兹拟合应优于高斯 ——
        这条对比在真实数据上就是一次零成本的模型检验。
        """
        f, p = synth_psd(amp=6.0, shape="lorentzian")
        est = estimate_mu_peak(f, p, shape="both")
        assert est.r2 > est.r2_alt

    def test_flags_absent_peak(self):
        """纯 1/f 无峰：应拒绝而不是硬凑一个峰位。"""
        f, p = synth_psd(amp=0.0, noise=0.1)
        est = estimate_mu_peak(f, p)
        assert not est.ok

    def test_flags_edge_peak(self):
        """峰贴近搜索窗边界（多半是 alpha/beta 的肩）应告警。"""
        f, p = synth_psd(f_peak=13.9, W=0.5, amp=4.0)
        est = estimate_mu_peak(f, p)
        assert any("边界" in r for r in est.reasons)


# =========================================================================== #
# B. 收缩与最优性
# =========================================================================== #
class TestShrinkage:
    def test_limits(self):
        """s→0 ⟹ λ→1（完全个体化）；s→∞ ⟹ λ→0（退化为 μ_pop）。"""
        pr = Prior(mu_pop=10.5, sd_pop=1.0)
        m, tau, lam = shrink(12.0, 1e-6, pr)
        assert lam == pytest.approx(1.0, abs=1e-6) and m == pytest.approx(12.0, abs=1e-4)
        m, tau, lam = shrink(12.0, 1e4, pr)
        assert lam < 1e-6 and m == pytest.approx(pr.mu_pop, abs=1e-3)

    def test_posterior_mean_is_bayes_optimal(self):
        """
        核心命题：对称后验 × 对称单峰调谐曲线 ⟹ argmax_Ω E[g(ω̄−Ω)] = 后验均值。
        数值验证：在 Ω 网格上搜索，峰位必须落在后验均值。
        """
        mean, tau, W = 10.8, 0.4, 0.9
        grid = np.linspace(mean - 3, mean + 3, 6001)
        g = np.array([expected_gain(O, mean, tau, W) for O in grid])
        assert grid[int(np.argmax(g))] == pytest.approx(mean, abs=2e-3)

    def test_optimality_holds_under_efficacy_transform(self):
        """命题与疗效变换无关：对 E = 1−exp(−Bρ) 取期望，argmax 不变。"""
        mean, tau, W, B = 10.8, 0.4, 0.9, 1.3
        x = np.linspace(mean - 6 * tau, mean + 6 * tau, 4001)
        wgt = np.exp(-0.5 * ((x - mean) / tau) ** 2)
        grid = np.linspace(mean - 2, mean + 2, 2001)
        vals = []
        for O in grid:
            rho = 1.0 / np.sqrt(1.0 + ((x - O) / W) ** 2)
            vals.append(np.trapezoid(wgt * (1 - np.exp(-B * rho)), x))
        assert grid[int(np.argmax(vals))] == pytest.approx(mean, abs=5e-3)

    def test_shrinkage_dominates_both_extremes(self):
        """
        收缩解恒不劣于"完全个体化"与"固定 μ_pop"两个极端。
        这是本模块不设"是否个体化"开关的理由。
        """
        pr = Prior(10.5, 1.0)
        rng = np.random.default_rng(1)
        for _ in range(200):
            f_hat = rng.normal(10.5, 1.5)
            s = rng.uniform(0.05, 1.5)
            W = rng.uniform(0.4, 2.0)
            mean, tau, _ = shrink(f_hat, s, pr)
            g_shrink = expected_gain(mean, mean, tau, W)
            g_raw = expected_gain(f_hat, mean, tau, W)
            g_fix = expected_gain(pr.mu_pop, mean, tau, W)
            assert g_shrink >= g_raw - 1e-9
            assert g_shrink >= g_fix - 1e-9

    def test_h_factor_properties(self):
        """h(0)=1；单调递减；u→∞ 时 →0。"""
        assert h_factor(0.0) == pytest.approx(1.0, abs=1e-9)
        us = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        hs = [h_factor(u) for u in us]
        assert all(hs[i] > hs[i + 1] for i in range(len(hs) - 1))
        assert hs[-1] < 0.35

    def test_breakeven_monotone(self):
        """不确定度越大，个体化的等效失谐 d* 越大（越不划算）。"""
        ds = [breakeven_detuning(u) for u in [0.1, 0.3, 0.6, 1.0, 2.0]]
        assert all(ds[i] < ds[i + 1] for i in range(len(ds) - 1))

    def test_breakeven_small_u_asymptote(self):
        """
        小 u 展开：h ≈ 1 − u²/2 ⟹ 1/h² ≈ 1 + u² ⟹ d*/W ≈ u。
        即"以不确定度 u 做个体化"约等于"固定频率恰好失谐 u 个 W"。
        这个渐近关系本身就是个好用的经验法则。
        """
        for u in [0.05, 0.1, 0.2]:
            assert breakeven_detuning(u) == pytest.approx(u, rel=0.05)

    def test_population_gain_nonnegative_and_vanishing(self):
        """人群增量收益恒非负；s→∞ 时归零；s→0 时最大。"""
        pr = Prior(10.5, 1.0)
        g0 = population_incremental_gain(1e-6, pr, W=1.0)["incremental"]
        g1 = population_incremental_gain(0.5, pr, W=1.0)["incremental"]
        g2 = population_incremental_gain(50.0, pr, W=1.0)["incremental"]
        assert g0 > g1 > g2 >= -1e-9
        assert g2 == pytest.approx(0.0, abs=1e-3)


# =========================================================================== #
# C. 端到端
# =========================================================================== #
class TestPrescribe:
    def test_end_to_end_recovery(self):
        """完整链路：合成谱 → 建议频率应接近真峰位。"""
        f, p = synth_psd(f_peak=11.6, W=0.8, amp=5.0, noise=0.15, seed=3)
        rx = prescribe(f, p, prior=Prior(10.5, 1.0), freq_step=None)
        assert abs(rx.omega_star - 11.6) < 0.5
        assert rx.incremental_vs_device > 0        # 优于设备固定 10 Hz

    def test_quantisation_accounted(self):
        """设备频率步进的损失必须被显式核算，而不是悄悄吞掉。"""
        f, p = synth_psd(f_peak=11.3, W=0.5, amp=5.0)
        coarse = prescribe(f, p, freq_step=1.0)
        fine = prescribe(f, p, freq_step=None)
        assert coarse.quantisation_loss >= fine.quantisation_loss
        assert coarse.omega_star == pytest.approx(round(coarse.omega_bayes), abs=1e-9)

    def test_no_peak_degrades_to_population_mean(self):
        """
        纯 1/f 无峰：拟合仍会吐出一个垃圾峰位，但 SE 极大 ⟹ λ→0 ⟹ 自动落到 μ_pop。
        **不需要显式的无峰分支** —— 这是收缩解设计的关键验证。
        """
        f, p = synth_psd(amp=0.0, noise=0.1)
        pr = Prior(10.5, 1.0)
        rx = prescribe(f, p, prior=pr, freq_step=None)
        assert rx.fallback
        assert rx.omega_star == pytest.approx(pr.mu_pop, abs=0.25)
        assert not rx.ok_reasons_empty if hasattr(rx, "ok_reasons_empty") else True

    def test_fallback_target_is_configurable(self):
        """
        硬回退（拟合崩溃）目标可选：population（统计正确）或 device（保守）。
        默认 population —— 没有个体信息时最优猜测是人群均值，不是设备出厂值。
        """
        from fkm.individualize import PeakEstimate
        dead = PeakEstimate(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, "lorentzian", ok=False, reasons=["拟合失败"])
        pr = Prior(10.5, 1.0)
        assert prescribe(peak=dead, prior=pr, freq_step=None).omega_star == \
            pytest.approx(10.5)
        assert prescribe(peak=dead, prior=pr, freq_step=None,
                         fallback_to="device", device_fixed=10.0).omega_star == \
            pytest.approx(10.0)

    def test_graceful_degradation_no_branching(self):
        """
        大 s 不需要额外分支：λ 自动压向 0，收缩解自然退化为 μ_pop。
        这验证了"降级由收缩系数自动完成"的设计。
        """
        from fkm.individualize import PeakEstimate
        pr = Prior(10.5, 1.0)
        bad = PeakEstimate(f_peak=13.0, se_f_peak=6.0, hwhm=1.0, se_hwhm=0.5,
                           amplitude=1.0, snr=1.0, r2=0.5, shape="lorentzian",
                           ok=False, reasons=["低信噪比"])
        rx = prescribe(peak=bad, prior=pr, freq_step=None)
        assert rx.omega_star == pytest.approx(pr.mu_pop, abs=0.1)
        assert rx.lam < 0.1 and rx.fallback

    def test_gains_are_ordered(self):
        """个体化 ≥ 最优固定；且设备 10 Hz 不优于最优固定（当 μ_pop ≠ 10）。"""
        f, p = synth_psd(f_peak=11.8, W=0.7, amp=5.0)
        rx = prescribe(f, p, prior=Prior(10.5, 1.0), freq_step=None)
        assert rx.gain_individualised >= rx.gain_best_fixed - 1e-9
        assert rx.gain_best_fixed >= rx.gain_device_fixed - 1e-9

    def test_quantise_helper(self):
        assert quantise(10.73, 0.5) == pytest.approx(10.5)
        assert quantise(10.76, 0.5) == pytest.approx(11.0)
        assert quantise(10.73, None) == pytest.approx(10.73)
