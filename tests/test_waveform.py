"""
波形与相干增益测试 (tests/test_waveform.py)
============================================

重点是第 3 组：**解析结果 vs 真实脉冲串直接模拟**。waveform.py 的推导用了
圆对称复高斯近似与旋转波近似，必须独立验证，否则 2.18(1) 的重写没有依据。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from fkm.observation import jitter_factor
from fkm.waveform import (
    TWO_PI,
    RiceResponse,
    coherence_gain,
    drive_spectrum,
    efficacy_rice,
    mean_exp_rice,
    pulse_times,
    response_moments,
    simulate_response,
)

Q, RATE, W_HZ = 0.02, 12.0, 1.0
W, OB = TWO_PI * W_HZ, TWO_PI * RATE


# =========================================================================== #
# 1. 驱动谱分解
# =========================================================================== #
class TestDriveSpectrum:
    def test_periodic_is_fully_coherent(self):
        ds = drive_spectrum("periodic", Q, RATE, OB)
        assert ds.phi == 1.0
        assert ds.F_coherent == pytest.approx(Q * RATE)
        assert ds.S_incoherent == pytest.approx(0.0)

    def test_poisson_is_fully_incoherent(self):
        """泊松串：无谱线，连续谱为散粒噪声 q²r。"""
        ds = drive_spectrum("poisson", Q, RATE, OB)
        assert ds.phi == 0.0
        assert ds.F_coherent == pytest.approx(0.0)
        assert ds.S_incoherent == pytest.approx(Q**2 * RATE)

    @pytest.mark.parametrize("jit", [0.002, 0.01, 0.03, 0.1])
    def test_jitter_interpolates_monotonically(self, jit):
        ds = drive_spectrum("jittered", Q, RATE, OB, jit)
        assert 0.0 <= ds.phi <= 1.0
        assert ds.F_coherent == pytest.approx(Q * RATE * ds.phi)
        assert ds.S_incoherent == pytest.approx(Q**2 * RATE * (1 - ds.phi**2))

    def test_jitter_phi_equals_appendix_A8(self):
        """
        **跨模块一致性**：drive_spectrum 的 Φ 必须与 式 (A.33) 的 Ξ_jitter 相同。
        这正是"A.8 与 2.18(1) 是同一条曲线"的形式化表述。
        """
        for jit in [0.003, 0.01, 0.03]:
            assert drive_spectrum("jittered", Q, RATE, OB, jit).phi == \
                pytest.approx(jitter_factor(jit, OB), rel=1e-12)

    def test_monotone_in_jitter(self):
        phis = [drive_spectrum("jittered", Q, RATE, OB, j).phi
                for j in [0.001, 0.005, 0.02, 0.05]]
        assert all(phis[i] > phis[i + 1] for i in range(len(phis) - 1))


# =========================================================================== #
# 2. 线性响应
# =========================================================================== #
class TestResponseMoments:
    def test_coherent_part_matches_regime1(self):
        """相干分量必须与 式 (2.16) 一致：ν = F/sqrt(A²+4Δ²)，A = 2W。"""
        from fkm.core import rho_regime1
        ds = drive_spectrum("periodic", Q, RATE, OB)
        for d in [0.0, 1.0, 3.0]:
            rr = response_moments(ds, W, detuning_rad=d)
            # rho_regime1(delta, K, gamma_eff, F) with A = 2*gamma_eff - K
            expect = ds.F_coherent / np.sqrt((2 * W) ** 2 + 4 * d**2)
            assert rr.nu == pytest.approx(expect, rel=1e-12)
            assert rho_regime1(d, 0.0, W, ds.F_coherent) == pytest.approx(expect, rel=1e-12)

    def test_incoherent_part_is_detuning_independent(self):
        """
        白噪声在通带内均匀 ⟹ σ_z 与失谐无关。
        结构性预测：**非节律刺激的疗效对刺激频率不敏感，节律刺激敏感。**
        """
        ds = drive_spectrum("poisson", Q, RATE, OB)
        s0 = response_moments(ds, W, 0.0).sigma
        for d in [0.5, 2.0, 10.0]:
            assert response_moments(ds, W, d).sigma == pytest.approx(s0, rel=1e-12)

    def test_coherence_gain_formula(self):
        """增益 = sqrt(2r/W_rad) = √(2N)，且等于矩比。"""
        for r in [10.0, 12.0, 20.0]:
            for w_hz in [0.5, 1.0, 2.0]:
                w = TWO_PI * w_hz
                ob = TWO_PI * r
                rrR = response_moments(drive_spectrum("periodic", Q, r, ob), w)
                rrA = response_moments(drive_spectrum("poisson", Q, r, ob), w)
                g = coherence_gain(r, w_hz)
                assert g == pytest.approx(np.sqrt(2 * r / w), rel=1e-12)
                assert rrR.rms / rrA.rms == pytest.approx(g, rel=1e-12)

    def test_clinical_gain_is_small(self):
        """
        本节的核心断言：临床参数下增益只有 ~2×，不是"远大于"。
        若这条挂了，2.18(1) 的重写就失去依据。
        """
        assert 1.5 < coherence_gain(12.0, 1.0) < 2.5
        assert coherence_gain(12.0, 3.0) < 1.3      # 宽峰个体几乎无差别


# =========================================================================== #
# 3. 解析 vs 直接模拟（关键验证）
# =========================================================================== #
@pytest.mark.slow
class TestAgainstSimulation:
    @pytest.mark.parametrize("kind,jit", [("periodic", 0.0), ("jittered", 0.010),
                                          ("jittered", 0.030), ("poisson", 0.0)])
    def test_moments_match_simulation(self, kind, jit):
        """
        用真实脉冲串直接积分 ż = (iω̄−W)z + s(t)/2，与解析矩对照。
        不使用高斯/圆对称近似 —— 这是对 response_moments 的独立检验。
        """
        ds = drive_spectrum(kind, Q, RATE, OB, jit)
        an = response_moments(ds, W)
        _, w = simulate_response(kind, Q, RATE, W, OB, OB,
                                 duration=600.0, jitter=jit, seed=3)
        nu_sim = abs(w.mean())
        sg_sim = np.sqrt(np.mean(np.abs(w - w.mean()) ** 2) / 2)
        assert nu_sim == pytest.approx(an.nu, abs=0.1 * an.rms)
        assert sg_sim == pytest.approx(an.sigma, abs=0.1 * an.rms + 2.5e-3)

    def test_rms_ratio_matches_simulation(self):
        """最终引用的那个数（增益 ≈ 2）必须在模拟里复现。"""
        rms = {}
        for kind in ("periodic", "poisson"):
            _, w = simulate_response(kind, Q, RATE, W, OB, OB, duration=800.0, seed=5)
            rms[kind] = np.sqrt(np.mean(np.abs(w) ** 2))
        assert rms["periodic"] / rms["poisson"] == pytest.approx(
            coherence_gain(RATE, W_HZ), rel=0.10
        )

    def test_pulse_rates_are_correct(self):
        """三种波形的平均率必须一致 —— 否则比较无意义。"""
        for kind, jit in [("periodic", 0.0), ("jittered", 0.02), ("poisson", 0.0)]:
            t = pulse_times(kind, RATE, 200.0, jit, seed=0)
            assert len(t) / 200.0 == pytest.approx(RATE, rel=0.05)


# =========================================================================== #
# 4. 莱斯分布门控
# =========================================================================== #
class TestRiceGating:
    def test_deterministic_limit(self):
        """σ→0：⟨e^{−β|z|}⟩ → e^{−βν}。"""
        rr = RiceResponse(nu=0.02, sigma=0.0)
        assert mean_exp_rice(rr, 30.0) == pytest.approx(np.exp(-30.0 * 0.02), rel=1e-9)

    def test_rayleigh_closed_form(self):
        """
        ν=0（纯瑞利）有闭式：
            ⟨e^{−βR}⟩ = 1 − βσ√(π/2)·e^{β²σ²/2}·erfc(βσ/√2)
        """
        s, beta = 0.01, 40.0
        rr = RiceResponse(nu=0.0, sigma=s)
        x = beta * s
        closed = 1.0 - x * np.sqrt(np.pi / 2) * np.exp(x**2 / 2) * erfc(x / np.sqrt(2))
        assert mean_exp_rice(rr, beta) == pytest.approx(closed, rel=1e-4)

    def test_rice_mean_matches_numeric(self):
        """RiceResponse.mean_abs 的贝塞尔闭式 vs 数值积分。"""
        from scipy.special import i0e
        for nu, s in [(0.0, 0.01), (0.01, 0.01), (0.03, 0.005)]:
            rr = RiceResponse(nu, s)
            r = np.linspace(1e-12, nu + 12 * s, 40001)
            pdf = (r / s**2) * np.exp(-((r - nu) ** 2) / (2 * s**2)) * i0e(r * nu / s**2)
            num = np.trapezoid(pdf * r, r) / np.trapezoid(pdf, r)
            assert rr.mean_abs == pytest.approx(num, rel=1e-4)

    def test_rayleigh_mean_below_rms(self):
        """
        ⟨R⟩/rms = √(π/2)/√2 = 0.886。门控看 ⟨|z|⟩ 而非 rms，
        所以非节律条件下疗效比略**大于**振幅比。
        """
        rr = RiceResponse(0.0, 0.01)
        assert rr.mean_abs / rr.rms == pytest.approx(np.sqrt(np.pi / 2) / np.sqrt(2), rel=1e-4)

    def test_efficacy_ordering_and_magnitude(self):
        """
        本节要引用的结论：非节律保留节律疗效的 34–79%（W = 0.5–3 Hz）。
        """
        from scipy.optimize import brentq
        keep = []
        for w_hz in [0.5, 1.0, 2.0, 3.0]:
            w, ob = TWO_PI * w_hz, TWO_PI * RATE
            rR = response_moments(drive_spectrum("periodic", Q, RATE, ob), w)
            rA = response_moments(drive_spectrum("poisson", Q, RATE, ob), w)
            b = brentq(lambda x: efficacy_rice(rR, x) - 0.15, 1e-4, 5e4)
            keep.append(efficacy_rice(rA, b) / efficacy_rice(rR, b))
        assert all(0.30 < k < 0.85 for k in keep)
        assert all(keep[i] < keep[i + 1] for i in range(len(keep) - 1))  # W 越宽越接近

    def test_itc_gating_decoupling_P11(self):
        """
        P11：非节律条件下 ITC(∝|⟨z⟩|) 归零，但门控(∝⟨|z|⟩)不归零。
        这是把 Iverson 阴性结果从反证转成支持证据的关键。
        """
        rA = response_moments(drive_spectrum("poisson", Q, RATE, OB), W)
        assert rA.nu == pytest.approx(0.0, abs=1e-15)     # ITC = 0
        assert rA.mean_abs > 0.3 * response_moments(
            drive_spectrum("periodic", Q, RATE, OB), W).mean_abs
