"""
解析金标准硬断言 (tests/test_analytic.py)
==========================================

第 2 章 2.7.1 要求"代码必须硬断言"的三个极限，以及正文与附录 A 中每一条
可独立复核的代数/数值结论。每个测试的 docstring 标明其对应的公式编号。

约定：全部 rad/s。凡以 Hz 表述的正文数值（说明 2.4、表 A.1）在测试中显式转换。

运行：  pytest -q tests/test_analytic.py
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import i0, i1

from fkm.core import (
    A_width,
    F_eff,
    K_critical,
    LockedState,
    Params,
    adler_locked,
    adler_phase,
    hwhm,
    master_residual,
    master_residual_nondim,
    nondim,
    rho_free,
    rho_max_regime1,
    rho_pure_forced,
    rho_regime1,
    rho_von_mises,
    solve_rho,
    susceptibility,
)
from fkm.observation import (
    beta_from_rice,
    dlog_transfer,
    efficacy_regime1,
    jitter_factor,
    jitter_sigma_for_attenuation,
    lowpass_transfer,
    optimal_shift,
    optimal_shift_exact,
    second_order_ratio,
    tic_rate,
    tic_rate_exact,
)
from fkm.units import TWO_PI, hz_to_rad, rad_to_hz

TOL = 1e-9


# =========================================================================== #
# 2.7.1  三个极限检验 —— 正文明确要求的硬断言
# =========================================================================== #
class TestThreeLimits:
    """2.7.1：F=0 / (K=0,Δ=0) / (γ→0,K→0)。"""

    @pytest.mark.parametrize("gamma,K", [(1.0, 3.0), (0.5, 4.0), (2.0, 4.5)])
    def test_limit_i_free_running(self, gamma, K):
        """式 (2.12)：F=0 且 K>Kc ⟹ ρ₀ = sqrt(1 − 2γ/K)，Kc = 2γ。"""
        assert K > 2.0 * gamma
        assert rho_free(K, gamma) == pytest.approx(np.sqrt(1.0 - 2.0 * gamma / K), abs=TOL)
        assert K_critical(gamma) == pytest.approx(2.0 * gamma, abs=TOL)

    @pytest.mark.parametrize("gamma,K", [(1.0, 1.5), (2.0, 3.9)])
    def test_limit_i_subcritical_is_zero(self, gamma, K):
        """K < Kc = 2γ ⟹ 平均场 ρ₀ = 0（说明 2.5 的张力来源）。"""
        assert K < 2.0 * gamma
        assert rho_free(K, gamma) == 0.0

    @pytest.mark.parametrize("gamma,F", [(1.0, 0.1), (1.0, 1.0), (0.3, 5.0), (2.0, 0.01)])
    def test_limit_ii_pure_forcing(self, gamma, F):
        """
        式 (2.13)：K=0, Δ=0 ⟹ ρ* = (sqrt(γ²+F²) − γ)/F。
        同时验证它确实是主方程 式 (2.15) 的根。
        """
        rho = rho_pure_forced(gamma, F)
        assert master_residual(rho, 0.0, 0.0, gamma, F) == pytest.approx(0.0, abs=1e-12)
        assert solve_rho(0.0, 0.0, gamma, F).rho == pytest.approx(rho, abs=1e-10)

    def test_limit_ii_asymptotics(self):
        """式 (2.13) 两端：F≪γ ⟹ F/(2γ)（线性）；F≫γ ⟹ 1（饱和）。"""
        gamma = 1.0
        F_small = 1e-4
        assert rho_pure_forced(gamma, F_small) == pytest.approx(F_small / (2 * gamma), rel=1e-6)
        assert rho_pure_forced(gamma, 1e5) == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("delta,F", [(0.0, 1.0), (0.5, 1.0), (0.99, 1.0), (-0.7, 2.0)])
    def test_limit_iii_reduces_to_adler(self, delta, F):
        """
        (iii) γ→0, K→0 应退回 M0 的 Adler 方程 式 (2.1)–(2.2)。
        锁定时 ρ→1 且 Φ → φ* = arcsin(Δ/F)。
        """
        eps = 1e-9
        st = solve_rho(delta, eps, eps, F)
        assert adler_locked(delta, F)
        assert st.rho == pytest.approx(1.0, abs=1e-4)
        assert np.sin(st.Phi) == pytest.approx(np.sin(adler_phase(delta, F)), abs=1e-4)

    def test_limit_iii_unlocked_outside_tongue(self):
        """式 (2.2)：|Δ| > F 时无 1:1 锁定。"""
        assert not adler_locked(1.5, 1.0)
        assert np.isnan(adler_phase(1.5, 1.0))


# =========================================================================== #
# 2.8  主方程自洽性
# =========================================================================== #
class TestMasterEquation:
    @pytest.mark.parametrize(
        "delta,K,gamma,F",
        [(0.0, 1.0, 1.0, 0.3), (0.7, 0.5, 1.2, 0.4), (-1.3, 2.0, 1.5, 0.9),
         (2.5, 3.0, 1.0, 1.1), (0.2, 0.0, 0.8, 0.05)],
    )
    def test_root_satisfies_equation_and_phase_consistency(self, delta, K, gamma, F):
        """求得的 ρ 必须使 式 (2.15) 归零，且 式 (2.14) 满足 sin²Φ+cos²Φ=1。"""
        st = solve_rho(delta, K, gamma, F)
        assert master_residual(st.rho, delta, K, gamma, F) == pytest.approx(0.0, abs=1e-10)
        assert st.residual_check == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < st.rho <= 1.0

    def test_residual_sign_structure(self):
        """附录 A.3 存在性论证：ρ→0⁺ 残差 <0，ρ→1⁻ 残差 →+∞（γ>0）。"""
        d, K, g, F = 0.5, 1.0, 1.0, 0.4
        assert master_residual(1e-10, d, K, g, F) < 0
        assert master_residual(1 - 1e-8, d, K, g, F) > 1e10

    def test_rho_is_even_in_delta(self):
        """式 (2.27)：Δ 只以 Δ² 出现 ⟹ ρ*(Δ) 为偶函数 ⟹ Ω* = ω̄。"""
        K, g, F = 1.0, 1.0, 0.3
        for d in [0.1, 0.5, 1.3, 3.0]:
            assert solve_rho(d, K, g, F).rho == pytest.approx(
                solve_rho(-d, K, g, F).rho, abs=1e-11
            )

    def test_argmax_is_omega_bar(self):
        """Ω* = ω̄：在 Ω 网格上直接搜索，峰位必须落在 ω̄。"""
        p = Params(omega_bar=hz_to_rad(11.0), gamma=hz_to_rad(1.0), K=hz_to_rad(0.6),
                   F=hz_to_rad(0.15))
        grid = hz_to_rad(np.linspace(6.0, 16.0, 2001))
        rho = np.array([solve_rho(p.omega_bar - O, p.K, p.gamma, p.F).rho for O in grid])
        assert rad_to_hz(grid[int(np.argmax(rho))]) == pytest.approx(11.0, abs=0.01)


# =========================================================================== #
# 2.9.1  区制 I：洛伦兹调谐曲线与守恒关系
# =========================================================================== #
class TestRegimeI:
    def test_linearisation_matches_master_equation(self):
        """式 (2.16) 必须在其适用域 F ≪ A（式 A.17）内逼近 式 (2.15)。"""
        gamma, K = 1.0, 0.5
        A = A_width(gamma, K)
        F = 0.01 * A  # 深入适用域
        for d in [0.0, 0.3, 1.0, 2.0]:
            exact = solve_rho(d, K, gamma, F).rho
            approx = rho_regime1(d, K, gamma, F)
            assert approx == pytest.approx(exact, rel=2e-3)

    def test_linearisation_fails_outside_domain(self):
        """适用域声明的反向验证：F ~ A 时 式 (2.16) 明显失准。"""
        gamma, K = 1.0, 0.5
        A = A_width(gamma, K)
        F = 1.2 * A
        exact = solve_rho(0.0, K, gamma, F).rho
        approx = rho_regime1(0.0, K, gamma, F)
        assert abs(approx - exact) / exact > 0.1

    def test_hwhm(self):
        """式 (2.18)：功率 ρ² 的半高半宽 W = A/2 = γ − K/2。"""
        gamma, K, F = 1.5, 1.0, 0.05
        W = hwhm(gamma, K)
        assert W == pytest.approx(gamma - K / 2.0, abs=TOL)
        p0 = rho_regime1(0.0, K, gamma, F) ** 2
        pW = rho_regime1(W, K, gamma, F) ** 2
        assert pW == pytest.approx(p0 / 2.0, rel=1e-12)
        # W 同时是 ρ 的 1/sqrt(2) 半宽
        assert rho_regime1(W, K, gamma, F) == pytest.approx(
            rho_regime1(0.0, K, gamma, F) / np.sqrt(2.0), rel=1e-12
        )

    @pytest.mark.parametrize("gamma,K,F", [(1.0, 0.5, 0.02), (2.0, 1.7, 0.05), (0.8, 0.1, 0.01)])
    def test_conservation_law(self, gamma, K, F):
        """式 (2.20)：W · ρ_max = F/2，与 γ、K 的具体取值无关。"""
        assert hwhm(gamma, K) * rho_max_regime1(gamma, K, F) == pytest.approx(F / 2.0, rel=1e-12)

    def test_normalised_tuning_curve(self):
        """式 (2.19)：ρ(Δ)/ρ(0) = 1/sqrt(1 + (Δ/W)²)。"""
        gamma, K, F = 1.2, 0.4, 0.03
        W = hwhm(gamma, K)
        for d in [0.0, 0.5 * W, W, 2 * W, 5 * W]:
            lhs = rho_regime1(d, K, gamma, F) / rho_regime1(0.0, K, gamma, F)
            assert lhs == pytest.approx(1.0 / np.sqrt(1.0 + (d / W) ** 2), rel=1e-12)

    def test_note_2_4_quantitative_example(self):
        """说明 2.4：ω̄=11 Hz、W=0.5 Hz、固定 10 Hz ⟹ ρ/ρ_max = 1/sqrt(5) = 0.447。"""
        W_hz, delta_hz = 0.5, 1.0
        assert 1.0 / np.sqrt(1.0 + (delta_hz / W_hz) ** 2) == pytest.approx(1 / np.sqrt(5), rel=1e-12)
        # 同一数字在 rad/s 下必须不变（比值无量纲）
        W, d = hz_to_rad(W_hz), hz_to_rad(delta_hz)
        assert 1.0 / np.sqrt(1.0 + (d / W) ** 2) == pytest.approx(1 / np.sqrt(5), rel=1e-12)


# =========================================================================== #
# 2.9.2  区制 II：Adler 楔形
# =========================================================================== #
class TestRegimeII:
    def test_F_eff_amgm_bound(self):
        """A.5 注 1：由 AM–GM，F_eff = (F/2)(ρ₀+1/ρ₀) ≥ F，等号在 ρ₀=1。"""
        F = 0.5
        for rho0 in [0.1, 0.3, 0.6, 0.9, 1.0]:
            assert F_eff(F, rho0) >= F - 1e-15
        assert F_eff(F, 1.0) == pytest.approx(F, abs=TOL)

    def test_F_eff_diverges_at_critical(self):
        """A.5 注 2：ρ₀→0⁺ 时 F_eff→∞ 是伪发散（ρ≈ρ₀ 近似失效的信号）。"""
        assert F_eff(0.5, 1e-6) > 1e5
        assert np.isinf(F_eff(0.5, 0.0))

    def test_regime2_wedge_boundary(self):
        """式 (2.22)：锁定条件是硬阈值，边界斜率由 F_eff 给出。"""
        gamma, K, F = 1.0, 4.0, 0.2
        rho0 = rho_free(K, gamma)
        assert rho0 == pytest.approx(np.sqrt(1 - 0.5), abs=TOL)
        thr = F_eff(F, rho0)
        from fkm.core import regime2_locked
        assert regime2_locked(0.99 * thr, F, K, gamma)
        assert not regime2_locked(1.01 * thr, F, K, gamma)


# =========================================================================== #
# 2.10  无量纲化
# =========================================================================== #
class TestNondimensionalisation:
    @pytest.mark.parametrize(
        "delta,K,gamma,F", [(0.0, 1.0, 1.0, 0.3), (1.1, 0.4, 1.3, 0.2), (-2.0, 2.5, 1.4, 0.8)]
    )
    def test_dimensional_and_nondimensional_agree(self, delta, K, gamma, F):
        """式 (2.24) 必须与 式 (2.15) 给出相同的根。"""
        kappa, f, d_nd = nondim(K, F, delta, gamma)
        rho = solve_rho(delta, K, gamma, F).rho
        assert master_residual_nondim(rho, d_nd, kappa, f) == pytest.approx(0.0, abs=1e-10)

    def test_transition_at_kappa_one(self):
        """同步相变位于 κ = 1（即 K = 2γ_eff = Kc）。"""
        gamma = 1.3
        kappa, _, _ = nondim(K_critical(gamma), 0.0, 0.0, gamma)
        assert kappa == pytest.approx(1.0, abs=TOL)

    def test_regime1_nondim_form(self):
        """式 (2.25)：ρ = f/sqrt((1−κ)² + δ²)，W/γ_eff = 1 − κ。"""
        gamma, K, F, delta = 1.0, 0.6, 0.01, 0.45
        kappa, f, d_nd = nondim(K, F, delta, gamma)
        assert rho_regime1(delta, K, gamma, F) == pytest.approx(
            f / np.sqrt((1 - kappa) ** 2 + d_nd**2), rel=1e-12
        )
        assert hwhm(gamma, K) / gamma == pytest.approx(1 - kappa, abs=TOL)


# =========================================================================== #
# 2.11 / A.6  噪声
# =========================================================================== #
class TestNoise:
    def test_von_mises_locking(self):
        """式 (2.4)：Δ=0 时 ρ = I₁(F/D)/I₀(F/D)。"""
        for F, D in [(0.5, 1.0), (2.0, 1.0), (0.1, 3.0)]:
            assert rho_von_mises(F, D) == pytest.approx(i1(F / D) / i0(F / D), rel=1e-12)

    def test_von_mises_linear_response(self):
        """式 (2.4) 弱驱动极限 ρ → F/(2D)：与 式 (2.13) 的 F/(2γ) 同结构（说明 2.1）。"""
        F, D = 1e-5, 1.0
        assert rho_von_mises(F, D) == pytest.approx(F / (2 * D), rel=1e-6)

    def test_gamma_eff_heuristic_consistency(self):
        """
        说明 2.1 的核心论据：噪声 D 与异质性 γ 在线性响应阶占据同一位置。
        F/(2D)（von Mises）与 F/(2γ)（式 2.13 弱驱动）系数完全一致。
        """
        F = 1e-6
        assert rho_von_mises(F, 1.0) == pytest.approx(rho_pure_forced(1.0, F), rel=1e-4)

    def test_critical_coupling_integral(self):
        """
        式 (A.22)–(A.26)：∫ g(ω)dω/(D+i(ω−ω̄)) = 1/(D+γ) 对洛伦兹精确，
        故 Kc = 2(γ+D)。此处直接数值积分核对。
        """
        gamma, D = 0.7, 0.4
        g = lambda u: (gamma / np.pi) / (u**2 + gamma**2)
        re, _ = quad(lambda u: g(u) * D / (D**2 + u**2), -np.inf, np.inf, limit=400)
        im, _ = quad(lambda u: g(u) * (-u) / (D**2 + u**2), -np.inf, np.inf, limit=400)
        assert re == pytest.approx(1.0 / (D + gamma), rel=1e-8)
        assert im == pytest.approx(0.0, abs=1e-10)   # 对称 g ⟹ 虚部为零
        assert K_critical(gamma, D) == pytest.approx(2.0 / re, rel=1e-8)

    def test_gamma_eff_property(self):
        """式 (2.26)：Params.gamma_eff = γ + D。"""
        p = Params(omega_bar=1.0, gamma=0.7, K=0.2, F=0.05, D=0.4)
        assert p.gamma_eff == pytest.approx(1.1, abs=TOL)
        assert p.A == pytest.approx(2 * 1.1 - 0.2, abs=TOL)


# =========================================================================== #
# 2.12 / A.7  最优频率偏移
# =========================================================================== #
class TestOptimalFrequency:
    def test_dlog_transfer_value(self):
        """式 (A.32)：一阶低通 fc=25 Hz，在 10 Hz 处 Ξ'/Ξ = −0.0138 Hz⁻¹。"""
        assert dlog_transfer(10.0, 25.0) == pytest.approx(-0.01379, rel=1e-3)

    @pytest.mark.parametrize("W_hz,expected", [(0.5, 0.00345), (1.0, 0.0138), (2.0, 0.0552)])
    def test_table_a1(self, W_hz, expected):
        """
        表 A.1：偏移量在 10⁻³–10⁻¹ Hz 量级。

        ERRATUM：表 A.1 第一行印作 "0.003"，精确值为 0.00345。其余两行
        （0.014 / 0.055）为两位有效数字，第一行只有一位，前后不一致。
        建议正文改为 0.0034。不影响任何结论。
        """
        A_hz = 2.0 * W_hz
        u = optimal_shift(A_hz, 10.0, 25.0)
        assert abs(u) == pytest.approx(expected, rel=1e-2)

    def test_optimal_shift_unit_invariance(self):
        """
        **单位陷阱守门测试**：式 (2.28) 在 Hz 与 rad/s 两套单位下计算，
        结果必须满足 u_rad = 2π·u_Hz。混用单位会静默给出错误答案。
        """
        W_hz, f_hz, fc_hz = 1.0, 10.0, 25.0
        u_hz = optimal_shift(2 * W_hz, f_hz, fc_hz)
        u_rad = optimal_shift(hz_to_rad(2 * W_hz), hz_to_rad(f_hz), hz_to_rad(fc_hz))
        assert u_rad == pytest.approx(TWO_PI * u_hz, rel=1e-12)

    def test_shift_below_eeg_precision(self):
        """说明 2.7：偏移 ≪ EEG 峰频估计精度 (±0.2–0.5 Hz) ⟹ 可直接取 Ω* = ω̂̄。"""
        for W_hz in [0.5, 1.0, 2.0]:
            assert abs(optimal_shift(2 * W_hz, 10.0, 25.0)) < 0.1

    def test_linearisation_vs_exact(self):
        """式 (2.28) 是小偏移线性化；与 式 (A.30) 的数值精确解对照。"""
        for W_hz in [0.5, 1.0, 2.0]:
            A = 2 * W_hz
            lin = optimal_shift(A, 10.0, 25.0)
            exact = optimal_shift_exact(A, 10.0, 25.0)
            assert lin == pytest.approx(exact, abs=5e-3)

    def test_shift_sensitivity_to_fc(self):
        """
        敏感性：fc 是全文唯一一个凭空设定的数。检验结论对它的稳健性。
        结论"偏移可忽略"在 fc ≳ 15 Hz、W ≤ 2 Hz 时成立；fc 更低时接近 EEG 精度。
        """
        assert abs(optimal_shift(4.0, 10.0, 25.0)) < 0.1     # fc=25, W=2 -> 0.055
        assert abs(optimal_shift(4.0, 10.0, 15.0)) < 0.2     # fc=15 -> ~0.12
        assert abs(optimal_shift(4.0, 10.0, 12.0)) > 0.1     # fc=12 -> ~0.16, 已不可忽略


# =========================================================================== #
# A.8  外周抖动
# =========================================================================== #
class TestJitter:
    def test_jitter_at_10hz(self):
        """式 (A.34)：σ_τ=3 ms、Ω=2π·10 ⟹ 衰减约 1.8%。"""
        assert jitter_factor(3e-3, hz_to_rad(10.0)) == pytest.approx(0.9824, rel=1e-3)

    def test_jitter_sigma_for_20pct(self):
        """式 (A.35)：产生 20% 衰减需 σ_τ ≈ 10.6 ms —— 外周通路不现实。"""
        assert jitter_sigma_for_attenuation(0.8, hz_to_rad(10.0)) == pytest.approx(10.6e-3, rel=1e-2)

    def test_jitter_matters_in_gamma_band(self):
        """A.8 结论：gamma 波段（Ω 大 4–5 倍）时指数中因子大 16–25 倍，不可忽略。"""
        assert jitter_factor(3e-3, hz_to_rad(40.0)) < 0.80


# =========================================================================== #
# 2.14 / A.9  观测模型
# =========================================================================== #
class TestObservationModel:
    def test_rice_expansion_recovers_beta(self):
        """式 (A.37)–(A.38)：一阶截断给出 β = aΘ₀/σ_m²。"""
        nu0, Theta0, a, sm = 5.0, 2.0, 0.5, 1.0
        beta = beta_from_rice(a, Theta0, sm)
        lam0 = nu0 * np.exp(-(Theta0**2) / (2 * sm**2))
        rho = 1e-6                                   # 深入一阶区
        assert tic_rate_exact(rho, nu0, Theta0, a, sm) == pytest.approx(
            tic_rate(rho, lam0, beta), rel=1e-8
        )

    def test_second_order_makes_it_steeper(self):
        """说明 2.9：被丢弃的 ρ² 项使真实依赖**略陡于**指数。"""
        nu0, Theta0, a, sm = 5.0, 2.0, 0.5, 1.0
        beta = beta_from_rice(a, Theta0, sm)
        lam0 = nu0 * np.exp(-(Theta0**2) / (2 * sm**2))
        rho = 0.5
        assert tic_rate_exact(rho, nu0, Theta0, a, sm) < tic_rate(rho, lam0, beta)

    def test_second_order_ratio_formula(self):
        """式 (A.39)：二阶/一阶 = aρ/(2Θ₀)。"""
        assert second_order_ratio(0.4, 0.5, 2.0) == pytest.approx(0.4 * 0.5 / 4.0, abs=TOL)

    def test_beta_scales_inversely_with_noise(self):
        """β ∝ 1/σ_m²：TS 的高噪声同时压低 ρ（经 D）与 β（经 σ_m），双路径削弱疗效。"""
        assert beta_from_rice(1, 1, 2.0) == pytest.approx(beta_from_rice(1, 1, 1.0) / 4.0, abs=TOL)

    def test_efficacy_peaks_at_omega_bar(self):
        """式 (2.34)：E_i(Ω) 的峰位是 ω̄_i。"""
        ob = hz_to_rad(11.2)
        grid = hz_to_rad(np.linspace(7, 15, 4001))
        E = efficacy_regime1(grid, ob, A=hz_to_rad(1.5), B=hz_to_rad(0.4))
        assert rad_to_hz(grid[int(np.argmax(E))]) == pytest.approx(11.2, abs=0.01)

    def test_efficacy_monotone_in_B_and_decreasing_in_A(self):
        """B=βF 越大疗效越高；A 越大（宽而平）天花板越低。"""
        ob, O = hz_to_rad(10.0), hz_to_rad(10.0)
        assert efficacy_regime1(O, ob, 1.0, 0.5) < efficacy_regime1(O, ob, 1.0, 1.0)
        assert efficacy_regime1(O, ob, 2.0, 0.5) < efficacy_regime1(O, ob, 1.0, 0.5)


# =========================================================================== #
# 2.16 / A.10  可辨识性
# =========================================================================== #
class TestIdentifiability:
    def test_gamma_K_null_direction(self):
        """
        式 (A.42)：沿 (δγ_eff, δK) ∝ (1, 2) 移动不改变 A，故不改变任何观测量。
        Fisher 信息矩阵在该方向有严格零特征值。
        """
        g, K, F = 1.0, 0.5, 0.02
        eps = 1e-6
        A0 = A_width(g, K)
        A1 = A_width(g + eps, K + 2 * eps)
        assert A1 == pytest.approx(A0, abs=1e-15)
        for d in [0.0, 0.4, 1.1]:
            assert rho_regime1(d, K, g, F) == pytest.approx(
                rho_regime1(d, K + 2 * eps, g + eps, F), rel=1e-12
            )

    def test_beta_F_null_direction(self):
        """式 (A.43)：δlnβ = −δlnF 保持 B = βF 不变。"""
        from fkm.observation import efficacy
        from fkm.core import rho_regime1 as rr
        g, K, ob = 1.0, 0.5, hz_to_rad(10.0)
        F1, b1 = 0.02, 3.0
        s = 1.7
        F2, b2 = F1 * s, b1 / s
        d = 0.3
        assert efficacy(rr(d, K, g, F1), 0.0, b1) == pytest.approx(
            efficacy(rr(d, K, g, F2), 0.0, b2), rel=1e-12
        )

    def test_effective_dimension_is_three(self):
        """
        式 (2.37)：区制 I 下 9 维参数空间的有效维数为 3。
        数值检验：对 (ω̄, γ, K, F, β) 的 Jacobian 秩应为 3。
        """
        from fkm.observation import efficacy_regime1 as E

        def model(theta, Omegas):
            ob, g, K, F, b = theta
            A, B = 2 * g - K, b * F
            return E(Omegas, ob, A, B)

        theta = np.array([hz_to_rad(10.5), 1.0, 0.5, 0.02, 3.0])
        Om = hz_to_rad(np.linspace(8, 13, 40))
        J = np.zeros((len(Om), 5))
        for j in range(5):
            h = 1e-7 * max(abs(theta[j]), 1.0)
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h
            tm[j] -= h
            J[:, j] = (model(tp, Om) - model(tm, Om)) / (2 * h)
        sv = np.linalg.svd(J, compute_uv=False)
        assert np.sum(sv > sv[0] * 1e-8) == 3


# =========================================================================== #
# 新增：非相干态易感率 —— 对 说明 2.5 张力的一个解析回应
# =========================================================================== #
class TestSusceptibility:
    def test_susceptibility_equals_regime1_tuning(self):
        """
        式 (2.10) 在 w=0 线性化 ⟹ |w*| = (F/2)|χ| = F/sqrt(A²+4Δ²)
        —— 与 式 (2.16) 逐字相同。

        含义：亚临界时 ρ₀=0 与"静息 mu 存在"不矛盾 —— 静息谱峰是这条易感率
        共振被内噪声激发的结果，其半高半宽同样是 A/2。若数值 SDE 检验通过，
        则静息 EEG mu 峰宽可直接读出治疗调谐曲线宽度（零拟合参数）。
        """
        g, K, F = 1.2, 0.4, 0.03
        for d in [0.0, 0.5, 1.7, 4.0]:
            assert 0.5 * F * abs(susceptibility(d, K, g)) == pytest.approx(
                rho_regime1(d, K, g, F), rel=1e-12
            )

    def test_susceptibility_hwhm(self):
        """|χ|² 的半高半宽同为 A/2 = W。"""
        g, K = 1.2, 0.4
        W = hwhm(g, K)
        p0 = abs(susceptibility(0.0, K, g)) ** 2
        assert abs(susceptibility(W, K, g)) ** 2 == pytest.approx(p0 / 2, rel=1e-12)

    def test_susceptibility_diverges_at_criticality(self):
        """κ→1 (A→0) 时易感率发散 —— 线性化失效，对应 说明 2.6 的临界邻域。"""
        g = 1.0
        assert abs(susceptibility(0.0, 2 * g - 1e-8, g)) > 1e7
