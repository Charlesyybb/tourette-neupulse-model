"""
M2 数值层的硬断言。

分三组：
  A. 圆累积量方程的自洽性 —— 纯 ODE，毫秒级，无标记
  B. 估计器的无偏性       —— 合成过程，秒级，无标记
  C. 与 N 振子 SDE 对照   —— @pytest.mark.slow

C 组是唯一能证伪"代码正确实现了第 2 章"以外命题的部分：它拿掉了 OA
假设，直接积分 式 (2.5)。A、B 两组的存在是为了保证 C 组失败时能定位到
底是物理还是管线。
"""
from __future__ import annotations

import numpy as np
import pytest

from fkm.core import rho_regime1, solve_rho
from fkm.numerics import (acf, acf_decay_rate, cumulant_rhs, cumulant_steady,
                          driven_hwhm, lorentzian_frequencies, ou_reference,
                          simulate, simulate_batch)

TP = 2 * np.pi


# ========================================================================== #
# A. 圆累积量方程 (★) 的自洽性
# ========================================================================== #
class TestCumulantEquations:

    def test_D0_kappa2_is_invariant_manifold(self):
        """
        D = 0 ⟹ κ₂ = 0 是不变流形。

        由 (★)：κ̇₂ = [2(iΔ−γ)−4D]κ₂ − 2Dκ₁² − H*κ₁κ₂。D=0 且 κ₂=0 时
        每一项都含 κ₂ 或 D ⟹ κ̇₂ ≡ 0。这是 OA 精确性的方程层面表述。
        """
        for K, F, d in [(0.5, 1.0, 0.0), (1.2, 0.3, 0.7), (0.0, 2.0, -1.1)]:
            y = [0.4, 0.2, 0.0, 0.0]
            dk2 = cumulant_rhs(0.0, y, d, 1.0, K, F, 0.0, 0.0)[2:]
            assert np.allclose(dk2, 0.0, atol=1e-14)

    def test_D0_reduces_to_OA_equation(self):
        """D=0、κ₂=0 时 κ̇₁ 必须逐位等于 式 (2.9) 的 OA 右端。"""
        rng = np.random.default_rng(0)
        for _ in range(20):
            k1 = complex(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8))
            K, F, g = rng.uniform(0, 2), rng.uniform(0, 3), rng.uniform(0.2, 2)
            d = rng.uniform(-2, 2)
            got = cumulant_rhs(0.0, [k1.real, k1.imag, 0.0, 0.0],
                               d, g, K, F, 0.0, 0.0)
            H = K * k1 + F
            want = (1j * d - g) * k1 + 0.5 * (H - np.conj(H) * k1 ** 2)
            assert got[0] == pytest.approx(want.real, rel=1e-12, abs=1e-14)
            assert got[1] == pytest.approx(want.imag, rel=1e-12, abs=1e-14)

    def test_free_decay_rate_is_gamma_plus_D(self):
        """
        F = K = 0 ⟹ κ₁ 以 γ + D 衰减。

        这是 γ_eff = γ + D 在**线性阶精确**的方程层证据（附录 A.6 的
        Kc = 2(γ+D) 是它的另一面）。
        """
        for g, D in [(1.0, 0.0), (0.7, 0.3), (0.5, 1.5)]:
            k1 = complex(0.3, -0.1)
            got = cumulant_rhs(0.0, [k1.real, k1.imag, 0.0, 0.0],
                               0.0, g, 0.0, 0.0, D, 0.0)
            want = -(g + D) * k1
            assert complex(got[0], got[1]) == pytest.approx(want, rel=1e-12)

    def test_kappa2_driven_by_rho_squared(self):
        """
        κ₂ 的源项是 −2Dκ₁² ⟹ OA 偏差随 ρ² 增长。

        这解释了"OA 在弱驱动下好、锁定态下差"，也是 2.11 该保留的理由。
        """
        prev = 0.0
        for F in [0.3, 1.0, 3.0]:
            k1, k2 = cumulant_steady(gamma=1.0, K=0.5, F=F, D=0.5)
            assert abs(k2) > prev
            prev = abs(k2)

    def test_kappa_ratio_matches_analytic_estimate(self):
        """稳态 |κ₂|/|κ₁|² ≈ D/(γ+2D)（文档中由 (★) 读出的量级估计）。"""
        g = 1.0
        for D in [0.2, 0.5, 1.0]:
            k1, k2 = cumulant_steady(gamma=g, K=0.5, F=0.3, D=D)
            got = abs(k2) / abs(k1) ** 2
            assert got == pytest.approx(D / (g + 2 * D), rel=0.05)

    def test_D0_cumulant_equals_analytic_master_equation(self):
        """
        D = 0 时 累积量稳态 |κ₁| 必须等于 core.py 解 式 (2.15) 的 ρ*。

        这条把 M2 的 ODE 求解器与 M1 的代数求根互锁 —— 任何一边改错都会红。
        """
        for K, F in [(0.5, 0.3), (0.5, 3.0), (1.5, 1.0)]:
            k1, k2 = cumulant_steady(gamma=1.0, K=K, F=F, D=0.0)
            assert abs(k2) == pytest.approx(0.0, abs=1e-9)
            assert abs(k1) == pytest.approx(solve_rho(0.0, K, 1.0, F).rho, rel=1e-6)


# ========================================================================== #
# B. 估计器
# ========================================================================== #
class TestEstimators:

    def test_acf_uses_unbiased_normalisation(self):
        """
        Ĉ(k) 必须除以 M−k 而非 M。

        有偏归一化引入 (1−k/M) 的额外衰减，会直接加到速率上 —— 这是
        自相关法最容易踩的坑。用常值序列检验：C(k) 应恒等于 |x|²。
        """
        x = np.full(1000, 1.0 + 0.0j)
        c = acf(x, 100)
        assert np.allclose(c.real, 1.0, atol=1e-9)

    def test_acf_decay_rate_unbiased_on_known_OU(self):
        """
        已知速率的复 OU：估计器必须无偏。

        M2 的第一轮跳过了这一步，把 Welch 谱拟合的 −3~−8% 估计器偏差
        当成了物理效应。这条测试的存在就是为了不再犯。
        """
        for rate in [0.4, 1.0]:
            r = ou_reference(rate, T=6000.0, dt=0.05, seed=3)
            est, se = acf_decay_rate(r, burn_frac=0.0, n_chunk=8)
            assert est == pytest.approx(rate, rel=0.06)
            assert se / rate < 0.06

    def test_acf_decay_rate_reports_finite_error_bar(self):
        r = ou_reference(0.5, T=3000.0, dt=0.05, seed=1)
        est, se = acf_decay_rate(r, burn_frac=0.0, n_chunk=6)
        assert np.isfinite(est) and np.isfinite(se) and se > 0

    def test_driven_hwhm_recovers_regime1_width(self):
        """
        对解析的区制 I 调谐曲线，driven_hwhm 必须原样还原 W = A/2。

        无噪声的自洽性检验：拟合器本身不能引入宽度偏差。
        """
        geff, F = 1.0, 0.05
        for K in [0.0, 0.5, 1.0, 1.5]:
            A2 = geff - K / 2
            Om = TP * 10.0 + np.linspace(-3, 3, 21) * A2
            rho = rho_regime1(TP * 10.0 - Om, K, geff, F)
            assert driven_hwhm(Om, rho, TP * 10.0) == pytest.approx(A2, rel=1e-6)

    def test_ou_reference_has_requested_stationary_variance(self):
        r = ou_reference(0.8, T=4000.0, dt=0.05, seed=7, sigma=1.0)
        assert np.mean(np.abs(r.w) ** 2) == pytest.approx(1.0, rel=0.1)


# ========================================================================== #
# C. N 振子 SDE
# ========================================================================== #
class TestFrequencySampling:

    def test_lorentzian_quantiles_are_symmetric(self):
        om = lorentzian_frequencies(1000, omega_bar=5.0, gamma=1.0)
        assert np.median(om) == pytest.approx(5.0, abs=1e-9)
        assert om[0] + om[-1] == pytest.approx(10.0, abs=1e-9)

    def test_empirical_cf_approaches_exponential(self):
        """
        (1/n)Σ e^{i(ω_i−ω̄)τ} → e^{−γτ} 只在 n→∞ 成立。

        这条是 M2 第二个坑的成文记录：K=0 的自发衰减率**不是**任意 N
        精确等于 γ_eff，因为经验特征函数带 O(1/n) 的求积误差。所以 K=0
        只能当"逐 n 的管线标定"，不能当绝对真值。
        """
        tau = 1.0
        errs = []
        for n in [50, 200, 800]:
            om = lorentzian_frequencies(n, 0.0, 1.0)
            errs.append(abs(np.mean(np.exp(1j * om * tau)) - np.exp(-1.0)))
        assert errs[-1] < errs[0]          # 随 n 收敛
        assert errs[0] > 1e-4              # 但小 n 下确实非零


@pytest.mark.slow
class TestAgainstSDE:

    def test_D0_three_layers_agree(self):
        """
        D = 0 时 OA 解析 / 圆累积量 / N 振子 SDE 三者必须吻合。

        整个数值栈的互验：SDE 积分器、ODE 求解、代数求根若有任一处写错，
        三者不可能同时对上。
        """
        Fs = np.array([0.3, 1.0, 3.0])
        res = simulate_batch({"K": 0.5, "F": Fs, "D": 0.0}, n=3000, gamma=1.0,
                             T=60.0, dt=5e-4, seed=7)
        for F, r in zip(Fs, res):
            sde = r.rho_locked(0.5)
            oa = solve_rho(0.0, 0.5, 1.0, float(F)).rho
            cum = abs(cumulant_steady(gamma=1.0, K=0.5, F=float(F), D=0.0)[0])
            assert oa == pytest.approx(sde, rel=0.02)
            assert cum == pytest.approx(sde, rel=0.02)

    def test_gamma_eff_underestimates_but_is_bounded(self):
        """
        D ≠ 0 时 OA+γ_eff 单向低估 ρ，且幅度有界。

        这是 core.py 得以保留 γ_eff = γ+D 的**量化依据**（正文 2.11）。
        若哪天误差跑出这个界，说明 core.py 的适用域声明要改。
        回归界取 −12%，比实测的 −6.4% 留一倍余量。
        """
        Fs = np.array([1.0, 3.0])
        for D in [0.5, 1.0]:
            res = simulate_batch({"K": 0.5, "F": Fs, "D": D}, n=3000, gamma=1.0,
                                 T=60.0, dt=5e-4, seed=7)
            for F, r in zip(Fs, res):
                err = solve_rho(0.0, 0.5, 1.0 + D, float(F)).rho / r.rho_locked(0.5) - 1
                assert -0.12 < err < 0.005, f"D={D} F={F} 误差 {err:+.1%} 越界"

    def test_cumulant_beats_OA_when_noisy(self):
        """二阶截断必须比 γ_eff 近似更准 —— 否则 (b) 层没有存在意义。"""
        D, F = 1.0, 3.0
        r = simulate(n=3000, gamma=1.0, K=0.5, F=F, D=D, T=60.0, dt=5e-4, seed=7)
        sde = r.rho_locked(0.5)
        oa = solve_rho(0.0, 0.5, 1.0 + D, F).rho
        cum = abs(cumulant_steady(gamma=1.0, K=0.5, F=F, D=D)[0])
        assert abs(cum / sde - 1) < abs(oa / sde - 1)

    def test_susceptibility_identity_driven_side(self):
        """
        受迫调谐曲线的半高半宽 = A/2 = γ_eff − K/2。

        承重：`individualize.py` 用 A = 2W 把静息谱宽当作驱动调谐宽。
        """
        gam, D, ob, F = 0.7, 0.3, TP * 10.0, 0.05
        geff = gam + D
        for K in [0.0, 1.0]:
            A2 = geff - K / 2
            Om = ob + np.array([-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0]) * A2 * 1.6
            res = simulate_batch({"K": K, "F": F, "D": D, "Omega": Om}, n=2000,
                                 omega_bar=ob, gamma=gam, T=300.0, dt=1e-2,
                                 seed=23, record_every=5)
            W = driven_hwhm(Om, np.array([r.rho_locked(0.35) for r in res]), ob)
            assert W == pytest.approx(A2, rel=0.12)

    def test_spontaneous_linewidth_matches_A_over_2(self):
        """
        自发涨落的线宽 = A/2（易感率恒等式的自发侧）。

        短跑版，容差放宽到 20%：完整的 1/N 外推见
        `scripts/m2_validation.py spont`（T = 20000 s，不适合放进测试）。
        """
        gam, D, ob = 0.7, 0.3, TP * 10.0
        geff = gam + D
        res = simulate_batch({"K": np.array([0.0, 1.0]), "D": D}, n=400,
                             omega_bar=ob, gamma=gam, T=4000.0, dt=1e-2,
                             seed=17, record_every=5)
        for K, r in zip([0.0, 1.0], res):
            est, se = acf_decay_rate(r, burn_frac=0.05, n_chunk=8)
            assert est == pytest.approx(geff - K / 2, rel=0.20)
