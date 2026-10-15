"""
观测模型 (observation.py)
=========================

把序参量 ρ 映射到三条可测链路（表 2.3）：ITC、诱发 mu 功率、抽动率 TPM。

    tic_rate            式 (2.30)/(2.32)  Rice 越阈 -> 指数门控
    tic_rate_exact      式 (2.30)+(2.31)  未截断的完整形式
    beta_from_rice      β = aΘ₀/σ_m²
    efficacy            式 (2.33)
    efficacy_regime1    式 (2.34)  个体调谐曲线闭式解
    lowpass_transfer    Ξ(Ω) 一阶低通
    optimal_shift       式 (2.28)  最优频率偏移
    jitter_factor       式 (A.33)  外周抖动低通因子
    adaptation          式 (2.36)  刺激期内 F 的衰减
"""

from __future__ import annotations

import numpy as np

from .core import A_width, rho_regime1

__all__ = [
    "tic_rate",
    "tic_rate_exact",
    "beta_from_rice",
    "second_order_ratio",
    "efficacy",
    "efficacy_regime1",
    "lowpass_transfer",
    "dlog_transfer",
    "optimal_shift",
    "jitter_factor",
    "jitter_sigma_for_attenuation",
    "adaptation",
]


# --------------------------------------------------------------------------- #
# 门控：Rice 越阈公式 (2.14.1)
# --------------------------------------------------------------------------- #
def beta_from_rice(a: float, Theta0: float, sigma_m: float) -> float:
    """式 (2.32)：β = aΘ₀/σ_m²。注意 β ∝ 1/σ_m²（噪声越大门控越弱）。"""
    return a * Theta0 / sigma_m**2


def tic_rate(rho, lambda0: float, beta: float):
    """式 (2.32)：一阶截断的唯象门控 λ = λ₀ e^{−βρ}。"""
    return lambda0 * np.exp(-beta * np.asarray(rho, dtype=float))


def tic_rate_exact(rho, nu0: float, Theta0: float, a: float, sigma_m: float):
    """
    式 (2.30)+(2.31) 未截断形式：

        λ = ν₀ exp[ −(Θ₀ + aρ)² / (2σ_m²) ]

    与 tic_rate 的差别是被丢弃的 ρ² 项 —— 即"真实依赖略陡于指数"（说明 2.9）。
    """
    rho = np.asarray(rho, dtype=float)
    return nu0 * np.exp(-((Theta0 + a * rho) ** 2) / (2.0 * sigma_m**2))


def second_order_ratio(rho, a: float, Theta0: float):
    """式 (A.39)：二阶项/一阶项 = aρ/(2Θ₀)。一阶截断成立要求此比值 ≪ 1。"""
    return a * np.asarray(rho, dtype=float) / (2.0 * Theta0)


# --------------------------------------------------------------------------- #
# 疗效函数 (2.14.2)
# --------------------------------------------------------------------------- #
def efficacy(rho_stim, rho_base: float, beta: float):
    """式 (2.33)：E = 1 − exp[−β(ρ* − ρ₀)]。"""
    d = np.asarray(rho_stim, dtype=float) - rho_base
    return 1.0 - np.exp(-beta * d)


def efficacy_regime1(Omega, omega_bar: float, A: float, B: float):
    """
    式 (2.34)：区制 I 的个体调谐曲线闭式解

        E_i(Ω) = 1 − exp[ −B_i / sqrt(A_i² + 4(ω̄_i − Ω)²) ],   B ≡ βF

    只依赖三个可辨识参数 Θ_i = {ω̄_i, A_i, B_i}（式 2.37）。
    argmax_Ω E = ω̄（Δ 只以 Δ² 出现，式 2.27）。
    """
    delta = omega_bar - np.asarray(Omega, dtype=float)
    return 1.0 - np.exp(-B / np.sqrt(A**2 + 4.0 * delta**2))


# --------------------------------------------------------------------------- #
# 传输函数与最优频率偏移 (2.12 / A.7)
# --------------------------------------------------------------------------- #
def lowpass_transfer(f, fc: float):
    """一阶低通 Ξ(f) = [1 + (f/fc)²]^{-1/2}。f 与 fc 同单位。"""
    x = np.asarray(f, dtype=float) / fc
    return 1.0 / np.sqrt(1.0 + x * x)


def dlog_transfer(f, fc: float):
    """d lnΞ/df = −(f/fc²)/[1+(f/fc)²]。单位为 1/[f]。"""
    f = np.asarray(f, dtype=float)
    return -(f / fc**2) / (1.0 + (f / fc) ** 2)


def optimal_shift(A: float, f_eval: float, fc: float):
    """
    式 (2.28)：u = Ω* − ω̄ ≈ (A²/4)·(Ξ'/Ξ)|_ω̄

    **单位敏感**：A、f_eval、fc 必须同单位。结果与该单位一致。
    单位无关性由 tests/test_analytic.py::test_optimal_shift_unit_invariance 断言。
    """
    return (A**2 / 4.0) * dlog_transfer(f_eval, fc)


def optimal_shift_exact(A: float, omega_bar: float, fc: float,
                        span: float = 3.0, n: int = 200001):
    """
    式 (A.30) 的数值精确解（不做小偏移线性化），用于界定 式 (2.28) 的适用域。
    在 [ω̄−span, ω̄+span] 上直接最大化 ρ = Ξ(Ω)/sqrt(A²+4(ω̄−Ω)²)。
    """
    grid = np.linspace(omega_bar - span, omega_bar + span, n)
    val = lowpass_transfer(grid, fc) / np.sqrt(A**2 + 4.0 * (omega_bar - grid) ** 2)
    return float(grid[int(np.argmax(val))] - omega_bar)


# --------------------------------------------------------------------------- #
# 外周抖动 (A.8)
# --------------------------------------------------------------------------- #
def jitter_factor(sigma_tau: float, Omega_rad: float) -> float:
    """式 (A.33)：Ξ_jitter = exp(−σ_τ²Ω²/2)。Omega_rad 必须是 rad/s。"""
    return float(np.exp(-(sigma_tau**2) * Omega_rad**2 / 2.0))


def jitter_sigma_for_attenuation(target: float, Omega_rad: float) -> float:
    """反解：产生给定衰减因子 target 所需的 σ_τ（秒）。"""
    return float(np.sqrt(-2.0 * np.log(target)) / Omega_rad)


# --------------------------------------------------------------------------- #
# 刺激期内适应 (2.15.2)
# --------------------------------------------------------------------------- #
def adaptation(t, F0: float, a_inf: float, Ta: float):
    """式 (2.36)：F(t) = F₀[a∞ + (1−a∞)e^{−t/Ta}]。TS 组 Ta 更短。"""
    t = np.asarray(t, dtype=float)
    return F0 * (a_inf + (1.0 - a_inf) * np.exp(-t / Ta))


# --------------------------------------------------------------------------- #
# 便捷组合：给定 Params 直接给出调谐曲线（区制 I）
# --------------------------------------------------------------------------- #
def tuning_curve_regime1(Omega, params, beta: float):
    """由 Params + β 生成 E(Ω)。仅在区制 I 适用域内有效。"""
    A = A_width(params.gamma_eff, params.K)
    rho = rho_regime1(params.delta(Omega), params.K, params.gamma_eff, params.F)
    return efficacy(rho, 0.0, beta)
