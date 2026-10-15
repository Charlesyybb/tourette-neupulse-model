"""
刺激波形与相干增益 (waveform.py)
=================================

回答 2.18 (1)：模型预测节律刺激比非节律强多少？

正文把 Iverson et al. [2023] 的"节律 ≈ 非节律"列为模型无法解释的最强反证。
这个让步过大 —— 非节律脉冲串**不是宽带白噪声**，它是有相同平均率、但相位
不相干的驱动，在 ω̄ 附近仍有可观功率。模型给出的是一个**可计算的比值**，
而不是"无法解释"。

核心结构
--------
区制 I 下（式 2.16 的适用域，正是临床阈下刺激所处的区制），系统对驱动是
**线性滤波器**：把 式 (2.9) 在 z = 0 处线性化，

    ż = (iω̄ − W)z + s(t)/2,      W = A/2 = γ_eff − K/2

即中心 ω̄、半宽 W 的单极点洛伦兹滤波器。相干时间 τ_c = 1/W。

于是节律 vs 非节律的差别归结为一句话：

    **相干时间窗内的脉冲，是同相叠加（∝ N）还是随机叠加（∝ √N）。**

    增益 = N/√N = √N,   N = r/W  （相干窗内的脉冲数）

而临床参数下 N 很小：r = 12 Hz、W ≈ 1 Hz ⟹ W_rad = 6.28 rad/s ⟹ N ≈ 1.9。
**√N ≈ 1.4。模型根本没有预测"节律 ≫ 非节律"。**

统一的参数化
------------
抖动脉冲串 t_n = nT + ε_n，ε_n ~ N(0, σ_τ²)，其功率谱是标准结果：

    S(ω) = q²r[1 − |Φ(ω)|²]  +  q²r²|Φ(ω)|² Σ_k δ(ω − kΩ),   Φ(ω) = e^{−σ_τ²ω²/2}

σ_τ = 0 给周期串（全部功率在谱线），σ_τ → ∞ 给泊松串（全部功率在连续谱）。
**附录 A.8 的 Ξ_jitter 与本节的非节律问题是同一个计算的两端** —— 正文把它们
当作互不相关的两件事（A.8 是"可忽略的负面结果"，2.18(1) 是"最强反证"），
其实是一条曲线上的两个点。

响应分布
--------
相干分量给确定振幅 ν，不相干分量给圆对称复高斯涨落 σ_z，故 |z| 服从
**莱斯分布** Rice(ν, σ_z)。σ_τ=0 退化为确定值，σ_τ→∞ 退化为瑞利分布。

门控（式 2.32）看到的是**瞬时** |z(t)|，不是 ⟨z⟩。这一点很重要：非节律刺激
的 ⟨z⟩ = 0（ITC 为零），但 ⟨|z|⟩ > 0，仍然抑制抽动。**ITC 与疗效在非节律
条件下解耦** —— 这本身是一条可判伪预测。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import i0e

__all__ = [
    "DriveSpectrum",
    "RiceResponse",
    "pulse_times",
    "drive_spectrum",
    "response_moments",
    "coherence_gain",
    "mean_exp_rice",
    "efficacy_rice",
    "simulate_response",
]

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# 1. 脉冲串
# --------------------------------------------------------------------------- #
def pulse_times(kind: str, rate: float, duration: float, jitter: float = 0.0,
                seed: int | None = None) -> np.ndarray:
    """
    生成脉冲到达时刻 (秒)。

    kind : "periodic"  严格周期
           "jittered"  周期 + 高斯抖动 σ_τ = jitter
           "poisson"   齐次泊松（平均率相同，间隔完全随机）
    """
    rng = np.random.default_rng(seed)
    if kind == "periodic":
        return np.arange(0.0, duration, 1.0 / rate)
    if kind == "jittered":
        t = np.arange(0.0, duration, 1.0 / rate)
        return np.sort(t + rng.normal(0.0, jitter, t.size))
    if kind == "poisson":
        n = rng.poisson(rate * duration * 1.2)
        return np.sort(rng.uniform(0.0, duration, n))[: int(rate * duration) + 50]
    raise ValueError(f"未知 kind: {kind}")


# --------------------------------------------------------------------------- #
# 2. 驱动谱：相干分量 + 不相干分量
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DriveSpectrum:
    """在频率 Ω_rad 处的驱动分解。q 为单脉冲相位冲量（无量纲）。"""

    F_coherent: float      # 相干驱动振幅 (rad/s)，即式 (2.5) 里的 F
    S_incoherent: float    # 连续谱功率密度 (rad²/s²)/(rad/s)
    phi: float             # |Φ(Ω)| = e^{−σ²Ω²/2}，相干保留比例


def drive_spectrum(kind: str, q: float, rate: float, Omega_rad: float,
                   jitter: float = 0.0) -> DriveSpectrum:
    """
    抖动脉冲串在 Ω 处的相干/不相干分解。

    periodic  : Φ=1  ⟹ 全相干,  F = q·r,      S_inc = 0
    poisson   : Φ=0  ⟹ 全不相干, F = 0,        S_inc = q²r
    jittered  : 介于两者之间，由 Φ = e^{−σ²Ω²/2}（即 式 A.33）单参数插值
    """
    if kind == "periodic":
        phi = 1.0
    elif kind == "poisson":
        phi = 0.0
    elif kind == "jittered":
        phi = float(np.exp(-(jitter**2) * Omega_rad**2 / 2.0))
    else:
        raise ValueError(f"未知 kind: {kind}")
    return DriveSpectrum(F_coherent=q * rate * phi,
                         S_incoherent=q**2 * rate * (1.0 - phi**2),
                         phi=phi)


# --------------------------------------------------------------------------- #
# 3. 线性响应：|z| 的莱斯分布参数
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiceResponse:
    """|z| ~ Rice(nu, sigma)。nu 为相干分量，sigma 为每正交分量的涨落标准差。"""

    nu: float
    sigma: float

    @property
    def mean_abs(self) -> float:
        """⟨|z|⟩ 的解析式（莱斯分布均值，用 exp-scaled Bessel 防溢出）。"""
        s, n = self.sigma, self.nu
        if s <= 0:
            return float(n)
        x = n**2 / (4.0 * s**2)
        # L_{1/2}(−x) = e^{−x}[(1+2x)I₀(x) + 2x I₁(x)]，用 i0e/i1e 展开
        from scipy.special import i1e
        L = (1.0 + 2.0 * x) * i0e(x) + 2.0 * x * i1e(x)
        return float(s * np.sqrt(np.pi / 2.0) * L)

    @property
    def rms(self) -> float:
        """sqrt(⟨|z|²⟩) = sqrt(ν² + 2σ²)。"""
        return float(np.sqrt(self.nu**2 + 2.0 * self.sigma**2))


def response_moments(ds: DriveSpectrum, W_rad: float, detuning_rad: float = 0.0
                     ) -> RiceResponse:
    """
    把驱动谱经洛伦兹滤波器映射为 |z| 的莱斯参数。

    相干：   ν = (F/2)/sqrt(W² + Δ²) = F/sqrt(A² + 4Δ²)   —— 即 式 (2.16)
    不相干： ⟨|z|²⟩ = ∫(dω/2π)·S·(1/4)/(W² + (ω−ω̄)²) = S/(8W)
             圆对称复高斯 ⟹ ⟨|z|²⟩ = 2σ² ⟹ σ² = S/(16W)

    注意不相干项**与失谐无关**（白噪声在滤波器通带内均匀），这是一条结构性
    差别：非节律刺激的效应对刺激频率不敏感，节律刺激敏感。
    """
    nu = 0.5 * ds.F_coherent / np.sqrt(W_rad**2 + detuning_rad**2)
    sigma = np.sqrt(ds.S_incoherent / (16.0 * W_rad))
    return RiceResponse(float(nu), float(sigma))


def coherence_gain(rate: float, W_hz: float) -> float:
    """
    节律 / 非节律的 RMS 振幅比（同 q、同平均率、零失谐）。

        ν_periodic / rms_poisson = sqrt(2r / W_rad),   W_rad = 2πW_hz

    等价形式 sqrt(2N)，N = r/W_rad 为相干窗内脉冲数 —— 即 √N 定律。
    """
    return float(np.sqrt(2.0 * rate / (TWO_PI * W_hz)))


# --------------------------------------------------------------------------- #
# 4. 门控平均：⟨e^{−β|z|}⟩
# --------------------------------------------------------------------------- #
def mean_exp_rice(rr: RiceResponse, beta: float, n: int = 20001) -> float:
    """
    ⟨e^{−β|z|}⟩，|z| ~ Rice(ν, σ)。数值积分（无初等闭式）。

    σ → 0 退化为 e^{−βν}；ν → 0 退化为瑞利情形的 erfc 闭式。
    """
    nu, s = rr.nu, rr.sigma
    if s <= 1e-15:
        return float(np.exp(-beta * nu))
    hi = nu + 12.0 * s
    r = np.linspace(1e-12, hi, n)
    # Rice pdf，用 i0e 承接大宗量
    pdf = (r / s**2) * np.exp(-((r - nu) ** 2) / (2 * s**2)) * i0e(r * nu / s**2)
    return float(np.trapezoid(pdf * np.exp(-beta * r), r) / np.trapezoid(pdf, r))


def efficacy_rice(rr: RiceResponse, beta: float, rr_base: RiceResponse | None = None
                  ) -> float:
    """式 (2.33) 的推广：E = 1 − ⟨λ⟩_stim/⟨λ⟩_base，门控看瞬时 |z|。"""
    num = mean_exp_rice(rr, beta)
    den = 1.0 if rr_base is None else mean_exp_rice(rr_base, beta)
    return float(1.0 - num / den)


# --------------------------------------------------------------------------- #
# 5. 直接模拟（用于验证上面全部解析式）
# --------------------------------------------------------------------------- #
def simulate_response(kind, q, rate, W_rad, omega_bar_rad, Omega_rad,
                      duration=400.0, dt=1e-4, jitter=0.0, seed=0, burn=5.0):
    """
    直接积分线性响应 ż = (i ω̄ − W)z + s(t)/2，s(t) 为真实脉冲串。

    **不使用**任何高斯/圆对称近似 —— 这是对 response_moments 的独立检验。
    返回稳态段的 z(t)。
    """
    t = np.arange(0.0, duration, dt)
    times = pulse_times(kind, rate, duration, jitter, seed)
    s = np.zeros(t.size)
    idx = np.clip((times / dt).astype(int), 0, t.size - 1)
    np.add.at(s, idx, q / dt)                       # δ 函数的离散化

    # 精确指数积分器（脉冲之间解析传播，避免刚性）
    lam = (1j * omega_bar_rad - W_rad) * dt
    decay = np.exp(lam)
    z = np.zeros(t.size, dtype=complex)
    drive = 0.5 * s * dt
    for k in range(1, t.size):
        z[k] = z[k - 1] * decay + drive[k]
    # 转到旋转系：相干分量应变为常数
    w = z * np.exp(-1j * Omega_rad * t)
    keep = t >= burn
    return t[keep], w[keep]
