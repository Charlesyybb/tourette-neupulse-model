"""
受迫 Kuramoto 模型解析核心 (core.py)
=====================================

实现第 2 章的主方程与其全部解析极限。**不含任何数值积分** —— 锁定态
由代数方程求根得到，这是 OA 精确降维的直接红利（2.6 节）。

对应关系
--------
    master_residual        式 (2.15)  主方程
    solve_rho              式 (2.15)  求根 + 稳定性判定
    rho_regime1            式 (2.16)  亚临界洛伦兹调谐曲线
    rho_free               式 (2.12)  F=0 自发同步
    rho_pure_forced        式 (2.13)  K=0, Δ=0 纯受迫
    adler_locked           式 (2.2)   M0 Arnold tongue
    rho_von_mises          式 (2.4)   零去谐含噪声锁相值
    F_eff / regime2_locked 式 (2.21)/(2.22)  超临界 Adler 约化
    susceptibility         新增：非相干态线性响应（见 README 讨论）

约定：全部量为 rad/s（见 units.py）。Δ ≡ ω̄ − Ω。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e

__all__ = [
    "Params",
    "master_residual",
    "solve_rho",
    "LockedState",
    "rho_regime1",
    "rho_free",
    "rho_pure_forced",
    "adler_locked",
    "adler_phase",
    "rho_von_mises",
    "K_critical",
    "F_eff",
    "regime2_locked",
    "A_width",
    "hwhm",
    "rho_max_regime1",
    "nondim",
    "master_residual_nondim",
    "susceptibility",
]

# ρ 求根的数值边界。ρ→1⁻ 时式 (2.15) 因 (1−ρ²)⁻² 发散，必须留出裕度。
_RHO_LO = 1.0e-12
_RHO_HI = 1.0 - 1.0e-10


# --------------------------------------------------------------------------- #
# 参数容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """
    单个个体的模型参数（全部 rad/s，β 无量纲）。

    omega_bar : mu 峰频（洛伦兹分布中心）
    gamma     : mu 谱峰半宽（洛伦兹半宽）
    K         : 内源耦合
    F         : 外部驱动强度
    D         : 相位噪声强度（默认并入 gamma_eff，见 2.11 (a)）
    """

    omega_bar: float
    gamma: float
    K: float
    F: float
    D: float = 0.0

    @property
    def gamma_eff(self) -> float:
        """式 (2.26)：γ_eff = γ + D（非相干态线性稳定性层面精确，见附录 A.6）。"""
        return self.gamma + self.D

    @property
    def A(self) -> float:
        """式 (2.18)：A ≡ 2γ_eff − K。A > 0 ⟺ 亚临界。"""
        return 2.0 * self.gamma_eff - self.K

    @property
    def is_subcritical(self) -> bool:
        return self.A > 0.0

    def delta(self, Omega):
        """Δ = ω̄ − Ω。"""
        return self.omega_bar - np.asarray(Omega, dtype=float)


# --------------------------------------------------------------------------- #
# 主方程
# --------------------------------------------------------------------------- #
def master_residual(rho, delta, K, gamma, F):
    """
    式 (2.15) 的残差：LHS − F²。

        4Δ²ρ²/(1+ρ²)²  +  ρ²[2γ − K(1−ρ²)]²/(1−ρ²)²  −  F²

    ρ→0⁺ 时 → −F² < 0；ρ→1⁻ 时（若 γ>0）→ +∞。故 (0,1) 内至少一根。
    """
    rho = np.asarray(rho, dtype=float)
    r2 = rho * rho
    term_phase = 4.0 * delta**2 * r2 / (1.0 + r2) ** 2
    term_amp = r2 * (2.0 * gamma - K * (1.0 - r2)) ** 2 / (1.0 - r2) ** 2
    return term_phase + term_amp - F * F


def _phase_from_rho(rho, delta, K, gamma, F):
    """式 (2.14)：由 ρ 唯一确定 Φ。返回 (sinΦ, cosΦ, Φ)。"""
    r2 = rho * rho
    s = 2.0 * delta * rho / (F * (1.0 + r2))
    c = rho * (2.0 * gamma - K * (1.0 - r2)) / (F * (1.0 - r2))
    return s, c, float(np.arctan2(s, c))


def _jacobian(rho, Phi, delta, K, gamma, F):
    """式 (2.11) 在不动点处的 Jacobian。用于稳定性判定（说明 2.3 的警告）。"""
    r2 = rho * rho
    sP, cP = np.sin(Phi), np.cos(Phi)
    j11 = -gamma + 0.5 * K * (1.0 - 3.0 * r2) - F * rho * cP
    j12 = -0.5 * F * (1.0 - r2) * sP
    j21 = -0.5 * F * (1.0 - 1.0 / r2) * sP
    j22 = -0.5 * F * (rho + 1.0 / rho) * cP
    return np.array([[j11, j12], [j21, j22]])


@dataclass(frozen=True)
class LockedState:
    """式 (2.15) 的一个根及其相位、稳定性。"""

    rho: float
    Phi: float
    stable: bool
    trace: float
    det: float
    residual_check: float  # sin²Φ+cos²Φ−1，应 ~0


def solve_rho(
    delta: float,
    K: float,
    gamma: float,
    F: float,
    *,
    n_scan: int = 4001,
    all_roots: bool = False,
):
    """
    求解主方程 式 (2.15)。

    策略：在 ρ∈(0,1) 网格上扫描残差符号变化，逐段 brentq。这样可以
    检出双稳区（说明 2.3：边界并非处处鞍结，存在 Hopf 与双稳）。

    Returns
    -------
    all_roots=False : LockedState，最大的**稳定**根（无稳定根时返回最大根并
                      置 stable=False）
    all_roots=True  : list[LockedState]，按 ρ 升序

    特例
    ----
    F == 0        : 退化为 式 (2.12)，直接返回 ρ₀。
    gamma == 0 且
    K < F         : ρ→1⁻ 时残差 → K²−F² < 0，(0,1) 内无根 ⟹ 完全锁定 ρ=1。
    """
    if F == 0.0:
        r0 = rho_free(K, gamma)
        return [] if (all_roots and r0 == 0.0) else (
            [LockedState(r0, 0.0, True, np.nan, np.nan, 0.0)]
            if all_roots
            else LockedState(r0, 0.0, True, np.nan, np.nan, 0.0)
        )

    grid = np.linspace(_RHO_LO, _RHO_HI, n_scan)
    res = master_residual(grid, delta, K, gamma, F)

    roots = []
    sign_change = np.where(np.sign(res[:-1]) * np.sign(res[1:]) < 0)[0]
    for i in sign_change:
        r = brentq(master_residual, grid[i], grid[i + 1], args=(delta, K, gamma, F),
                   xtol=1e-14, rtol=1e-15)
        s, c, Phi = _phase_from_rho(r, delta, K, gamma, F)
        J = _jacobian(r, Phi, delta, K, gamma, F)
        tr, det = float(np.trace(J)), float(np.linalg.det(J))
        roots.append(
            LockedState(float(r), Phi, bool(tr < 0 and det > 0), tr, det,
                        float(s * s + c * c - 1.0))
        )

    if not roots:
        # 无内点根：完全锁定（γ=0 且 K<F 的退化情形）
        roots = [LockedState(1.0, 0.0, True, np.nan, np.nan, 0.0)]

    if all_roots:
        return roots
    stable = [r for r in roots if r.stable]
    return max(stable or roots, key=lambda s: s.rho)


# --------------------------------------------------------------------------- #
# 解析极限：2.7.1 三个硬断言 + 两个区制
# --------------------------------------------------------------------------- #
def rho_free(K: float, gamma: float) -> float:
    """式 (2.12)：F=0 自发同步。K ≤ Kc = 2γ 时为 0。"""
    Kc = 2.0 * gamma
    return float(np.sqrt(1.0 - Kc / K)) if K > Kc else 0.0


def K_critical(gamma: float, D: float = 0.0) -> float:
    """式 (A.26)：Kc = 2(γ + D)。含噪声时在非相干态线性稳定性层面精确。"""
    return 2.0 * (gamma + D)


def rho_pure_forced(gamma: float, F: float) -> float:
    """式 (2.13)：K=0, Δ=0。F≪γ 线性 F/(2γ)；F≫γ 饱和 → 1。"""
    if F == 0.0:
        return 0.0
    return float((np.sqrt(gamma**2 + F**2) - gamma) / F)


def adler_locked(delta, F) -> bool:
    """式 (2.2)：M0 的 1:1 Arnold tongue 判据 |Δ| ≤ F。"""
    return bool(np.abs(delta) <= F)


def adler_phase(delta, F) -> float:
    """M0 稳定不动点 φ* = arcsin(Δ/F) ∈ (−π/2, π/2)。未锁定返回 nan。"""
    return float(np.arcsin(delta / F)) if adler_locked(delta, F) else float("nan")


def rho_von_mises(F: float, D: float) -> float:
    """式 (2.4)：Δ=0 含噪声单振子，ρ = I₁(F/D)/I₀(F/D)。用 exp-scaled Bessel 防溢出。"""
    if D <= 0.0:
        return 1.0
    kappa = F / D
    return float(i1e(kappa) / i0e(kappa))


def A_width(gamma_eff: float, K: float) -> float:
    """A ≡ 2γ_eff − K。"""
    return 2.0 * gamma_eff - K


def hwhm(gamma_eff: float, K: float) -> float:
    """式 (2.18)：功率半高半宽 W = A/2 = γ_eff − K/2。"""
    return A_width(gamma_eff, K) / 2.0


def rho_max_regime1(gamma_eff: float, K: float, F: float) -> float:
    """式 (2.18)：ρ_max = F/A。"""
    return F / A_width(gamma_eff, K)


def rho_regime1(delta, K: float, gamma_eff: float, F: float):
    """
    式 (2.16)：亚临界弱驱动洛伦兹调谐曲线

        ρ(Δ) = F / sqrt(A² + 4Δ²)

    适用域（式 A.17）：F ≪ A，且 K < Kc。K→Kc 时 A→0，必须回到 solve_rho。
    """
    A = A_width(gamma_eff, K)
    return F / np.sqrt(A**2 + 4.0 * np.asarray(delta, dtype=float) ** 2)


def F_eff(F: float, rho0: float) -> float:
    """式 (2.21)：超临界有效驱动 F_eff = (F/2)(ρ₀ + 1/ρ₀) ≥ F（AM–GM）。"""
    if rho0 <= 0.0:
        return float("inf")
    return 0.5 * F * (rho0 + 1.0 / rho0)


def regime2_locked(delta, F: float, K: float, gamma: float) -> bool:
    """式 (2.22)：超临界真阈值 |Δ| ≤ F_eff。"""
    return bool(np.abs(delta) <= F_eff(F, rho_free(K, gamma)))


# --------------------------------------------------------------------------- #
# 无量纲形式（2.10）
# --------------------------------------------------------------------------- #
def nondim(K: float, F: float, delta, gamma_eff: float):
    """式 (2.23)：(κ, f, δ) = (K/2γ_eff, F/2γ_eff, Δ/γ_eff)。相变在 κ=1。"""
    g2 = 2.0 * gamma_eff
    return K / g2, F / g2, np.asarray(delta, dtype=float) / gamma_eff


def master_residual_nondim(rho, delta_nd, kappa, f):
    """式 (2.24)：无量纲主方程残差。"""
    rho = np.asarray(rho, dtype=float)
    r2 = rho * rho
    t1 = delta_nd**2 * r2 / (1.0 + r2) ** 2
    t2 = r2 * (1.0 - kappa * (1.0 - r2)) ** 2 / (1.0 - r2) ** 2
    return t1 + t2 - f * f


# --------------------------------------------------------------------------- #
# 新增：非相干态线性响应（用于解决 说明 2.5 的张力）
# --------------------------------------------------------------------------- #
def susceptibility(delta, K: float, gamma_eff: float):
    """
    式 (2.10) 在 w=0 处线性化后的复响应函数：

        ẇ = (iΔ − γ_eff)w + (K/2)w + F/2
        ⟹ w* = (F/2) / (A/2 − iΔ),   A = 2γ_eff − K

    故 |w*| = F / sqrt(A² + 4Δ²) —— **与 式 (2.16) 逐字相同**。

    含义（正文 说明 2.5 未展开的一条）：亚临界时 ρ₀ = 0 与"静息 mu 确实存在"
    并不矛盾。静息谱峰不是序参量，而是这条**易感率共振**被内噪声激发的表现，
    其半高半宽同样是 A/2。若成立，则静息 EEG mu 峰宽可直接读出治疗调谐曲线
    的宽度，无需任何拟合。

    该等价性在线性层面是解析恒等；"静息谱宽 = 驱动调谐宽"还额外需要涨落–耗散
    型论证，非严格，必须用 N 振子 SDE 数值检验（见 roadmap M2）。

    Returns
    -------
    complex ndarray : χ(Δ) = 1 / (A/2 − iΔ)，使得 w* = (F/2)·χ
    """
    A = A_width(gamma_eff, K)
    return 1.0 / (0.5 * A - 1j * np.asarray(delta, dtype=float))
