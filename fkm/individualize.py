"""
个体化最优频率算法 (individualize.py)
======================================

理论给定的答案很短：Ω*_i = ω̄_i（式 2.27），低通修正 < 0.1 Hz 可忽略
（说明 2.7）。**算法的实际内容不在求峰位，而在估计误差的处理。**

核心命题（本模块的理论依据，正文未收录）
-----------------------------------------
设 ω̄ 的后验为对称单峰分布，调谐曲线 g(Δ) 对称单峰。则

    argmax_Ω  E_{ω̄|ω̂}[ g(ω̄ − Ω) ]  =  后验均值,

**与疗效变换无关** —— 因为两个对称单峰函数的卷积仍对称单峰，峰位在中心之和。
式 (2.33) 的 exp 变换是 ρ 的单调函数，不改变 argmax。

推论：正确的算法是**向人群均值收缩**，而不是"个体化 or 固定"的二选一：

    Ω*_i = λ_i·ω̂_i + (1−λ_i)·μ_pop,     λ_i = σ_pop² / (σ_pop² + s_i²)

    s_i → 0   ⟹ λ→1，完全个体化
    s_i → ∞   ⟹ λ→0，退化为固定频率 —— 但固定在 μ_pop，不一定是 10 Hz

收缩解**恒不劣于**两个极端策略中的任何一个。因此本模块不提供"是否个体化"
的决策开关；它提供的是收缩系数，以及个体化相对**最优固定频率**的增量收益
（这才是决定试验是否值得做、需要多大样本的量）。

一条顺带的模型检验
------------------
式 (2.17) 预测静息 mu 谱峰是**洛伦兹型**（功率 ∝ 1/(A²+4Δ²)），而 specparam
默认拟合高斯。二者在同一段数据上的拟合优度对比是一次零成本的模型检验：
`estimate_mu_peak(..., shape="both")` 会同时给出两者的 R²。

方法学硬约束（说明 2.11，非可选）
---------------------------------
ω̄ 必须在剥离 1/f 非周期成分后提取。Adelhöfer et al. (2021) 已证明 TS 在
感觉运动整合中非周期活动升高；直接取 FFT 峰会引入**与组别相关**的系统偏差，
而该偏差恰好污染本模型最核心的参数。本模块的 `fit_aperiodic` 强制执行这一步。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

__all__ = [
    "PeakEstimate",
    "Prior",
    "Prescription",
    "fit_aperiodic",
    "estimate_mu_peak",
    "shrink",
    "expected_gain",
    "h_factor",
    "breakeven_detuning",
    "quantise",
    "prescribe",
]

MU_BAND = (7.0, 14.0)          # 感觉运动 mu 搜索窗
_EPS = 1e-12


# =========================================================================== #
# 1. 谱估计：1/f 剥离 + 峰拟合
# =========================================================================== #
def _aperiodic_loglog(log_f, offset, exponent):
    """无 knee 的非周期成分：log10 P = b − χ·log10 f。"""
    return offset - exponent * log_f


def fit_aperiodic(freqs, psd, exclude=MU_BAND, fit_range=(2.0, 40.0), n_iter=3):
    """
    迭代拟合 1/f 背景（specparam 风格）。先在排除 mu 带的频点上拟合，
    再用残差的下半分位重拟合，抑制残余峰的拉扯。

    Returns
    -------
    (offset, exponent), 以及在全 freqs 上求值的背景 log10-功率
    """
    freqs = np.asarray(freqs, float)
    psd = np.asarray(psd, float)
    lf, lp = np.log10(freqs + _EPS), np.log10(psd + _EPS)

    in_range = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    mask = in_range & ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
    if mask.sum() < 5:
        raise ValueError("可用于 1/f 拟合的频点不足（<5）")

    p = np.polyfit(lf[mask], lp[mask], 1)
    offset, exponent = float(p[1]), float(-p[0])
    for _ in range(n_iter - 1):
        resid = lp - _aperiodic_loglog(lf, offset, exponent)
        keep = in_range & (resid <= np.quantile(resid[in_range], 0.5))
        if keep.sum() < 5:
            break
        p = np.polyfit(lf[keep], lp[keep], 1)
        offset, exponent = float(p[1]), float(-p[0])

    return (offset, exponent), _aperiodic_loglog(lf, offset, exponent)


def _lorentzian(f, amp, centre, hwhm):
    """式 (2.17) 预测的形状：功率 ∝ 1/(1 + ((f−c)/W)²)。"""
    return amp / (1.0 + ((f - centre) / hwhm) ** 2)


def _gaussian(f, amp, centre, sigma):
    """specparam 默认形状，作为对照。"""
    return amp * np.exp(-((f - centre) ** 2) / (2.0 * sigma**2))


@dataclass
class PeakEstimate:
    """静息 mu 峰的估计结果及其质量标记。"""

    f_peak: float                  # ω̂̄ / 2π  (Hz)
    se_f_peak: float               # 峰位标准误 s (Hz)
    hwhm: float                    # W (Hz)；洛伦兹拟合时直接是 A/2
    se_hwhm: float
    amplitude: float               # 去除 1/f 后的峰高
    snr: float                     # 峰高 / 残差 RMS
    r2: float
    shape: str
    r2_alt: float = float("nan")   # 另一形状的 R²，用于模型检验
    ok: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def A(self) -> float:
        """A = 2W（Hz）。经易感率恒等式，静息谱峰宽直接给出调谐曲线宽度。"""
        return 2.0 * self.hwhm


def estimate_mu_peak(
    freqs,
    psd,
    band=MU_BAND,
    shape: str = "lorentzian",
    min_snr: float = 2.0,
    max_se: float = 1.0,
    edge_margin: float = 0.3,
) -> PeakEstimate:
    """
    从静息 PSD 提取 mu 峰频与半宽。

    Parameters
    ----------
    freqs, psd : 静息谱（建议 C3/CP3，做过 Laplacian 或 CSD 参考）
    shape      : "lorentzian"（模型预测）| "gaussian"（specparam 默认）| "both"
    min_snr    : 峰高/残差 RMS 低于此值判为无可用峰
    max_se     : 峰位标准误上限 (Hz)
    edge_margin: 峰位距搜索窗边界小于此值时告警（多半是 alpha 或 beta 的肩）

    Notes
    -----
    本函数**不能**区分感觉运动 mu 与枕区 alpha —— 二者频段重叠。必须在输入端
    解决（Laplacian 参考 + 运动 ERD 定位），否则 ω̄ 会系统性偏向 alpha。
    这一点在 ok/reasons 里无法检出，属于已知盲区。
    """
    freqs = np.asarray(freqs, float)
    psd = np.asarray(psd, float)
    reasons: list[str] = []

    (_, _), bg_log = fit_aperiodic(freqs, psd, exclude=band)
    periodic = psd - 10.0**bg_log

    sel = (freqs >= band[0]) & (freqs <= band[1])
    if sel.sum() < 6:
        return PeakEstimate(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, shape, ok=False, reasons=["mu 带内频点不足"])
    fs, ps = freqs[sel], periodic[sel]

    def _fit(fn, w0):
        p0 = [max(ps.max(), _EPS), fs[int(np.argmax(ps))], w0]
        bounds = ([0.0, band[0], 0.05], [np.inf, band[1], (band[1] - band[0]) / 2])
        popt, pcov = curve_fit(fn, fs, ps, p0=p0, bounds=bounds, maxfev=20000)
        resid = ps - fn(fs, *popt)
        ss = 1.0 - np.sum(resid**2) / max(np.sum((ps - ps.mean()) ** 2), _EPS)
        se = np.sqrt(np.diag(pcov))
        return popt, se, float(ss), float(np.sqrt(np.mean(resid**2)))

    try:
        if shape in ("lorentzian", "both"):
            popt, se, r2, rms = _fit(_lorentzian, 1.0)
            hw = popt[2]
            r2_alt = _fit(_gaussian, 1.0)[2] if shape == "both" else float("nan")
            used = "lorentzian"
        else:
            popt, se, r2, rms = _fit(_gaussian, 1.0)
            hw = popt[2] * np.sqrt(2.0 * np.log(2.0))   # 高斯 σ → HWHM
            se[2] *= np.sqrt(2.0 * np.log(2.0))
            r2_alt = float("nan")
            used = "gaussian"
    except (RuntimeError, ValueError) as exc:
        return PeakEstimate(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, shape, ok=False, reasons=[f"拟合失败: {exc}"])

    amp, centre = float(popt[0]), float(popt[1])
    snr = amp / max(rms, _EPS)

    if snr < min_snr:
        reasons.append(f"信噪比过低 (SNR={snr:.1f} < {min_snr})")
    if not np.isfinite(se[1]) or se[1] > max_se:
        reasons.append(f"峰位标准误过大 (s={se[1]:.2f} Hz > {max_se})")
    if centre - band[0] < edge_margin or band[1] - centre < edge_margin:
        reasons.append(f"峰位贴近搜索窗边界 ({centre:.2f} Hz)，疑似 alpha/beta 肩")

    return PeakEstimate(
        f_peak=centre, se_f_peak=float(se[1]), hwhm=float(hw), se_hwhm=float(se[2]),
        amplitude=amp, snr=float(snr), r2=r2, shape=used, r2_alt=r2_alt,
        ok=len(reasons) == 0, reasons=reasons,
    )


# =========================================================================== #
# 2. 收缩与贝叶斯最优频率
# =========================================================================== #
@dataclass(frozen=True)
class Prior:
    """
    人群先验 ω̄ ~ N(mu_pop, sd_pop²)，单位 Hz。

    默认值来自表 2.4：成人感觉运动 mu 去同步峰 10–12 Hz，个体标准差约 1 Hz。
    **这是全流程中最需要用自有数据替换的量** —— μ_pop 若真高于 10 Hz，
    则固定 10 Hz 本身就系统性失谐，其影响大于个体差异（见 headline_numbers.py）。
    """

    mu_pop: float = 10.5
    sd_pop: float = 1.0


def shrink(f_hat: float, se: float, prior: Prior) -> tuple[float, float, float]:
    """
    共轭正态后验。

    Returns
    -------
    (后验均值 = 贝叶斯最优刺激频率, 后验标准差 τ, 收缩系数 λ)

    λ = σ_pop²/(σ_pop² + s²)。τ = sqrt(λ)·s = sσ/sqrt(s²+σ²) ≤ min(s, σ)。
    """
    if not np.isfinite(se) or se <= 0:
        return float(f_hat), 0.0, 1.0
    v_pop = prior.sd_pop**2
    lam = v_pop / (v_pop + se**2)
    mean = lam * f_hat + (1.0 - lam) * prior.mu_pop
    tau = np.sqrt(lam) * se
    return float(mean), float(tau), float(lam)


def h_factor(u: float, n: int = 40001) -> float:
    """
    h(u) ≡ E_{z~N(0,1)}[ 1/sqrt(1 + u²z²) ]，u = 频率不确定度 / 调谐宽度 W。

    这是"给定不确定度 u，期望能拿到最大带动量的多少比例"。h(0)=1，h→0 as u→∞。
    """
    if u <= 0:
        return 1.0
    z = np.linspace(-8.0, 8.0, n)
    w = np.exp(-0.5 * z**2)
    return float(np.trapezoid(w / np.sqrt(1.0 + (u * z) ** 2), z) / np.trapezoid(w, z))


def expected_gain(Omega: float, post_mean: float, tau: float, W: float,
                  n: int = 4001) -> float:
    """
    E_{ω̄|ω̂}[ ρ(ω̄−Ω)/ρ_max ]，即在后验下选定 Ω 的期望相对带动量。
    式 (2.19)：ρ/ρ_max = 1/sqrt(1+(Δ/W)²)。
    """
    if tau <= 0:
        return 1.0 / np.sqrt(1.0 + ((post_mean - Omega) / W) ** 2)
    x = np.linspace(post_mean - 8 * tau, post_mean + 8 * tau, n)
    w = np.exp(-0.5 * ((x - post_mean) / tau) ** 2)
    g = 1.0 / np.sqrt(1.0 + ((x - Omega) / W) ** 2)
    return float(np.trapezoid(w * g, x) / np.trapezoid(w, x))


def breakeven_detuning(u: float) -> float:
    """
    盲设固定频率的"等效失谐"：d*/W = sqrt(1/h(u)² − 1)。

    读法：以不确定度 u = s/W 做个体化，其期望收益等于把固定频率精确设在
    真峰位偏离 d* 处。若个体的 |ω̄ − Ω_fix| 小于 d*，个体化反而不划算。
    """
    h = h_factor(u)
    return float(np.sqrt(max(1.0 / h**2 - 1.0, 0.0)))


def quantise(Omega: float, step: float | None) -> float:
    """设备频率步进量化。step=None 表示连续可调。"""
    return float(Omega) if not step else float(np.round(Omega / step) * step)


# =========================================================================== #
# 3. 端到端处方
# =========================================================================== #
@dataclass
class Prescription:
    omega_star: float              # 建议刺激频率 (Hz)，已量化
    omega_bayes: float             # 量化前的贝叶斯最优频率
    f_hat: float                   # 原始 EEG 峰频估计
    se: float                      # 峰位标准误 s (Hz)
    tau: float                     # 后验标准差 (Hz)
    lam: float                     # 收缩系数
    W: float                       # 调谐半宽 (Hz)
    gain_individualised: float     # 期望相对带动量
    gain_best_fixed: float         # 最优固定频率 (= μ_pop) 的期望相对带动量
    gain_device_fixed: float       # 现设备 10 Hz 的期望相对带动量
    incremental_vs_best_fixed: float
    incremental_vs_device: float
    quantisation_loss: float
    fallback: bool
    reasons: list[str]

    def summary(self) -> str:
        tag = "  [FALLBACK]" if self.fallback else ""
        return (
            f"Ω* = {self.omega_star:.2f} Hz{tag}\n"
            f"  ω̂̄ = {self.f_hat:.2f} ± {self.se:.2f} Hz   W = {self.W:.2f} Hz   "
            f"λ = {self.lam:.2f}  (τ = {self.tau:.2f} Hz)\n"
            f"  期望带动量:  个体化 {self.gain_individualised:.3f}  |  "
            f"最优固定 {self.gain_best_fixed:.3f}  |  设备 10 Hz {self.gain_device_fixed:.3f}\n"
            f"  增量:  vs 最优固定 {self.incremental_vs_best_fixed:+.1%}  |  "
            f"vs 设备 10 Hz {self.incremental_vs_device:+.1%}"
            + (f"\n  量化损失 {self.quantisation_loss:.1%}" if self.quantisation_loss > 1e-4 else "")
            + ("\n  原因: " + "; ".join(self.reasons) if self.reasons else "")
        )


def prescribe(
    freqs=None,
    psd=None,
    *,
    peak: PeakEstimate | None = None,
    prior: Prior = Prior(),
    device_fixed: float = 10.0,
    fallback_to: str = "population",
    freq_step: float | None = 0.5,
    W_default: float = 1.0,
    W_bounds: tuple[float, float] = (0.3, 3.0),
    **peak_kw,
) -> Prescription:
    """
    端到端：静息 PSD → 建议刺激频率 + 全部收益核算。

    可以直接传 `peak=PeakEstimate(...)` 跳过谱估计（用于虚拟队列）。

    降级策略
    --------
    **不需要 if/else 分支。** 任何质量问题都会推高 s，λ = σ²/(σ²+s²) 随之压向 0，
    收缩解自动退化为 μ_pop。这正是收缩解优于"个体化 or 固定"二选一的地方 ——
    降级是连续的，不存在阈值抖动。

    仅当拟合彻底失败（f_peak 非有限）时才走硬回退，此时：

      fallback_to="population" (默认) : 回到 μ_pop。**统计上正确** —— 没有个体
          信息时最优猜测就是人群均值，而不是设备出厂设定的那个数。
      fallback_to="device"            : 回到 device_fixed。保守/监管口径，
          代价是若 μ_pop ≠ device_fixed 则系统性失谐（见 headline_numbers.py）。
    """
    if peak is None:
        if freqs is None or psd is None:
            raise ValueError("需要 (freqs, psd) 或 peak")
        peak = estimate_mu_peak(freqs, psd, **peak_kw)

    if fallback_to not in ("population", "device"):
        raise ValueError("fallback_to 必须是 'population' 或 'device'")
    reasons = list(peak.reasons)

    if not np.isfinite(peak.f_peak):
        target = prior.mu_pop if fallback_to == "population" else device_fixed
        return Prescription(quantise(target, freq_step), target, np.nan, np.nan,
                            np.nan, 0.0, W_default, np.nan, np.nan, np.nan,
                            np.nan, np.nan, 0.0, True,
                            reasons + [f"无可用 mu 峰，硬回退到 {fallback_to}"])

    W = float(np.clip(peak.hwhm if np.isfinite(peak.hwhm) else W_default, *W_bounds))
    if not (W_bounds[0] < peak.hwhm < W_bounds[1]):
        reasons.append(f"W={peak.hwhm:.2f} 越界，已截断到 {W:.2f} Hz")

    mean, tau, lam = shrink(peak.f_peak, peak.se_f_peak, prior)
    omega_q = quantise(mean, freq_step)

    g_ind = expected_gain(omega_q, mean, tau, W)
    g_ind_cont = expected_gain(mean, mean, tau, W)
    g_best_fixed = expected_gain(prior.mu_pop, mean, tau, W)
    g_device = expected_gain(device_fixed, mean, tau, W)

    return Prescription(
        omega_star=omega_q,
        omega_bayes=mean,
        f_hat=peak.f_peak,
        se=peak.se_f_peak,
        tau=tau,
        lam=lam,
        W=W,
        gain_individualised=g_ind,
        gain_best_fixed=g_best_fixed,
        gain_device_fixed=g_device,
        incremental_vs_best_fixed=g_ind / g_best_fixed - 1.0,
        incremental_vs_device=g_ind / g_device - 1.0,
        quantisation_loss=1.0 - g_ind / g_ind_cont,
        fallback=lam < 0.1,
        reasons=reasons,
    )


# =========================================================================== #
# 4. 人群层面的先验收益（试验设计用）
# =========================================================================== #
def population_incremental_gain(s: float, prior: Prior = Prior(), W: float = 1.0):
    """
    还没拿到任何个体数据时，个体化相对**最优固定频率**的人群期望增量。

        最优固定 (Ω = μ_pop) : h(σ_pop / W)
        收缩个体化           : h(τ / W),  τ = sσ/sqrt(s²+σ²)

    τ ≤ σ 恒成立 ⟹ 增量恒非负；s → ∞ 时归零。
    这是**决定试验值不值得做、要多大样本**的那个数。
    """
    sigma = prior.sd_pop
    tau = sigma if s <= 0 else s * sigma / np.sqrt(s**2 + sigma**2)
    g_fix, g_ind = h_factor(sigma / W), h_factor(tau / W)
    return {"g_best_fixed": g_fix, "g_individualised": g_ind,
            "incremental": g_ind / g_fix - 1.0, "tau": float(tau)}
