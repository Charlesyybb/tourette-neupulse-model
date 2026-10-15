"""
数值层 (numerics.py) —— M2
===========================

OA 降维（2.6）只在 D = 0 时精确。core.py 全程把噪声折进 γ_eff = γ + D，
而正文 2.11 自己承认这在完全锁定态下只是近似，从未量化过偏差。本模块提供
2.11 (b)(c) 两级更严格的方法，用来给 (a) 定误差界。

三个层级
--------
  (a) OA + γ_eff        core.py，解析，D≠0 时近似
  (b) 二阶圆累积量      本模块 `cumulant_steady`，四维实 ODE
  (c) N 振子 SDE        本模块 `simulate`，最终判据

圆累积量方程的推导（不引用，自行导出，可复核）
------------------------------------------------
密度的 Fourier 模式 a_m = ⟨e^{imθ}⟩ 满足

    ȧ_m = m(iω̄ − γ)a_m + (m/2)(H a_{m−1} − H* a_{m+1}) − m²D a_m,   a₀ = 1

其中 −mγ 来自洛伦兹分布的极点贡献，−m²D 来自相位扩散。**OA 假设 a_m = a₁^m
与 −m²D 不相容**（那需要 −mD），这正是噪声破坏 OA 的机制。

定义生成函数 Φ(k) = Σ_{m≥0} a_m k^m，圆累积量由

    1 − 1/Φ(k) = Σ_{m≥1} κ_m k^m

给出。这个变换的好处：OA 流形 a_m = Z^m 给 Φ = 1/(1−Zk)，于是
1 − 1/Φ = Zk **严格线性** ⟹ κ₁ = Z, κ_{m≥2} = 0。

展开得 κ₁ = a₁,  κ₂ = a₂ − a₁²,  κ₃ = a₃ − 2a₁a₂ + a₁³。
取截断 κ₃ = 0 ⟹ a₃ = 2a₁κ₂ + a₁³，代回层级方程得闭合的两方程组：

    κ̇₁ = (iω̄ − γ − D)κ₁ + ½[H − H*(κ₁² + κ₂)]
    κ̇₂ = [2(iω̄ − γ) − 4D]κ₂ − 2D κ₁² − H* κ₁ κ₂                      (★)

    H = K κ₁ + F e^{iΩt}

自洽性检验（已在 tests/test_numerics.py 中断言）
  · D = 0 ⟹ κ₂ = 0 是不变流形，κ̇₁ 退回 式 (2.9) 的 OA 方程
  · F = K = 0 ⟹ κ₁ 以速率 γ + D 衰减，即 γ_eff = γ + D 在**线性阶**精确
  · κ₂ 由 −2Dκ₁² 驱动 ⟹ **偏差随 ρ² 增长**，故 OA 在弱驱动下好、锁定态下差

由 (★) 可读出偏差的量级：稳态 |κ₂|/|κ₁|² ≈ D/(γ + 2D)，而 κ₂ 通过 −½H*κ₂
反馈进 κ̇₁，与 −½H*κ₁² 项同量级。**D ~ γ 且 ρ 不小时修正是 O(1)，不是小量。**

数值积分说明
------------
N 振子 SDE 用**算子分裂**：ω_i 项有精确解（相位线性推进），单独精确推进；
耦合+噪声项有界（≤|H|），用 Euler–Maruyama。这样洛伦兹分布的重尾（Cauchy
抽样会出现 |ω| ~ 10³γ 的振子）不再造成刚性，无需截断分布。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

__all__ = [
    "lorentzian_frequencies",
    "SimResult",
    "simulate",
    "simulate_batch",
    "cumulant_rhs",
    "cumulant_steady",
    "spontaneous_psd",
    "fit_lorentzian_psd",
    "hwhm_with_error",
    "acf",
    "acf_decay_rate",
    "ou_reference",
    "driven_tuning_curve",
    "driven_hwhm",
]


# --------------------------------------------------------------------------- #
# 固有频率
# --------------------------------------------------------------------------- #
def lorentzian_frequencies(n: int, omega_bar: float, gamma: float,
                           deterministic: bool = True, seed: int = 0):
    """
    洛伦兹（柯西）分布的固有频率。

    deterministic=True 用确定性分位点 ω_i = ω̄ + γ·tan(π(u_i − ½))，
    u_i = (i+½)/n。相比随机抽样，有限尺度涨落显著更小，且可复现。
    不做截断 —— 重尾由算子分裂消化。
    """
    if deterministic:
        u = (np.arange(n) + 0.5) / n
    else:
        u = np.random.default_rng(seed).uniform(0.0, 1.0, n)
    return omega_bar + gamma * np.tan(np.pi * (u - 0.5))


# --------------------------------------------------------------------------- #
# (c) N 振子 SDE
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    t: np.ndarray
    w: np.ndarray          # 旋转系序参量 w = z e^{−iΩt}（F=0 时按 ω̄ 旋转）
    frame_rate: float      # 旋转系角频率

    def steady(self, burn_frac: float = 0.4):
        k = int(len(self.t) * burn_frac)
        return self.t[k:], self.w[k:]

    def rho_mean(self, burn_frac: float = 0.4) -> float:
        """稳态段 ⟨|w|⟩ —— 门控看到的量。"""
        return float(np.mean(np.abs(self.steady(burn_frac)[1])))

    def rho_locked(self, burn_frac: float = 0.4) -> float:
        """稳态段 |⟨w⟩| —— 与刺激的锁定程度，对应 ITC 与 式 (2.15) 的 ρ*。"""
        return float(np.abs(np.mean(self.steady(burn_frac)[1])))


def simulate(n=4000, omega_bar=0.0, gamma=1.0, K=0.0, F=0.0, D=0.0,
             Omega=None, T=200.0, dt=5e-4, seed=0, record_every=20,
             deterministic_freqs=True):
    """
    积分 式 (2.5)：dθ_i = [ω_i + Im(H e^{−iθ_i})]dt + √(2D)dW_i,  H = Kz + Fe^{iΩt}

    在以 Omega（F=0 时取 ω̄）旋转的参考系中积分，使系统自治。
    算子分裂：先精确推进 (ω_i − Ω)dt，再显式处理耦合与噪声。
    """
    Omega = omega_bar if Omega is None else Omega
    rng = np.random.default_rng(seed)
    om = lorentzian_frequencies(n, omega_bar, gamma, deterministic_freqs, seed)
    dom = om - Omega                                  # 旋转系去谐
    phi = rng.uniform(0.0, 2 * np.pi, n)

    nsteps = int(T / dt)
    nrec = nsteps // record_every
    tt = np.empty(nrec)
    ww = np.empty(nrec, dtype=complex)

    sq = np.sqrt(2.0 * D * dt) if D > 0 else 0.0
    rot = np.exp(1j * dom * dt)                       # 频率项的精确传播子
    e = np.exp(1j * phi)
    j = 0
    for k in range(nsteps):
        e = e * rot                                   # 精确：ω 项
        w = e.mean()
        H = K * w + F                                 # 旋转系中 Fe^{iΩt} → F
        # Im(H e^{−iφ}) = Im(H)cosφ − Re(H)sinφ
        c, s = e.real, e.imag
        dphi = (H.imag * c - H.real * s) * dt
        if sq:
            dphi = dphi + sq * rng.standard_normal(n)
        e = e * np.exp(1j * dphi)
        if k % record_every == 0 and j < nrec:
            tt[j] = k * dt
            ww[j] = w
            j += 1
    return SimResult(tt[:j], ww[:j], Omega)


# --------------------------------------------------------------------------- #
# (b) 二阶圆累积量
# --------------------------------------------------------------------------- #
def cumulant_rhs(t, y, omega_bar, gamma, K, F, D, Omega):
    """(★) 在以 Omega 旋转的参考系中的实数化形式。y = [Re κ₁, Im κ₁, Re κ₂, Im κ₂]。"""
    k1 = y[0] + 1j * y[1]
    k2 = y[2] + 1j * y[3]
    H = K * k1 + F
    d = omega_bar - Omega
    dk1 = (1j * d - gamma - D) * k1 + 0.5 * (H - np.conj(H) * (k1**2 + k2))
    dk2 = (2j * d - 2 * gamma - 4 * D) * k2 - 2 * D * k1**2 - np.conj(H) * k1 * k2
    return [dk1.real, dk1.imag, dk2.real, dk2.imag]


def cumulant_steady(omega_bar=0.0, gamma=1.0, K=0.0, F=0.0, D=0.0, Omega=None,
                    T=400.0, y0=None, rtol=1e-10, atol=1e-12):
    """长时积分 (★) 取稳态。返回 (κ₁, κ₂)。"""
    Omega = omega_bar if Omega is None else Omega
    y0 = [1e-3, 0.0, 0.0, 0.0] if y0 is None else y0
    sol = solve_ivp(cumulant_rhs, (0.0, T), y0, method="LSODA",
                    args=(omega_bar, gamma, K, F, D, Omega),
                    rtol=rtol, atol=atol, dense_output=False)
    y = sol.y[:, -1]
    return complex(y[0], y[1]), complex(y[2], y[3])


def simulate_batch(params, n=4000, omega_bar=0.0, gamma=1.0, T=200.0, dt=5e-4,
                   seed=0, record_every=20, deterministic_freqs=True):
    """
    批量版：把 B 组参数向量化为矩阵的行，一次循环全部推进。

    Python 层的循环次数与单次运行相同，但每步做 B 倍的工作 —— 扫描 Ω 或 D
    的成本几乎与单次相同。这是让 M2 的参数扫描可行的关键。

    Parameters
    ----------
    params : dict of array-like，长度 B，键取自 {"K", "F", "D", "Omega"}
             未给出的键取标量默认 0。

    Returns
    -------
    list[SimResult]
    """
    keys = ("K", "F", "D", "Omega")
    B = max(len(np.atleast_1d(v)) for v in params.values())
    P = {k: np.broadcast_to(np.atleast_1d(params.get(k, 0.0)).astype(float),
                            (B,)).copy() for k in keys}
    if "Omega" not in params:
        P["Omega"][:] = omega_bar

    rng = np.random.default_rng(seed)
    om = lorentzian_frequencies(n, omega_bar, gamma, deterministic_freqs, seed)
    dom = om[None, :] - P["Omega"][:, None]           # (B, n)
    e = np.exp(1j * rng.uniform(0.0, 2 * np.pi, (B, n)))

    nsteps = int(T / dt)
    nrec = nsteps // record_every
    tt = np.empty(nrec)
    ww = np.empty((nrec, B), dtype=complex)
    rot = np.exp(1j * dom * dt)
    sq = np.sqrt(2.0 * P["D"] * dt)[:, None]
    Kc, Fc = P["K"][:, None], P["F"][:, None]
    noisy = bool(np.any(P["D"] > 0))

    j = 0
    for k in range(nsteps):
        e *= rot
        w = e.mean(axis=1, keepdims=True)
        H = Kc * w + Fc
        dphi = (H.imag * e.real - H.real * e.imag) * dt
        if noisy:
            dphi = dphi + sq * rng.standard_normal((B, n))
        e *= np.exp(1j * dphi)
        if k % record_every == 0 and j < nrec:
            tt[j] = k * dt
            ww[j] = w[:, 0]
            j += 1
    return [SimResult(tt[:j], ww[:j, b], float(P["Omega"][b])) for b in range(B)]


# --------------------------------------------------------------------------- #
# 自发谱：易感率恒等式的检验
# --------------------------------------------------------------------------- #
def _lor(f, amp, centre, hwhm, floor):
    return amp / (1.0 + ((f - centre) / hwhm) ** 2) + floor


def spontaneous_psd(res: SimResult, burn_frac=0.4, nperseg=None):
    """
    未驱动运行的序参量功率谱（实验室系频率，Hz）。

    理论（RPA 线性响应）：
        S_z(ω) = S₀(ω)/|1 − Kχ₀|²，  χ₀ = ½/(γ_eff + iΔω)
        S₀(ω) = (1/N)·2γ_eff/(Δω² + γ_eff²)     （K=0 的裸涨落）
        |1 − Kχ₀|² = (Δω² + (A/2)²)/(Δω² + γ_eff²)
        ⟹ S_z(ω) = (1/N)·2γ_eff/(Δω² + (A/2)²)

    **γ_eff² 因子精确抵消 ⟹ 自发谱是半宽 A/2 的洛伦兹，与受迫调谐曲线同宽。**
    这不是涨落–耗散的类比，是线性响应下的恒等式。本函数用来数值确认它。
    """
    from scipy.signal import welch
    t, w = res.steady(burn_frac)
    dt = float(np.median(np.diff(t)))
    z = w * np.exp(1j * res.frame_rate * t)          # 回实验室系
    nps = nperseg or min(len(z) // 8, 8192)
    f, p = welch(z, fs=1.0 / dt, nperseg=nps, return_onesided=False, detrend=False)
    o = np.argsort(f)
    return f[o], p[o]


def fit_lorentzian_psd(f_hz, psd, centre_hz, span_hz=6.0):
    """在峰附近拟合洛伦兹，返回 (中心 Hz, 半高半宽 Hz)。"""
    m = np.abs(f_hz - centre_hz) <= span_hz
    x, y = f_hz[m], psd[m]
    lo, hi = 1e-4, max(span_hz, 1e-3)
    p0 = [max(y.max() - np.median(y), 1e-30), float(x[np.argmax(y)]),
          float(np.clip(span_hz / 5, lo * 1.01, hi * 0.99)), max(np.median(y), 0.0)]
    popt, _ = curve_fit(_lor, x, y, p0=p0, maxfev=40000,
                        bounds=([0, centre_hz - span_hz, lo, 0],
                                [np.inf, centre_hz + span_hz, hi, np.inf]))
    return float(popt[1]), float(abs(popt[2]))


def hwhm_with_error(res: SimResult, centre_hz: float, n_chunk: int = 6,
                     burn_frac: float = 0.3, span_hz: float = 2.5):
    """
    **已弃用 —— 请用 `acf_decay_rate`。保留仅作记录。**

    把稳态段切成 n_chunk 块，各自对 Welch 谱峰拟合洛伦兹，返回 (均值, 标准误)。

    为什么被取代
    ------------
    这条路走不通，原因值得写下来：

    1. **估计器本身有偏。** 用已知宽度的过程跑同一管线，量到 −3% ~ −8% 的
       负偏差，且随峰变窄增大 —— 恰好与要测的效应同向，无法分离。
    2. **方差太大。** Welch 段数受 T 限制，单次估计的散布远大于 10–20% 的
       目标效应；分块标准误又低估了真实变异，给出虚假的置信。
    3. 谱峰拟合还叠加了窗泄漏与频率分辨率两重系统误差。

    `acf_decay_rate` 测的是同一个量（洛伦兹半宽 ⟺ 自相关衰减率），但在短
    滞后处用到全部 M−k 个样本对，不受上述三条限制，且在复 OU 参考过程上
    验证无偏（见 tests/test_numerics.py）。
    """
    import warnings
    warnings.warn("hwhm_with_error 已弃用，请用 acf_decay_rate", DeprecationWarning,
                  stacklevel=2)
    t, w = res.steady(burn_frac)
    m = len(t) // n_chunk
    out = []
    for i in range(n_chunk):
        sub = SimResult(t[i * m:(i + 1) * m], w[i * m:(i + 1) * m], res.frame_rate)
        try:
            f, p = spontaneous_psd(sub, burn_frac=0.0)
            out.append(fit_lorentzian_psd(f, p, centre_hz, span_hz)[1])
        except Exception:
            continue
    a = np.asarray(out)
    if a.size < 2:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def acf(w: np.ndarray, n_lag: int) -> np.ndarray:
    """
    复自相关 C(k) = ⟨w*(t) w(t+k)⟩，**无偏归一化**（除以 M−k，不是 M）。

    有偏归一化会引入 (1 − k/M) 的额外衰减，直接污染衰减率。用 FFT 算，
    补零到 2M 以上避免循环卷绕。
    """
    w = np.asarray(w, dtype=complex)
    M = w.size
    nfft = 1 << (2 * M - 1).bit_length()
    W = np.fft.fft(w, nfft)
    c = np.fft.ifft(np.abs(W) ** 2)[:n_lag]
    return c / (M - np.arange(n_lag))


def acf_decay_rate(res: "SimResult", burn_frac: float = 0.2, n_chunk: int = 8,
                   lag_lo: float = 0.5, lag_hi: float = 2.0,
                   rate_guess: float | None = None):
    """
    由旋转系自相关的指数衰减率估计洛伦兹半宽 A/2（rad/s）。

    区制 I 的线性响应下，未驱动的序参量涨落是复 OU 过程：
        dw = −(A/2) w dt + 噪声      ⟹      C(τ) = C(0)·exp(−(A/2)τ)
    自发功率谱是它的 Fourier 变换，即半宽 A/2 的洛伦兹 —— 所以拟合 ACF 与
    拟合谱峰宽度**测的是同一个量**，但 ACF 不受窗函数泄漏、频率分辨率、
    Welch 段数的限制，短滞后处用到全部 M−k 个样本对。

    实现要点（每一条都是踩过的坑）
      · 用 Re C 而非 |C|：|C| 在噪声底附近正偏，把衰减压平 ⟹ 低估 A/2。
      · **只拟合 τ ∈ [lag_lo/r, lag_hi/r]**。滞后超过 ~3/r 时 Ĉ 已是纯噪声，
        把它纳入拟合会让方差爆炸到 ±30%，而分块标准误反而看不出来。
        短端也要切掉：τ ≲ 0.5/r 处受采样间隔与非 RPA 快分量污染。
      · 拟合 ln(Re C) 的斜率，权重 ∝ C²（Ĉ 的方差近似与 τ 无关 ⟹
        var(ln Ĉ) ∝ C⁻²）。
      · 分块给标准误 —— 没有误差棒就分不清系统偏差与涨落。

    Returns
    -------
    (rate, se) : 单位 rad/s
    """
    t, w = res.steady(burn_frac)
    dts = float(np.median(np.diff(t)))

    def _one(x: np.ndarray) -> float:
        n_probe = min(x.size // 3, 20000)
        if n_probe < 16:
            return float("nan")
        c = acf(x, n_probe).real
        c0 = c[0]
        if not np.isfinite(c0) or c0 <= 0:
            return float("nan")
        r = rate_guess
        if r is None:                                  # 粗估：首次跌破 1/e
            below = np.nonzero(c / c0 < np.exp(-1.0))[0]
            if below.size == 0:
                return float("nan")
            r = 1.0 / max(below[0] * dts, dts)
        klo = max(int(lag_lo / (r * dts)), 1)
        khi = min(int(lag_hi / (r * dts)) + 1, n_probe)
        if khi - klo < 5:
            return float("nan")
        lag = np.arange(klo, khi) * dts
        y = c[klo:khi]
        good = y > 0
        if good.sum() < 5:
            return float("nan")
        lag, y = lag[good], y[good]
        wt = (y / c0) ** 2
        # 加权最小二乘：ln y = b − r·τ
        sw = wt.sum()
        mx = (wt * lag).sum() / sw
        my = (wt * np.log(y)).sum() / sw
        num = (wt * (lag - mx) * (np.log(y) - my)).sum()
        den = (wt * (lag - mx) ** 2).sum()
        if den <= 0:
            return float("nan")
        return float(-num / den)

    m = w.size // n_chunk
    vals = np.array([_one(w[i * m:(i + 1) * m]) for i in range(n_chunk)])
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size))


def ou_reference(rate: float, T: float, dt: float, seed: int = 0,
                 sigma: float = 1.0) -> "SimResult":
    """
    已知衰减率的复 OU 过程，采样方式与 `simulate` 的输出一致。

    用途：把估计器本身的偏差与物理效应分开。任何声称在 SDE 数据上看到的
    偏差，必须先证明在这条**解析已知**的参考轨迹上不出现。
    """
    n = int(T / dt)
    rng = np.random.default_rng(seed)
    a = np.exp(-rate * dt)
    s = sigma * np.sqrt(1.0 - a * a)
    x = np.empty(n, dtype=complex)
    z = complex(rng.standard_normal(), rng.standard_normal()) * sigma / np.sqrt(2)
    for i in range(n):
        z = a * z + s * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        x[i] = z
    return SimResult(np.arange(n) * dt, x, 0.0)


def driven_hwhm(Omegas, rho, omega_bar: float):
    """
    对受迫调谐曲线 ρ(Ω) 拟合 式 (2.16)：ρ = F/√(A² + 4Δ²)，返回半高半宽 W（rad/s）。

    这是易感率恒等式的**另一半**。恒等式断言它等于自发 ACF 给出的 A/2。
    """
    Omegas = np.asarray(Omegas, float)
    rho = np.asarray(rho, float)

    def model(O, amp, W):
        return amp / np.sqrt(1.0 + ((omega_bar - O) / W) ** 2)

    popt, _ = curve_fit(model, Omegas, rho, maxfev=40000,
                        p0=[rho.max(), max(np.ptp(Omegas) / 4, 1e-3)],
                        bounds=([0.0, 1e-6], [np.inf, np.inf]))
    return float(popt[1])


def driven_tuning_curve(Omegas, n=3000, omega_bar=0.0, gamma=1.0, K=0.0,
                        F=0.05, D=0.0, T=150.0, dt=5e-4, seed=0):
    """扫 Ω，返回锁定幅度 |⟨w⟩|(Ω)。用于对照自发谱宽度。批量执行。"""
    res = simulate_batch({"K": K, "F": F, "D": D, "Omega": np.asarray(Omegas, float)},
                         n=n, omega_bar=omega_bar, gamma=gamma, T=T, dt=dt, seed=seed)
    return np.array([r.rho_locked() for r in res])
