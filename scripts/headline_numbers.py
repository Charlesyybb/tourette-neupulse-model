"""
两个数字 + 一个结构性结论 (scripts/headline_numbers.py)
=======================================================

(1) 固定 10 Hz 在**人群层面**损失多少可达带动量 —— 本课题摘要该出现的数字。
    正文 说明 2.4 只给了单个体的 55%；人群平均才是可引用的量。

(2) 若必须保留单一固定频率，最优的那个是几赫兹。这是零成本、可立即落地的建议。

(3) 一维频率扫描只能定出 A = 2γ_eff − K（式 2.37）；**二维 (Ω, I) 扫描**能否
    把 γ 与 K 分开？附录 A.4 注只说了区制 I 与 II 的可辨识组合不同，
    没讨论强度轴。这里用 Fisher/Jacobian 秩直接回答。
"""

from __future__ import annotations

import numpy as np

from fkm.core import A_width, hwhm, rho_regime1, solve_rho
from fkm.units import hz_to_rad, rad_to_hz


# --------------------------------------------------------------------------- #
# (1)(2) 人群层面的失谐代价
# --------------------------------------------------------------------------- #
def population_loss(mu_hz, sd_hz, W_hz, Omega_fix_hz=10.0, n=200_001, seed=0):
    """
    ⟨ρ/ρ_max⟩ = ⟨1/sqrt(1+((ω̄−Ω_fix)/W)²)⟩，按 ω̄ ~ N(mu, sd²) 求期望。
    式 (2.19) 是纯比值，Hz 与 rad/s 下同值，故此处直接用 Hz。
    """
    rng = np.random.default_rng(seed)
    ob = rng.normal(mu_hz, sd_hz, n)
    ratio = 1.0 / np.sqrt(1.0 + ((ob - Omega_fix_hz) / W_hz) ** 2)
    return ratio.mean(), float((ratio < 0.5).mean())


def best_fixed_frequency(mu_hz, sd_hz, W_hz, grid=None, n=200_001, seed=0):
    """若只允许一个固定频率，使 ⟨ρ/ρ_max⟩ 最大的那个。"""
    rng = np.random.default_rng(seed)
    ob = rng.normal(mu_hz, sd_hz, n)
    grid = np.linspace(8.0, 13.0, 501) if grid is None else grid
    score = [np.mean(1 / np.sqrt(1 + ((ob - O) / W_hz) ** 2)) for O in grid]
    i = int(np.argmax(score))
    return float(grid[i]), float(score[i])


def report_population():
    print("=" * 74)
    print("(1) 固定 10 Hz 的人群失谐代价   ⟨ρ/ρ_max⟩  (1.0 = 无损失)")
    print("=" * 74)
    print(f"{'mu(Hz)':>7}{'SD':>6}{'W(Hz)':>7}{'<ratio>':>10}{'P(loss>50%)':>13}"
          f"{'best fixed':>12}{'gain':>8}")
    for mu in (10.0, 10.5, 11.0):
        for W in (0.5, 1.0, 2.0):
            m, frac = population_loss(mu, 1.0, W)
            bf, bs = best_fixed_frequency(mu, 1.0, W)
            print(f"{mu:>7.1f}{1.0:>6.1f}{W:>7.1f}{m:>10.3f}{frac:>13.1%}"
                  f"{bf:>12.2f}{bs / m - 1:>+8.1%}")
    print()
    print("  读法：'gain' 是把固定频率从 10 Hz 换成最优固定频率的相对收益 ——")
    print("  这是零成本改动。它与个体化的收益（把 ratio 拉到 1.0）不是一回事。")
    print()


# --------------------------------------------------------------------------- #
# (3) 一维 vs 二维扫描的可辨识性
# --------------------------------------------------------------------------- #
def _rho_grid(theta, Omegas, Is):
    """精确主方程 式 (2.15) 前向模型。theta = (ω̄, γ, K, c)，F = c·I。"""
    ob, g, K, c = theta
    out = np.empty(len(Omegas) * len(Is))
    k = 0
    for I in Is:
        F = c * I
        for O in Omegas:
            out[k] = solve_rho(ob - O, K, g, F).rho
            k += 1
    return out


def jacobian_svd(theta, Omegas, Is, scale=None):
    """
    对**相对**参数扰动求 Jacobian（各列按参数量级归一），再做 SVD。
    返回 (奇异值, 最小奇异方向)。相对扰动才使不同量纲的参数可比。
    """
    theta = np.array(theta, float)
    scale = np.abs(theta) if scale is None else np.asarray(scale, float)
    J = np.zeros((len(Omegas) * len(Is), len(theta)))
    for j in range(len(theta)):
        h = 1e-6 * scale[j]
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        J[:, j] = (_rho_grid(tp, Omegas, Is) - _rho_grid(tm, Omegas, Is)) / (2 * h) * scale[j]
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    return sv, Vt[-1]


def report_identifiability():
    print("=" * 74)
    print("(3) 设计几何决定可辨识维数：一维频率扫描 vs 二维 (频率 x 强度) 扫描")
    print("=" * 74)

    ob = hz_to_rad(10.5)
    gamma = hz_to_rad(1.0)
    K = hz_to_rad(0.7)
    c = hz_to_rad(0.05)                       # F = c·I，I 以 mA 计
    theta = (ob, gamma, K, c)
    A_hz = rad_to_hz(A_width(gamma, K))
    print(f"  真值: ω̄=10.5 Hz, γ=1.0 Hz, K=0.7 Hz  ⟹ A=2γ−K={A_hz:.2f} Hz, "
          f"W={A_hz / 2:.2f} Hz")
    print(f"  F/A 在 I=1 mA 时为 {c / A_width(gamma, K):.3f}（深在区制 I）\n")

    Om = hz_to_rad(np.linspace(7.5, 13.5, 25))

    designs = [
        ("1D 频率扫描, I=1 mA",              [1.0]),
        ("2D 低强度段, I=1,2,3 mA",          [1.0, 2.0, 3.0]),
        ("2D 跨饱和拐点, I=1,5,20,60 mA",    [1.0, 5.0, 20.0, 60.0]),
    ]
    print(f"  {'设计':<30}{'σ₄/σ₁':>10}   {'零方向 (绝对), 归一到 δγ=1':<28}{'δK/δγ':>8}")
    print("  " + "-" * 86)
    for name, Is in designs:
        sv, v = jacobian_svd(theta, Om, np.array(Is))
        # v 是相对扰动坐标下的方向；乘回各参数量级还原为绝对扰动
        v_abs = v * np.abs(np.array(theta, float))
        v_abs = v_abs / v_abs[1]                       # 归一到 δγ = 1
        vs = "(" + ", ".join(f"{x:+.2f}" for x in v_abs) + ")"
        print(f"  {name:<30}{sv[-1] / sv[0]:>10.1e}   {vs:<28}{v_abs[2]:>8.2f}")

    print()
    print("  三点结论（对正文 2.16 与 附录 A.4 的修正）：")
    print()
    print("  a) 式 (2.37) 的'只有三个可辨识参数'在**精确主方程**下不是严格的，")
    print("     只是区制 I 极限（ρ≪1）下的精确退化。正确表述是")
    print("     '三个良态方向 + 一个病态方向'，且病态程度可以量化。")
    print()
    print("  b) 深在区制 I 时零方向数值上就是 (0, 1, 2, 0) —— 式 (A.42) 的解析结论")
    print("     (δγ, δK) ∝ (1,2) 得到三位有效数字的确认，且 ω̄ 与 c 分量严格为零。")
    print("     跨饱和后该方向旋转到 δK/δγ ≈ 3，说明解析退化本身已经不成立。")
    print()
    print("  c) 强度轴只有**跨过饱和拐点**才带来信息：σ₄/σ₁ 提升约两个数量级。")
    print("     但临床 3.5–7 mA 全部阈下（运动阈值 ~8 mA），拐点跨不过去。")
    print("     所以现实方案仍是：γ、K 由静息 EEG 谱先验给定，只拟合 B。")
    print("     这应写为**设计约束**，而不是模型缺陷 —— 二者在论文里读感完全不同。")
    print()


if __name__ == "__main__":
    report_population()
    report_identifiability()
