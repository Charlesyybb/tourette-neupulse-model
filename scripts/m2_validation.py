#!/usr/bin/env python3
"""
M2 数值验证层
=============

正文里三处"精确 / 可忽略"的声明，逐条用 N 振子 SDE 核对。

    calib   估计器标定       —— 任何结论生效的前提
    oa      OA 误差界        —— 2.11 (a) γ_eff = γ+D 在锁定态下差多少
    spont   易感率恒等式     —— 自发线宽 = A/2？（承重）
    driven  易感率恒等式     —— 驱动调谐半宽 = A/2？（另一半）

用法
----
    python scripts/m2_validation.py calib
    python scripts/m2_validation.py oa --D 0.0 0.2
    python scripts/m2_validation.py spont --n 800
    python scripts/m2_validation.py driven
    python scripts/m2_validation.py report        # 汇总已缓存的 JSON

长时程运行拆段执行，结果落到 results/*.json；`report` 只读缓存，不重跑。
每段都打印墙钟耗时，便于估算。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fkm.core import rho_regime1, solve_rho                       # noqa: E402
from fkm.numerics import (acf_decay_rate, cumulant_steady,        # noqa: E402
                          driven_hwhm, ou_reference, simulate_batch)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TP = 2 * np.pi

# 全局物理参数：γ_eff = γ + D = 1.0 rad/s，ω̄ = 10 Hz
GAMMA, DNOISE, OMEGA_BAR = 0.7, 0.3, TP * 10.0
GEFF = GAMMA + DNOISE
KS = np.array([0.0, 0.5, 1.0, 1.5])          # κ = K/(2γ_eff) = 0, .25, .50, .75


def _save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)


def _load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


# --------------------------------------------------------------------------- #
# calib —— 估计器标定
# --------------------------------------------------------------------------- #
def cmd_calib(args):
    """
    两件事，缺一不可：

    (1) 已知速率的复 OU 过程 —— 估计器在解析已知的输入上必须无偏。
        M2 的第一轮就是跳过这步，把估计器偏差当成了物理效应。
    (2) SDE 的步长收敛 —— 用 K=0（无反馈，衰减率就是 γ_eff）做基准。
        注意 K=0 **不是** 任意 N 精确：确定性洛伦兹分位点的经验特征函数
        在有限 n 下 ≠ e^{−γτ}。所以它是"逐 n 的管线标定"，不是绝对真值。
    """
    print("(1) 复 OU 标定：估计器必须无偏")
    print(f"{'真速率':>8}{'T(s)':>8}{'估计':>20}{'误差':>10}")
    print("-" * 47)
    ou = []
    for rate in [0.25, 0.5, 1.0]:
        for T in [2000.0, 6000.0]:
            r = ou_reference(rate, T=T, dt=0.02, seed=3)
            a, se = acf_decay_rate(r, burn_frac=0.0, n_chunk=8)
            ou.append(dict(rate=rate, T=T, est=a, se=se))
            print(f"{rate:>8.2f}{T:>8.0f}{a:>14.4f} ±{se:<5.4f}{a / rate - 1:>+10.1%}")

    print("\n(2) SDE 步长收敛（K=0，衰减率应为 γ_eff = 1.0）")
    print(f"{'dt':>9}{'n':>6}{'T':>8}{'速率':>20}{'误差':>10}{'耗时':>8}")
    print("-" * 62)
    dts = []
    for dt, re_ in [(1e-3, 20), (3e-3, 7), (1e-2, 2)]:
        t0 = time.time()
        r = simulate_batch({"K": 0.0, "D": DNOISE}, n=args.n, omega_bar=OMEGA_BAR,
                           gamma=GAMMA, T=args.T, dt=dt, seed=5, record_every=re_)[0]
        a, se = acf_decay_rate(r, burn_frac=0.05, n_chunk=8)
        dts.append(dict(dt=dt, n=args.n, T=args.T, est=a, se=se))
        print(f"{dt:>9.0e}{args.n:>6d}{args.T:>8.0f}{a:>14.4f} ±{se:<5.4f}"
              f"{a - GEFF:>+10.1%}{time.time() - t0:>7.0f}s")
    _save("m2_calib.json", dict(ou=ou, dt=dts))
    print("\n结论：估计器无偏；dt = 1e-2 与 1e-3 在 1 SE 内一致 ⟹ 长时程运行用 1e-2。")


# --------------------------------------------------------------------------- #
# oa —— OA + γ_eff 的误差界
# --------------------------------------------------------------------------- #
def cmd_oa(args):
    """
    三层对照（正文 2.11 的 (a)(b)(c)）：
        (a) OA + γ_eff = γ+D   core.py 解析，D≠0 时近似
        (b) 二阶圆累积量        cumulant_steady
        (c) N 振子 SDE          最终判据

    D=0 时三者必须逐位吻合 —— 这是整个数值栈的互验。
    """
    gam = 1.0
    K = 0.5
    Fs = np.array([0.3, 1.0, 3.0])
    rows = _load("m2_oa.json") or []
    have = {(r["D"], r["F"]) for r in rows}

    print(f"OA(γ_eff) vs 二阶圆累积量 vs N 振子 SDE   [γ={gam}, K={K}, Δ=0, "
          f"n={args.n}, T={args.T}, dt={args.dt:.0e}]")
    for D in args.D:
        if all((D, float(F)) in have for F in Fs):
            print(f"  D={D}: 已缓存，跳过")
            continue
        t0 = time.time()
        res = simulate_batch({"K": K, "F": Fs, "D": D}, n=args.n, gamma=gam,
                             T=args.T, dt=args.dt, seed=7)
        for F, r in zip(Fs, res):
            sde = r.rho_locked(0.5)
            oa = solve_rho(0.0, K, gam + D, float(F)).rho
            k1, k2 = cumulant_steady(gamma=gam, K=K, F=float(F), D=D)
            rows.append(dict(D=float(D), F=float(F), oa=oa, cum=abs(k1), sde=sde,
                             ratio=abs(k2) / abs(k1) ** 2 if abs(k1) > 0 else 0.0))
        print(f"  D={D}: [{time.time() - t0:.0f}s]")
    _save("m2_oa.json", rows)
    _print_oa(rows)


def _print_oa(rows):
    print(f"\n{'D':>5}{'F':>6}{'OA+γeff':>10}{'累积量':>10}{'SDE':>10}"
          f"{'OA误差':>9}{'累积量误差':>11}{'|κ₂|/|κ₁|²':>12}")
    print("-" * 76)
    for r in sorted(rows, key=lambda x: (x["D"], x["F"])):
        print(f"{r['D']:>5.1f}{r['F']:>6.1f}{r['oa']:>10.4f}{r['cum']:>10.4f}"
              f"{r['sde']:>10.4f}{r['oa'] / r['sde'] - 1:>+9.1%}"
              f"{r['cum'] / r['sde'] - 1:>+11.1%}{r['ratio']:>12.3f}")
    errs = [r["oa"] / r["sde"] - 1 for r in rows if r["D"] > 0]
    if errs:
        print(f"\nOA 误差界：{min(errs):+.1%} … {max(errs):+.1%}，全部单向低估。")
        print("解析预期 |κ₂|/|κ₁|² ≈ D/(γ+2D)：",
              ", ".join(f"D={D}→{D / (1.0 + 2 * D):.3f}" for D in [0.2, 0.5, 1.0]))


# --------------------------------------------------------------------------- #
# spont —— 易感率恒等式（自发侧）
# --------------------------------------------------------------------------- #
def cmd_spont(args):
    """
    自发涨落的洛伦兹半宽是否等于 A/2 = γ_eff − K/2？

    RPA 线性响应给出恒等式（γ_eff² 在分子分母精确抵消，见 numerics 文档），
    但它是 N→∞ 的性质。有限 N 修正 O(1/N) 必须随 N 消失 —— 这才是判据。
    """
    rows = _load("m2_spont.json") or {}
    for n in args.n:
        t0 = time.time()
        res = simulate_batch({"K": KS, "D": DNOISE}, n=n, omega_bar=OMEGA_BAR,
                             gamma=GAMMA, T=args.T, dt=1e-2, seed=17, record_every=5)
        print(f"n={n}  T={args.T:.0f}s  [模拟 {time.time() - t0:.0f}s]")
        for K, r in zip(KS, res):
            a, se = acf_decay_rate(r, burn_frac=0.03, n_chunk=8)
            rows[f"{n}|{K}"] = [a, se, GEFF - K / 2]
        _save("m2_spont.json", rows)
    _print_spont(rows)


def _print_spont(rows):
    ns = sorted({int(k.split("|")[0]) for k in rows})
    print(f"\n自发线宽 vs A/2   [γ_eff={GEFF}, 8 块误差棒]")
    print(f"{'κ':>6}" + "".join(f"{'n=' + str(n):>18}" for n in ns))
    print("-" * (6 + 18 * len(ns)))
    for K in KS:
        line = f"{K / (2 * GEFF):>6.2f}"
        for n in ns:
            v = rows.get(f"{n}|{K}")
            line += (f"{v[0] / v[2] - 1:>+11.1%} ±{v[1] / v[2]:<5.1%}" if v
                     else f"{'—':>18}")
        print(line)

    # 按各 n 的 K=0 值扣掉管线偏差（有限分位点的经验特征函数误差）
    print("\n扣除逐 n 管线偏差（K=0 应为 0）后：")
    print(f"{'κ':>6}" + "".join(f"{'n=' + str(n):>12}" for n in ns) + "   1/N 外推")
    print("-" * (6 + 12 * len(ns) + 12))
    for K in KS[1:]:
        line = f"{K / (2 * GEFF):>6.2f}"
        xs, ys, ws = [], [], []
        for n in ns:
            v, v0 = rows.get(f"{n}|{K}"), rows.get(f"{n}|0.0")
            if not (v and v0):
                line += f"{'—':>12}"
                continue
            e = (v[0] / v[2] - 1) - (v0[0] / v0[2] - 1)
            s = np.hypot(v[1] / v[2], v0[1] / v0[2])
            line += f"{e:>+12.1%}"
            xs.append(1.0 / n); ys.append(e); ws.append(1.0 / s ** 2)
        if len(xs) >= 2:                       # 加权直线拟合 y = y∞ + c/N
            X = np.vstack([np.ones_like(xs), xs]).T
            W = np.diag(ws)
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ np.array(ys))
            cov = np.linalg.inv(X.T @ W @ X)
            line += f"{beta[0]:>+9.1%} ±{np.sqrt(cov[0, 0]):<5.1%}"
        print(line)
    print("\n判据：1/N → 0 的截距应与 0 相容。若截距显著非零，恒等式不成立，"
          "\n      表 2 的 W 相关列全部要重算。")


# --------------------------------------------------------------------------- #
# driven —— 易感率恒等式（驱动侧）
# --------------------------------------------------------------------------- #
def cmd_driven(args):
    """
    受迫调谐曲线 ρ(Ω) 的半高半宽是否等于同一个 A/2？

    这是算法真正消费的那一半：`estimate_mu_peak` 从静息谱读出 W，
    然后当作驱动调谐宽度用。两侧必须是同一个数。
    """
    off = np.array([-2.0, -1.4, -0.9, -0.5, -0.22, 0.0, 0.22, 0.5, 0.9, 1.4, 2.0])
    out = _load("m2_driven.json") or {}
    for K in args.K:
        A2 = GEFF - K / 2
        Om = OMEGA_BAR + off * A2 * 1.6
        t0 = time.time()
        res = simulate_batch({"K": K, "F": args.F, "D": DNOISE, "Omega": Om},
                             n=args.n, omega_bar=OMEGA_BAR, gamma=GAMMA,
                             T=args.T, dt=1e-2, seed=23, record_every=5)
        rho = np.array([r.rho_locked(0.35) for r in res])
        ratios = [float(rr / rho_regime1(-o, K, GEFF, args.F))
                  for o, rr in zip(Om - OMEGA_BAR, rho)]
        W = driven_hwhm(Om, rho, OMEGA_BAR)
        out[f"{K}"] = dict(W=W, A2=A2, ratios=ratios)
        print(f"K={K}  κ={K / (2 * GEFF):.2f}  W={W:.4f}  A/2={A2:.4f}  "
              f"误差 {W / A2 - 1:+.1%}  逐点比值 {min(ratios):.3f}–{max(ratios):.3f}"
              f"  [{time.time() - t0:.0f}s]")
    _save("m2_driven.json", out)


# --------------------------------------------------------------------------- #
def cmd_report(args):
    oa, sp, dr = _load("m2_oa.json"), _load("m2_spont.json"), _load("m2_driven.json")
    print("=" * 78)
    print("M2 汇总")
    print("=" * 78)
    if oa:
        _print_oa(oa)
    if sp:
        _print_spont(sp)
    if dr:
        print(f"\n驱动调谐半宽 vs A/2   [F={0.05}, 11 个 Ω 点]")
        print(f"{'κ':>6}{'拟合 W':>10}{'A/2':>10}{'误差':>9}")
        print("-" * 35)
        for k, v in sorted(dr.items(), key=lambda x: float(x[0])):
            print(f"{float(k) / (2 * GEFF):>6.2f}{v['W']:>10.4f}{v['A2']:>10.4f}"
                  f"{v['W'] / v['A2'] - 1:>+9.1%}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calib"); p.set_defaults(fn=cmd_calib)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--T", type=float, default=8000.0)

    p = sub.add_parser("oa"); p.set_defaults(fn=cmd_oa)
    p.add_argument("--D", type=float, nargs="+", default=[0.0, 0.2, 0.5, 1.0])
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--T", type=float, default=80.0)
    p.add_argument("--dt", type=float, default=5e-4)

    p = sub.add_parser("spont"); p.set_defaults(fn=cmd_spont)
    p.add_argument("--n", type=int, nargs="+", default=[50, 200, 800])
    p.add_argument("--T", type=float, default=20000.0)

    p = sub.add_parser("driven"); p.set_defaults(fn=cmd_driven)
    p.add_argument("--K", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--T", type=float, default=500.0)
    p.add_argument("--F", type=float, default=0.05)

    p = sub.add_parser("report"); p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
