"""
单位约定 (units.py)
===================

本项目内部**一律使用角频率 rad/s**。所有 ω, Ω, Δ, γ, K, F, D 均为 rad/s。
Hz 只出现在 I/O 边界（读取 EEG 峰频、写图注、报告结果）。

理由：第 2 章的推导（式 2.5、2.10、2.15）建立在 θ̇ = ω + ... 之上，
其中 ω 必须是角频率。文档正文的量化示例（说明 2.4、表 A.1）用的是 Hz，
这在**比值**（Δ/W）和**同幂次组合**下无害，但在混合幂次的表达式
（如式 2.28 的 u = (A²/4)·Ξ'/Ξ）里会静默出错，除非全部量纲一致。

式 2.28 的具体检验（见 tests/test_analytic.py::test_optimal_shift_unit_invariance）：
    u_rad = 2π · u_Hz  —— 在两套单位下计算，结果必须一致。
这是一条**硬断言**，防止后续开发中混用单位。
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def hz_to_rad(f_hz):
    """频率 (Hz) -> 角频率 (rad/s)。"""
    return TWO_PI * np.asarray(f_hz, dtype=float)


def rad_to_hz(w_rad):
    """角频率 (rad/s) -> 频率 (Hz)。"""
    return np.asarray(w_rad, dtype=float) / TWO_PI


def assert_rad(*values, name: str = "value", hi: float = 1.0e4):
    """
    弱类型防呆：Kuramoto 参数若以 Hz 误传入，数值会小 2π 倍。
    本函数不能真正区分单位，只能捕捉明显越界的情况。
    主要用途是在 public API 入口留下"这里应该是 rad/s"的显式痕迹。
    """
    for v in values:
        arr = np.asarray(v, dtype=float)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name}: 含有非有限值")
        if np.any(np.abs(arr) > hi):
            raise ValueError(f"{name}: |{name}| > {hi} rad/s，疑似单位错误")
    return True
