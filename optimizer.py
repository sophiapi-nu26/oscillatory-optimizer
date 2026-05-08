"""Optimization of  max_a a^* Q a  subject to  ||r||_p <= R, r_i = |a_i|, p in {1,2,inf}.

For the time-averaged real objectives, Q is Hermitian PSD, built by
response.build_quadratic_form. The closed-form / cheap-iteration methods below
exploit the structure of each constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass
class OptResult:
    a_star: np.ndarray          # complex (n,)
    r_star: np.ndarray          # real (n,) >= 0
    phi_star: np.ndarray        # real (n,) in [0, 2 pi)
    J_star: float               # achieved objective
    method: str                 # "l2_eig" | "linf_power" | "l1_frank_wolfe"
    history: list[float]        # objective trajectory (per restart best so far)


def _quad(a: np.ndarray, Q: np.ndarray) -> float:
    return float(np.real(np.vdot(a, Q @ a)))


def _polar(a: np.ndarray):
    r = np.abs(a)
    phi = np.angle(a) % (2 * np.pi)
    return r, phi


def optimize_l2(Q: np.ndarray, R: float = 1.0) -> OptResult:
    """max ||a||_2 <= R  of  a^* Q a  =  R^2 lambda_max(Q),
    attained at the top eigenvector (scaled by R)."""
    Qh = 0.5 * (Q + Q.conj().T)
    w, V = eigh(Qh)
    a = R * V[:, -1]
    r, phi = _polar(a)
    return OptResult(a, r, phi, _quad(a, Q), "l2_eig", [_quad(a, Q)])


def optimize_linf(
    Q: np.ndarray,
    R: float = 1.0,
    num_restarts: int = 20,
    max_iter: int = 500,
    tol: float = 1e-10,
    seed: int | None = None,
) -> OptResult:
    """Phase-projected power iteration:  a <- Q a; a <- R * a / |a| (entrywise).

    Multi-restart over uniform-on-torus initial points.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    Qh = 0.5 * (Q + Q.conj().T)

    best_J = -np.inf
    best_a = None
    history = []

    for _ in range(num_restarts):
        # init: random phases on |a|=R
        phi = rng.uniform(0, 2 * np.pi, size=n)
        a = R * np.exp(1j * phi)
        prev_J = _quad(a, Qh)
        for _it in range(max_iter):
            g = Qh @ a
            mag = np.abs(g)
            mag = np.where(mag < 1e-14, 1.0, mag)
            a = R * g / mag
            J = _quad(a, Qh)
            if J - prev_J < tol:
                break
            prev_J = J
        if J > best_J:
            best_J = J
            best_a = a.copy()
        history.append(best_J)

    r, phi = _polar(best_a)
    return OptResult(best_a, r, phi, best_J, "linf_power", history)


def _project_simplex_le(u: np.ndarray, R: float) -> np.ndarray:
    """Project u onto {r >= 0, sum r <= R}. If sum(max(u,0)) <= R, just clip."""
    u_clip = np.maximum(u, 0)
    if u_clip.sum() <= R:
        return u_clip
    # Project onto simplex {r >= 0, sum r = R}, Duchi et al.
    s = np.sort(u_clip)[::-1]
    cssv = np.cumsum(s) - R
    rho = np.where(s - cssv / (np.arange(len(s)) + 1) > 0)[0]
    if len(rho) == 0:
        return np.zeros_like(u_clip)
    rho = rho[-1]
    tau = cssv[rho] / (rho + 1)
    return np.maximum(u_clip - tau, 0)


def optimize_l1(
    Q: np.ndarray,
    R: float = 1.0,
    num_restarts: int = 20,
    max_iter: int = 500,
    tol: float = 1e-10,
    seed: int | None = None,
) -> OptResult:
    """Frank-Wolfe with vertex linear oracle.

    Feasible set: {a in C^n : sum |a_i| <= R}. Vertices: R * e_k * e^{i theta}
    for any k and theta. The linear oracle for direction g = Q a is:
       i* = argmax_i |g_i|,  s = R * exp(i arg(g_{i*})) e_{i*}.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    Qh = 0.5 * (Q + Q.conj().T)

    best_J = -np.inf
    best_a = None
    history = []

    for _ in range(num_restarts):
        # init at random vertex
        i0 = rng.integers(n)
        theta0 = rng.uniform(0, 2 * np.pi)
        a = np.zeros(n, dtype=complex)
        a[i0] = R * np.exp(1j * theta0)

        for k in range(max_iter):
            g = Qh @ a
            mags = np.abs(g)
            i_star = int(np.argmax(mags))
            phase = g[i_star] / max(mags[i_star], 1e-14)
            s = np.zeros(n, dtype=complex)
            s[i_star] = R * phase

            # exact line-search for max of quadratic on segment a + gamma*(s - a):
            #   f(gamma) = (a + gamma d)^* Q (a + gamma d), d = s - a
            d = s - a
            qaa = np.real(np.vdot(a, Qh @ a))
            qdd = np.real(np.vdot(d, Qh @ d))
            qad = np.real(np.vdot(a, Qh @ d))  # = real(a^* Q d) = real(d^* Q a) since Qh hermitian
            # f(gamma) = qaa + 2*gamma*qad + gamma^2 * qdd  (PSD => qdd >= 0)
            if qdd > 1e-14:
                gamma_unc = -qad / qdd  # unconstrained MAX is +inf if qdd<0, here qdd>=0 so it's a min;
                # actually for PSD Q, f is convex => max on [0,1] is at an endpoint.
                # Compare endpoints:
                f0 = qaa
                f1 = qaa + 2 * qad + qdd
                gamma = 0.0 if f0 >= f1 else 1.0
            else:
                # qdd ~ 0 => f linear in gamma; pick endpoint with larger qad
                gamma = 1.0 if qad > 0 else 0.0

            if gamma == 0.0:
                break
            a_new = a + gamma * d
            J_new = _quad(a_new, Qh)
            J_old = _quad(a, Qh)
            if abs(J_new - J_old) < tol:
                a = a_new
                break
            a = a_new

        # ensure feasibility numerically
        s_abs = np.abs(a).sum()
        if s_abs > R + 1e-9:
            a = a * (R / s_abs)

        J = _quad(a, Qh)
        if J > best_J:
            best_J = J
            best_a = a.copy()
        history.append(best_J)

    r, phi = _polar(best_a)
    return OptResult(best_a, r, phi, best_J, "l1_frank_wolfe", history)


def optimize_signal(
    Q: np.ndarray,
    norm_type: str = "l2",
    R: float = 1.0,
    num_restarts: int = 20,
    max_iter: int = 500,
    seed: int | None = None,
) -> OptResult:
    if norm_type == "l2":
        return optimize_l2(Q, R)
    elif norm_type == "linf":
        return optimize_linf(Q, R, num_restarts, max_iter, seed=seed)
    elif norm_type == "l1":
        return optimize_l1(Q, R, num_restarts, max_iter, seed=seed)
    else:
        raise ValueError(f"unknown norm_type: {norm_type}")
