"""Streamlit app: oscillatory information optimizer for Shi repelling networks.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from directions import (
    DEFAULT_PRESET,
    PRESET_DESCRIPTIONS,
    make_direction_vector,
    parse_custom_vector,
    relevant_presets,
)
from network_generators import (
    generate_cycle_signed,
    generate_signed_er,
    generate_signed_lattice,
    generate_signed_lollipop,
    generate_signed_star,
    generate_signed_wheel,
    generate_three_block_signed,
    generate_two_block_balanced,
)
from optimizer import optimize_signal
from plots import (
    plot_agent_trajectories,
    plot_centrality_bars,
    plot_centrality_scatter,
    plot_functional_over_time,
    plot_magnitude_bar,
    plot_phase_histogram,
    plot_signed_network,
)
from response import (
    build_quadratic_form,
    eigenvector_centrality_unsigned,
    frequency_response,
    functional_over_time,
    receiver_frequency_centrality,
    source_frequency_centrality,
    steady_state_trajectory,
)
from shi_model import (
    build_shi_repelling_matrix,
    compute_laplacians,
    compute_stability_diagnostics,
    compute_threshold,
)


st.set_page_config(
    page_title="Shi Oscillatory Optimizer",
    layout="wide",
)

st.title("Oscillatory Information Optimizer")
st.caption(
    "Shi repelling-with-information dynamics: $W = I - \\alpha L_+ + \\beta L_- - D$. "
    "Optimize the oscillatory signal $a_i = r_i e^{i\\phi_i}$ at frequency $\\omega$."
)

with st.expander("About this app  —  click to read"):
    st.markdown(
        r"""
**Model.** Each of $n$ agents holds a real-valued opinion $x_i(t)$. They update by
$$
x(t+1) = W\, x(t) + D\, \theta(t), \qquad
W = I - \alpha L_+ + \beta L_- - D, \quad D = \mathrm{diag}(c_1,\dots,c_n),
$$
where $L_+$ is the Laplacian of the positive (friendly) edges, $L_-$ of the negative
(antagonistic) edges, $c_i \ge 0$ is agent $i$'s media-attention weight, and
$\theta(t) \in \mathbb{R}^n$ is an external information signal.

**The signal.** We feed in an oscillatory signal $\theta(t) = \mathrm{Re}(a\, e^{i\omega t})$,
where $a_i = r_i e^{i\phi_i}$ encodes agent $i$'s **listening magnitude** $r_i \ge 0$
and **phase lag** $\phi_i \in [0, 2\pi)$.

**The frequency response.** When $\rho(W) < 1$, the steady-state response is
$x^{\mathrm{ss}}_t = \mathrm{Re}(B_\omega a\, e^{i\omega t})$ where
$B_\omega = (e^{i\omega} I - W)^{-1} D$.

**What we optimize.** Pick $a$ to maximize a time-averaged functional of $x^{\mathrm{ss}}_t$
(squared norm or cross-sectional variance) subject to $\|r\|_p \le R$, with $p \in \{1, 2, \infty\}$.
A small calculation shows the time-averaged objective equals
$\tfrac12 a^* B_\omega^* B_\omega a$ (norm) or $\tfrac{1}{2n} a^* B_\omega^* P B_\omega a$
(variance, $P = I - \tfrac1n \mathbf 1 \mathbf 1^\top$), so the problem is a PSD quadratic form.

**Editing the graph.** The midpoint diamond on each edge is **clickable**: clicking
flips that edge's sign (positive ↔ negative). A *Reset edge flips* button is in the sidebar.
        """
    )


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Network")
    network_type = st.selectbox(
        "Type",
        ["cycle", "two_block", "three_block", "star", "erdos_renyi",
         "lollipop", "wheel", "lattice"],
        index=1,
        help="Signed network template to generate.",
    )

    if network_type == "cycle":
        n = st.slider("n", 3, 40, 12)
        sign_pattern = st.selectbox(
            "Sign pattern",
            ["all_pos", "alternating", "one_neg"],
            index=1,
            help=(
                "all_pos: every edge positive. "
                "alternating: edges alternate +/- around the ring (perfect for even n; "
                "odd n leaves one mismatch at the wrap-around). "
                "one_neg: a single negative edge; the classic frustrated cycle."
            ),
        )
    elif network_type == "two_block":
        n1 = st.slider("n1", 2, 25, 5)
        n2 = st.slider("n2", 2, 25, 5)
        p_in = st.slider("p_in (within-block)", 0.0, 1.0, 1.0)
        p_out = st.slider("p_out (across-block)", 0.0, 1.0, 0.6)
    elif network_type == "three_block":
        n1 = st.slider("n1", 2, 20, 5)
        n2 = st.slider("n2", 2, 20, 5)
        n3 = st.slider("n3", 2, 20, 5)
        p_in = st.slider("p_in", 0.0, 1.0, 1.0)
        p_out = st.slider("p_out", 0.0, 1.0, 0.6)
    elif network_type == "star":
        n_leaves = st.slider("n_leaves", 3, 30, 10)
        leaf_signs = st.selectbox("Leaf signs", ["positive", "negative", "mixed"], index=2)
    elif network_type == "erdos_renyi":
        n = st.slider("n", 4, 40, 16)
        p_edge = st.slider("p_edge", 0.0, 1.0, 0.3)
        q_neg = st.slider("q (neg edge | edge)", 0.0, 1.0, 0.4)
    elif network_type == "lollipop":
        n_clique = st.slider("clique size", 2, 20, 6)
        n_path = st.slider("path length", 0, 20, 5)
        sign_clique = st.selectbox("clique edge sign", ["positive", "negative"], index=0)
        sign_path = st.selectbox("path edge sign", ["positive", "negative"], index=0)
        sign_join = st.selectbox("join edge sign", ["positive", "negative"], index=0)
    elif network_type == "wheel":
        n_outer = st.slider("outer (rim) nodes", 3, 30, 8)
        sign_hub = st.selectbox("hub-spoke sign", ["positive", "negative"], index=0)
        sign_rim = st.selectbox("rim edge sign", ["positive", "negative"], index=1)
    elif network_type == "lattice":
        rows = st.slider("rows", 2, 10, 4)
        cols = st.slider("cols", 2, 10, 4)
        lattice_pattern = st.selectbox(
            "edge sign pattern",
            ["uniform", "checkerboard"],
            index=1,
            help=(
                "uniform: all edges same sign. "
                "checkerboard: horizontal edges +, vertical edges - (induces structural balance "
                "with the (-1)^{r+c} bipartition along one axis)."
            ),
        )
        lattice_uniform_sign = st.selectbox("uniform sign", ["positive", "negative"], index=0)

    seed = st.number_input("seed", value=0, step=1)

    if st.button("Reset edge flips"):
        st.session_state["flipped_edges"] = set()
        st.session_state["last_sel_set"] = set()

    st.header("Model")
    alpha = st.slider("alpha", 0.0, 1.0, 0.05, step=0.005,
                      help="Friendly-attention weight on positive edges.")
    beta = st.slider("beta", 0.0, 2.0, 0.005, step=0.001,
                     help="Antagonism weight on negative edges. System is stable when beta < beta_thresh(c).")
    info_mode = st.selectbox("Information weights c", ["uniform", "random_uniform"], index=0)
    if info_mode == "uniform":
        c_uniform = st.slider("c (uniform)", 0.0, 1.0, 0.15, step=0.005)
    else:
        c_lo = st.slider("c min", 0.0, 1.0, 0.05, step=0.005)
        c_hi = st.slider("c max", 0.0, 1.0, 0.2, step=0.005)

    st.header("Signal")
    omega = st.slider("omega", 0.0, float(np.pi), 0.5, step=0.01,
                      help="Signal frequency. omega=0 is constant; omega=pi is alternating.")
    objective = st.selectbox(
        "Objective", ["norm", "variance", "direction"], index=1,
        help=(
            "norm: ||x_t||^2.  "
            "variance: cross-sectional variance (polarization).  "
            "direction: amplification along a chosen direction v, J(a) = avg (v^T x_t)^2."
        ),
    )
    if objective == "direction":
        preset_options = relevant_presets(network_type)
        default_preset = DEFAULT_PRESET.get(network_type, preset_options[0])
        if default_preset not in preset_options:
            preset_options = [default_preset] + [p for p in preset_options if p != default_preset]
        direction_preset = st.selectbox(
            "Direction preset",
            preset_options,
            index=preset_options.index(default_preset),
        )
        # Show the chosen preset's description directly below the selector.
        st.caption(PRESET_DESCRIPTIONS.get(direction_preset, ""))
        if direction_preset == "custom":
            custom_text = st.text_input(
                "Custom vector v",
                value="",
                placeholder="e.g.  [1, -1, 0, 1]   or   1, 1, -1",
                help=(
                    "Comma-, space-, or semicolon-separated list of n real numbers. "
                    "Brackets are optional. The vector is normalized internally."
                ),
            )
        else:
            custom_text = None
    else:
        direction_preset = None
        custom_text = None
    norm_type = st.selectbox("Norm constraint", ["l1", "l2", "linf"], index=0,
                             help=("l1: total listening budget. l2: total energy budget. "
                                   "linf: each agent capped at R."))
    R = st.slider("budget R", 0.1, 5.0, 1.0, step=0.1)
    T = st.slider("samples T", 32, 1024, 256, step=32)

    show_centrality_labels = st.checkbox(
        "Annotate nodes with kappa", value=True,
        help="Display each node's source frequency centrality below the node.",
    )

    st.header("Solver")
    num_restarts = st.slider("restarts", 1, 100, 20)
    max_iter = st.slider("max_iter", 50, 2000, 500, step=50)
    opt_seed = st.number_input("optimizer seed", value=42, step=1)


# ============================================================
# Network construction
# ============================================================
def _build_network():
    if network_type == "cycle":
        return generate_cycle_signed(n, sign_pattern)
    elif network_type == "two_block":
        return generate_two_block_balanced(n1, n2, p_in, p_out, seed=int(seed))
    elif network_type == "three_block":
        return generate_three_block_signed(n1, n2, n3, p_in, p_out, seed=int(seed))
    elif network_type == "star":
        return generate_signed_star(n_leaves, leaf_signs, seed=int(seed))
    elif network_type == "erdos_renyi":
        return generate_signed_er(n, p_edge, q_neg, seed=int(seed))
    elif network_type == "lollipop":
        return generate_signed_lollipop(
            n_clique, n_path, sign_clique, sign_path, sign_join, seed=int(seed)
        )
    elif network_type == "wheel":
        return generate_signed_wheel(n_outer, sign_hub, sign_rim, seed=int(seed))
    elif network_type == "lattice":
        return generate_signed_lattice(
            rows, cols,
            sign=lattice_uniform_sign,
            sign_pattern=lattice_pattern,
            seed=int(seed),
        )


A_plus, A_minus, communities, positions = _build_network()
n_total = A_plus.shape[0]

# Reset edge flips when the network signature changes
net_sig = (network_type, n_total, int(A_plus.sum()), int(A_minus.sum()), int(seed))
if st.session_state.get("net_sig") != net_sig:
    st.session_state["net_sig"] = net_sig
    st.session_state["flipped_edges"] = set()
    st.session_state["last_sel_set"] = set()

# ---- apply edge flips from session state ----
flipped_edges = st.session_state.get("flipped_edges", set())
A_plus_eff = A_plus.copy()
A_minus_eff = A_minus.copy()
for fr in flipped_edges:
    i, j = sorted(fr)
    if A_plus[i, j] > 0:
        w = A_plus[i, j]
        A_plus_eff[i, j] = A_plus_eff[j, i] = 0
        A_minus_eff[i, j] = A_minus_eff[j, i] = w
    elif A_minus[i, j] > 0:
        w = A_minus[i, j]
        A_minus_eff[i, j] = A_minus_eff[j, i] = 0
        A_plus_eff[i, j] = A_plus_eff[j, i] = w

# ---- info weights ----
if info_mode == "uniform":
    c_vec = np.full(n_total, float(c_uniform))
else:
    rng_c = np.random.default_rng(int(seed))
    c_vec = rng_c.uniform(float(c_lo), float(c_hi), size=n_total)

# ---- Build W and diagnostics ----
W = build_shi_repelling_matrix(A_plus_eff, A_minus_eff, alpha, beta, c_vec)
D = np.diag(c_vec)
L_plus, L_minus = compute_laplacians(A_plus_eff, A_minus_eff)

diag = compute_stability_diagnostics(W, omega, A_plus=A_plus_eff, alpha=alpha, c=c_vec)
beta_thresh = compute_threshold(alpha, L_plus, L_minus, D)


# ============================================================
# Diagnostics strip
# ============================================================
st.subheader("Diagnostics")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("rho(W)", f"{diag['rho']:.4f}", delta="stable" if diag["is_stable"] else "UNSTABLE")
d2.metric("lambda_min", f"{diag['lambda_min']:.4f}")
d3.metric("lambda_max", f"{diag['lambda_max']:.4f}")
d4.metric("cond(eIw - W)", f"{diag['condition_number']:.2e}")
if beta_thresh is None:
    d5.metric("beta_thresh", "n/a")
elif beta_thresh == float("inf"):
    d5.metric("beta_thresh", "inf")
else:
    d5.metric("beta_thresh(c)", f"{beta_thresh:.4f}")

with st.expander("What do these diagnostics mean?"):
    st.markdown(
        r"""
- **rho(W)**: spectral radius of the update matrix. The system has a unique
bounded steady-state response only when $\rho(W) < 1$.
- **lambda_min, lambda_max**: smallest and largest eigenvalues of $W$
(when $W$ is symmetric, which is the standard case here). Both lie in $(-1, 1)$
in the stable regime.
- **cond(eIw − W)**: condition number of $e^{i\omega} I - W$. Large values mean
the frequency response $B_\omega$ is numerically ill-conditioned at this $\omega$.
- **beta_thresh(c)**: the stability threshold from Theorem 6.10 of the thesis,
$\beta_{\mathrm{thresh}}(c) = \inf_{x : x^\top L_- x > 0} \frac{x^\top(\alpha L_+ + D)x}{x^\top L_- x}$.
The system is provably stable for $\beta < \beta_{\mathrm{thresh}}$.
        """
    )

if "baseline_self_weight" in diag and not diag["baseline_self_weight_ok"]:
    st.warning(
        f"Baseline self-weight `1 - alpha * deg+ - c` is negative for some nodes "
        f"(min = {diag['baseline_self_weight_min']:.3f}). "
        f"This violates the model's nonneg-self-weight assumption."
    )

if not diag["is_stable"]:
    st.error(
        f"Spectral radius rho(W) = {diag['rho']:.4f} >= 1. "
        f"Steady-state response does not exist; optimization disabled."
    )
    st.stop()

if diag["condition_number"] > 1e10:
    st.warning(
        f"e^(i omega) I - W is poorly conditioned (cond ~ {diag['condition_number']:.2e}). "
        f"Frequency response may be numerically unstable."
    )


# ============================================================
# Optimize
# ============================================================
B = frequency_response(W, D, omega)

# Build direction vector if needed
v_dir = None
if objective == "direction":
    if direction_preset == "custom":
        v_dir, parse_err = parse_custom_vector(custom_text, n_total)
        if parse_err is not None:
            st.error(
                f"Custom vector: {parse_err}  Falling back to consensus vector "
                f"so the rest of the page still renders."
            )
            v_dir = np.ones(n_total)
        else:
            st.success(
                "Using custom v = [" + ", ".join(f"{x:g}" for x in v_dir) + "]"
                + (f"   (normalized: " + ", ".join(
                    f"{x:.3f}" for x in v_dir / np.linalg.norm(v_dir)
                  ) + ")"
                  if np.linalg.norm(v_dir) > 0 else "")
            )
    else:
        grid_shape = (rows, cols) if network_type == "lattice" else None
        v_dir = make_direction_vector(
            direction_preset,
            n_total,
            communities=communities,
            A_minus=A_minus_eff,
            grid_shape=grid_shape,
        )

Q = build_quadratic_form(B, objective, n_total, v=v_dir)

with st.spinner("Optimizing..."):
    result = optimize_signal(
        Q,
        norm_type=norm_type,
        R=float(R),
        num_restarts=int(num_restarts),
        max_iter=int(max_iter),
        seed=int(opt_seed),
    )

a_star = result.a_star
r_star = result.r_star
phi_star = result.phi_star

period_samples = max(int(round(2 * np.pi / max(omega, 1e-3))), 8)
T_plot = max(T, 4 * period_samples)
X = steady_state_trajectory(B, a_star, omega, T_plot)
V_t = functional_over_time(X, objective, v=v_dir)


# ============================================================
# Result summary
# ============================================================
st.subheader("Optimization result")
m1, m2, m3, m4 = st.columns(4)
m1.metric("J* (time-avg)", f"{result.J_star:.4f}")
m2.metric("method", result.method)
m3.metric("|r*|_1", f"{r_star.sum():.4f}")
m4.metric("|r*|_inf", f"{r_star.max():.4f}")

if objective == "direction" and v_dir is not None:
    with st.expander("Direction vector v (preset: {})".format(direction_preset)):
        st.markdown(
            r"""
The directional objective is
$$
J_{\mathrm{dir}}(a) \;=\; \frac{1}{T}\sum_{t=0}^{T-1} \big(v^\top x^{\mathrm{ss}}_t\big)^2
\;=\; \tfrac{1}{2}\, \big| v^\top B_\omega a \big|^2
\;=\; \tfrac{1}{2}\, a^* (B_\omega^* v)(B_\omega^* v)^* a,
$$
i.e. a *rank-1* PSD Hermitian quadratic form. Closed-form optima:
- $\ell_2$:   $a^* = R\, B_\omega^* v / \| B_\omega^* v \|_2$, value $\tfrac{R^2}{2}\,\|B_\omega^* v\|_2^2$.
- $\ell_\infty$: $a_i^* = R\, e^{i\arg((B_\omega^* v)_i)}$, value $\tfrac{R^2}{2}\,\|B_\omega^* v\|_1^2$.
- $\ell_1$:   $a^* = R\,e_{k^*}\,e^{i\arg((B_\omega^* v)_{k^*})}$ with $k^*=\arg\max_k|(B_\omega^* v)_k|$, value $\tfrac{R^2}{2}\,\|B_\omega^* v\|_\infty^2$.
            """
        )
        v_normalized = v_dir / max(np.linalg.norm(v_dir), 1e-12)
        st.write(
            "Normalized v (entries):",
            ", ".join(f"{x:.3f}" for x in v_normalized),
        )

with st.expander("How is J* computed and what do |r*|_1 / |r*|_inf mean?"):
    st.markdown(
        r"""
- **J***: the time-averaged objective evaluated at the optimum. By the calculation
in the *About* expander, this equals $\tfrac12 a^{*}B^{*}Ba$ for the squared-norm
objective and $\tfrac{1}{2n} a^{*}B^{*}PBa$ for the variance objective.
- **|r\*|_1**: total listening budget used. Saturates at $R$ under the $\ell_1$ constraint.
- **|r\*|_inf**: the largest single-agent magnitude. Saturates at $R$ under the
$\ell_\infty$ constraint (and also at the optimum under $\ell_1$, which concentrates on a
single vertex for PSD $Q$).
- **method**: which solver was used. `l2_eig` (top eigenvector of $Q$, exact),
`linf_power` (phase-projected power iteration), or `l1_frank_wolfe` (vertex-greedy).
        """
    )


# ============================================================
# Network plot (click-to-flip)
# ============================================================
st.subheader("Signed network with optimized signal")

with st.expander("Reading this plot"):
    st.markdown(
        r"""
- **Node size** is proportional to the optimized magnitude $r_i^{*}$ (bigger = the
optimal signal listens more strongly through this agent).
- **Node color** encodes the optimized phase lag $\phi_i^{*} \in [0, 2\pi)$
on a cyclic HSV scale.
- **Edges**: green = positive (friendly), red = negative (antagonistic).
- **Diamond markers** sit at edge midpoints. **Click one to flip its sign**
(positive ↔ negative). Use *Reset edge flips* in the sidebar to revert.
        """
    )

_kappa_for_plot = source_frequency_centrality(B)
fig_net = plot_signed_network(
    A_plus_eff, A_minus_eff, positions, r_star, phi_star,
    communities=communities, c_vec=c_vec,
    centrality=_kappa_for_plot,
    centrality_label="kappa",
    show_centrality_labels=show_centrality_labels,
)

event = st.plotly_chart(
    fig_net,
    width='stretch',
    on_select="rerun",
    selection_mode=("points",),
    key="network_plot",
)

# ---- handle edge clicks ----
new_sel_set = set()
if event is not None:
    sel = getattr(event, "selection", None) or (event.get("selection") if isinstance(event, dict) else None)
    if sel:
        pts = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", [])
        for p in pts:
            cd = p.get("customdata") if isinstance(p, dict) else getattr(p, "customdata", None)
            if cd and len(cd) >= 2:
                # cd may be wrapped in a list of length 1 by plotly; accept both
                if isinstance(cd[0], (list, tuple)):
                    cd = cd[0]
                try:
                    i_e, j_e = int(cd[0]), int(cd[1])
                    new_sel_set.add(frozenset((i_e, j_e)))
                except (TypeError, ValueError):
                    pass

last_sel_set = st.session_state.get("last_sel_set", set())
new_clicks = new_sel_set - last_sel_set
if new_clicks:
    flipped = st.session_state.setdefault("flipped_edges", set())
    for e in new_clicks:
        if e in flipped:
            flipped.remove(e)
        else:
            flipped.add(e)
    st.session_state["last_sel_set"] = new_sel_set
    st.rerun()
else:
    st.session_state["last_sel_set"] = new_sel_set

if flipped_edges:
    pretty = ", ".join(f"{tuple(sorted(e))}" for e in sorted(flipped_edges, key=lambda s: tuple(sorted(s))))
    st.caption(f"**Edge flips active:** {pretty}")


# ============================================================
# Time + magnitude + trajectories + phases
# ============================================================
c1, c2 = st.columns(2)
with c1:
    if objective == "norm":
        label = "||x_t||^2"
    elif objective == "variance":
        label = "Var(x_t)"
    else:
        label = "(v . x_t)^2"
    st.plotly_chart(
        plot_functional_over_time(np.arange(T_plot), V_t, label=label),
        width='stretch',
    )
with c2:
    st.plotly_chart(plot_magnitude_bar(r_star, sort=True), width='stretch')

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(
        plot_agent_trajectories(np.arange(T_plot), X, max_lines=20),
        width='stretch',
    )
with c4:
    st.plotly_chart(
        plot_phase_histogram(phi_star, r_star=r_star),
        width='stretch',
    )

with st.expander("What do the time/agent/phase plots show?"):
    st.markdown(
        r"""
- **Functional over time**: $V_t(a^*) = \|x^{\mathrm{ss}}_t\|^2$ (norm objective)
or $\mathrm{Var}(x^{\mathrm{ss}}_t)$ (variance). For a single sinusoidal forcing
the response is also sinusoidal, so $V_t$ is itself a sinusoid in $t$ at twice
the frequency $\omega$.
- **Agent trajectories**: each line is one agent's $x_i^{\mathrm{ss}}(t)$ over a few periods.
Peaks/lags reveal who oscillates with whom.
- **Phase histogram**: distribution of optimized $\phi_i^*$ on the unit circle.
A bimodal distribution (two opposing clumps) is a fingerprint of a structurally
balanced bipartition.
- **Magnitude bar**: $r_i^*$ sorted descending. Under $\ell_1$ this is sparse
(Frank–Wolfe places mass on a single vertex for PSD $Q$); under $\ell_\infty$ it is flat at $R$.
        """
    )


# ============================================================
# Centrality analysis
# ============================================================
st.subheader("Frequency-specific centrality")

with st.expander("What is frequency-specific centrality?"):
    st.markdown(
        r"""
From thesis **Definition 6.25** (specialized to the spec's $B_\omega$ convention,
$|H| = |B|$):

- **Source frequency centrality** $\kappa_i^{(\beta)}(\omega) = \sum_j |B_{ji}|$
(column 1-norm of $|B|$): total amplitude of the network-wide oscillatory
response generated by a unit signal injected through agent $i$.
- **Receiver frequency centrality** $r_j^{(\beta)}(\omega) = \sum_i |B_{ji}|$
(row 1-norm of $|B|$): total amplitude received at agent $j$ when each agent receives
unit signal mass.

The static signed Bonacich centralities of thesis §5 are the $\omega = 0$ specialization.
The plots below cross-reference $r_i^*$ (where the optimizer chose to put listening mass)
against these structural centralities and against degree.
        """
    )

kappa = source_frequency_centrality(B)
recv = receiver_frequency_centrality(B)
d_plus = A_plus_eff.sum(axis=1)
d_minus = A_minus_eff.sum(axis=1)

# Pearson correlations
def _corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

r1, r2, r3, r4 = st.columns(4)
r1.metric("corr(r*, kappa)", f"{_corr(r_star, kappa):.3f}")
r2.metric("corr(r*, receiver)", f"{_corr(r_star, recv):.3f}")
r3.metric("corr(r*, deg+)", f"{_corr(r_star, d_plus):.3f}")
r4.metric("corr(r*, deg-)", f"{_corr(r_star, d_minus):.3f}")

cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(
        plot_centrality_scatter(
            kappa, r_star,
            x_label="source centrality kappa_i(omega)",
            y_label="optimized r_i*",
            title="Optimal listening allocation vs. source centrality",
            color_values=phi_star,
            color_label="phi_i*",
        ),
        width='stretch',
    )
with cc2:
    st.plotly_chart(
        plot_centrality_scatter(
            recv, r_star,
            x_label="receiver centrality r_j(omega)",
            y_label="optimized r_i*",
            title="Optimal listening allocation vs. receiver centrality",
            color_values=phi_star,
            color_label="phi_i*",
        ),
        width='stretch',
    )

cc3, cc4 = st.columns(2)
with cc3:
    st.plotly_chart(
        plot_centrality_scatter(
            d_plus, r_star,
            x_label="positive degree deg+_i",
            y_label="optimized r_i*",
            title="r_i* vs. positive (friendly) degree",
        ),
        width='stretch',
    )
with cc4:
    st.plotly_chart(
        plot_centrality_scatter(
            d_minus, r_star,
            x_label="negative degree deg-_i",
            y_label="optimized r_i*",
            title="r_i* vs. negative (antagonistic) degree",
        ),
        width='stretch',
    )

cc5, cc6 = st.columns(2)
with cc5:
    st.plotly_chart(plot_centrality_bars(kappa, label="source centrality kappa_i"),
                    width='stretch')
with cc6:
    st.plotly_chart(plot_centrality_bars(recv, label="receiver centrality r_j"),
                    width='stretch')

# Eigenvector centrality (unsigned underlying graph) scatter
eig_cent = eigenvector_centrality_unsigned(A_plus_eff, A_minus_eff)

st.markdown("**Frequency-specific vs. eigenvector centrality.**")
with st.expander("What does this comparison show?"):
    st.markdown(
        r"""
We compare each agent's *frequency-specific* centrality — derived from
$B_\omega = (e^{i\omega} I - W)^{-1} D$ and so signed-aware — to their
*standard eigenvector centrality* on the unsigned underlying graph (top
Perron eigenvector of $|A| = A^+ + A^-$).

The eigenvector centrality is a *topology-only* quantity (it ignores signs,
$\alpha$, $\beta$, $c$, and $\omega$). Differences between the two reveal
how much the antagonism structure and the signal frequency reshape the
"who is structurally important" question relative to the unsigned baseline.
        """
    )

cc7, cc8 = st.columns(2)
with cc7:
    st.plotly_chart(
        plot_centrality_scatter(
            eig_cent, kappa,
            x_label="eigenvector centrality (unsigned)",
            y_label="source freq centrality kappa_i(omega)",
            title="Source freq centrality vs. eigenvector centrality",
            color_values=phi_star,
            color_label="phi_i*",
        ),
        width='stretch',
    )
with cc8:
    st.plotly_chart(
        plot_centrality_scatter(
            eig_cent, recv,
            x_label="eigenvector centrality (unsigned)",
            y_label="receiver freq centrality r_j(omega)",
            title="Receiver freq centrality vs. eigenvector centrality",
            color_values=phi_star,
            color_label="phi_i*",
        ),
        width='stretch',
    )

r5, r6 = st.columns(2)
r5.metric("corr(kappa, eig)", f"{_corr(kappa, eig_cent):.3f}")
r6.metric("corr(receiver, eig)", f"{_corr(recv, eig_cent):.3f}")

with st.expander("Network details"):
    st.write(f"n = {n_total}")
    if communities is not None:
        u, ct = np.unique(communities, return_counts=True)
        st.write(f"communities: sizes = {dict(zip(map(int, u), map(int, ct)))}")
    st.write(f"|E+| = {int(A_plus_eff.sum() // 2)}, |E-| = {int(A_minus_eff.sum() // 2)}")
    st.write(f"flipped edges (vs. original template): {len(flipped_edges)}")
