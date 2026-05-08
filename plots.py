"""Plotly figures for the oscillatory optimizer."""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.graph_objects as go


def _phase_to_color(phi: float) -> str:
    """Convert phase in [0, 2pi) to an HSV-based hex color."""
    h = (phi % (2 * np.pi)) / (2 * np.pi)
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"


def plot_signed_network(
    A_plus: np.ndarray,
    A_minus: np.ndarray,
    positions: np.ndarray,
    r_star: np.ndarray,
    phi_star: np.ndarray,
    communities=None,
    c_vec=None,
    centrality=None,
    centrality_label: str = "kappa",
    show_centrality_labels: bool = False,
    node_size_min: float = 16.0,
    node_size_max: float = 60.0,
    title: str = "Signed network with optimized signal (click an edge midpoint to flip its sign)",
):
    n = A_plus.shape[0]
    pos = np.asarray(positions)

    # ---- edge line traces (positive green, negative red) ----
    edge_traces = []
    for sign, A, color, name in [
        ("+", A_plus, "rgba(60,140,80,0.65)", "positive"),
        ("-", A_minus, "rgba(200,60,60,0.65)", "negative"),
    ]:
        x_edges, y_edges = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > 0:
                    x_edges += [pos[i, 0], pos[j, 0], None]
                    y_edges += [pos[i, 1], pos[j, 1], None]
        edge_traces.append(
            go.Scatter(
                x=x_edges,
                y=y_edges,
                mode="lines",
                line=dict(color=color, width=2.0),
                hoverinfo="none",
                name=f"{name} edges",
                showlegend=True,
            )
        )

    # ---- edge midpoint markers (clickable, customdata = [i, j, sign_str]) ----
    mid_x, mid_y, mid_color, mid_custom, mid_hover = [], [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A_plus[i, j] > 0:
                sgn = "+"
                col = "rgba(60,140,80,0.9)"
            elif A_minus[i, j] > 0:
                sgn = "-"
                col = "rgba(200,60,60,0.9)"
            else:
                continue
            mid_x.append(0.5 * (pos[i, 0] + pos[j, 0]))
            mid_y.append(0.5 * (pos[i, 1] + pos[j, 1]))
            mid_color.append(col)
            mid_custom.append([i, j, sgn])
            mid_hover.append(f"edge ({i},{j}) [{sgn}] — click to flip")

    edge_mid_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(
            size=11,
            color=mid_color,
            symbol="diamond",
            line=dict(color="black", width=1.0),
        ),
        customdata=mid_custom,
        hovertext=mid_hover,
        hoverinfo="text",
        name="edge midpoints",
        showlegend=True,
    )

    # ---- node trace ----
    rmax = max(r_star.max(), 1e-12)
    sizes = node_size_min + (node_size_max - node_size_min) * (r_star / rmax)
    # Inactive nodes (r ~= 0) get a neutral grey instead of HSV(phase=0):
    # their phase is mathematically undefined.
    R_THRESHOLD = 1e-6
    colors = []
    for i in range(n):
        if r_star[i] < R_THRESHOLD:
            colors.append("rgb(180,180,180)")  # neutral grey for inactive
        else:
            colors.append(_phase_to_color(phi_star[i]))
    hover = []
    for i in range(n):
        lines = [
            f"<b>node {i}</b>",
            f"r*: {r_star[i]:.4f}",
        ]
        if r_star[i] < R_THRESHOLD:
            lines.append("phi*: undefined (r ~ 0)")
        else:
            lines.append(
                f"phi*: {phi_star[i]:.4f} rad ({np.degrees(phi_star[i]):.2f} deg)"
            )
        if communities is not None:
            lines.append(f"community: {int(communities[i])}")
        if c_vec is not None:
            lines.append(f"c_i: {c_vec[i]:.4f}")
        d_plus = int(A_plus[i].sum())
        d_minus = int(A_minus[i].sum())
        lines.append(f"deg+: {d_plus}, deg-: {d_minus}")
        if centrality is not None:
            lines.append(f"{centrality_label}: {centrality[i]:.4f}")
        hover.append("<br>".join(lines))

    node_trace = go.Scatter(
        x=pos[:, 0],
        y=pos[:, 1],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color="black", width=1.4),
        ),
        text=[str(i) for i in range(n)],
        textposition="middle center",
        textfont=dict(size=12, color="white"),
        hoverinfo="text",
        hovertext=hover,
        name="agents",
        showlegend=False,
    )

    extra_traces = []
    if show_centrality_labels and centrality is not None:
        y_extent = float(pos[:, 1].max() - pos[:, 1].min()) if n > 1 else 1.0
        offset = 0.08 * max(y_extent, 1.0) + 0.18
        extra_traces.append(
            go.Scatter(
                x=pos[:, 0],
                y=pos[:, 1] - offset,
                mode="text",
                text=[f"{centrality_label}={centrality[i]:.2f}" for i in range(n)],
                textfont=dict(size=10, color="rgba(20,20,20,0.85)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=edge_traces + [edge_mid_trace, node_trace] + extra_traces)
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=560,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        clickmode="event+select",
    )
    return fig


def plot_functional_over_time(t_grid, V_t, label="V_t"):
    fig = go.Figure(
        data=[go.Scatter(x=t_grid, y=V_t, mode="lines", line=dict(color="#3f5fff"))]
    )
    fig.update_layout(
        title=f"{label} over time (steady state)",
        xaxis_title="t",
        yaxis_title=label,
        height=300,
        margin=dict(l=40, r=10, t=40, b=40),
    )
    return fig


def plot_agent_trajectories(t_grid, X, max_lines: int = 25):
    n = X.shape[1]
    fig = go.Figure()
    show = list(range(min(n, max_lines)))
    for i in show:
        fig.add_trace(
            go.Scatter(x=t_grid, y=X[:, i], mode="lines", name=f"agent {i}", opacity=0.7)
        )
    fig.update_layout(
        title="Agent opinion trajectories (steady state)",
        xaxis_title="t",
        yaxis_title="x_i(t)",
        height=320,
        margin=dict(l=40, r=10, t=40, b=40),
    )
    return fig


def _cluster_phases_circular(phi: np.ndarray, tol_rad: float):
    """Group phases (mod 2 pi) into clusters whose consecutive members lie
    within `tol_rad` of each other on the circle. Returns a list of
    (representative_phase_in_[0, 2 pi), member_indices_in_phi)."""
    if len(phi) == 0:
        return []
    phi_mod = np.mod(np.asarray(phi), 2 * np.pi)
    order = np.argsort(phi_mod)
    sorted_phi = phi_mod[order]

    # Linear clustering on the sorted axis
    groups = [[order[0]]]
    for k in range(1, len(sorted_phi)):
        if sorted_phi[k] - sorted_phi[k - 1] <= tol_rad:
            groups[-1].append(order[k])
        else:
            groups.append([order[k]])

    # Wrap-around: merge first and last if they are close on the circle
    if len(groups) > 1:
        last_max = phi_mod[groups[-1][-1]]
        first_min = phi_mod[groups[0][0]]
        if (2 * np.pi - last_max) + first_min <= tol_rad:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    out = []
    for g in groups:
        vals = phi_mod[g]
        # If this group spans the wrap, unwrap members near 2 pi to negative
        if vals.max() - vals.min() > np.pi:
            vals = np.where(vals > np.pi, vals - 2 * np.pi, vals)
        rep = float(np.mean(vals)) % (2 * np.pi)
        out.append((rep, list(g)))
    return out


def plot_phase_histogram(
    phi_star,
    tol_deg: float = 0.5,
    r_star=None,
    r_threshold: float = 1e-6,
):
    """Vector view of optimized phases: cluster phases that coincide within
    `tol_deg` and draw one vector per cluster at the cluster's representative
    angle; line thickness scales with cluster size.

    Using a small tolerance (default 0.5 deg) keeps drawn vectors at
    essentially the exact node phases — so the hover-tooltip phase on a node
    matches the angle of its corresponding vector here.

    Nodes with magnitude below `r_threshold` are excluded: their phase is
    mathematically undefined (e.g. l1 vertex optima leave most r_i = 0, and
    np.angle(0) returns 0, which would otherwise display as a spurious
    zero-phase vector).
    """
    fig = go.Figure()
    phi_star = np.asarray(phi_star)
    if r_star is not None:
        r_star = np.asarray(r_star)
        mask = r_star >= r_threshold
        phi_active = phi_star[mask]
        n_excluded = int((~mask).sum())
    else:
        phi_active = phi_star
        n_excluded = 0

    if len(phi_active) == 0:
        fig.update_layout(title="Optimized phase lags (no active nodes)")
        return fig

    tol_rad = np.deg2rad(tol_deg)
    clusters = _cluster_phases_circular(phi_active, tol_rad)
    max_count = max((len(g) for _, g in clusters), default=1)

    for rep, members in clusters:
        count = len(members)
        deg = np.degrees(rep)
        width = 2.0 + 8.0 * (count / max_count)
        color = _phase_to_color(rep)
        member_str = ", ".join(str(int(m)) for m in members[:8])
        if count > 8:
            member_str += f", … (+{count - 8})"
        # shaft
        fig.add_trace(
            go.Scatterpolar(
                r=[0.0, 1.0],
                theta=[deg, deg],
                mode="lines",
                line=dict(color=color, width=width),
                hovertemplate=(
                    f"phase = {deg:.2f} deg<br>"
                    f"count = {count}<br>"
                    f"nodes: {member_str}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        # tip
        fig.add_trace(
            go.Scatterpolar(
                r=[1.0],
                theta=[deg],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=6 + 6 * (count / max_count),
                    color=color,
                    line=dict(color="black", width=0.5),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    title = (
        f"Optimized phase lags (vectors at exact node phases; "
        f"thickness = #nodes within {tol_deg:g}°)"
    )
    if n_excluded > 0:
        title += f"<br><sub>{n_excluded} node(s) excluded: r_i below {r_threshold:.0e} (phase undefined)</sub>"
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks="", range=[0, 1.15]),
            angularaxis=dict(direction="counterclockwise"),
        ),
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    return fig


def plot_magnitude_bar(r_star, sort: bool = True):
    n = len(r_star)
    idx = np.argsort(-r_star) if sort else np.arange(n)
    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(i) for i in idx],
                y=r_star[idx],
                marker_color="#5f7fff",
            )
        ]
    )
    fig.update_layout(
        title="Optimized magnitudes r_i*" + (" (sorted)" if sort else ""),
        xaxis_title="agent",
        yaxis_title="r_i*",
        height=300,
        margin=dict(l=40, r=10, t=40, b=40),
    )
    return fig


def plot_centrality_scatter(
    x_values,
    y_values,
    x_label: str,
    y_label: str,
    title: str,
    labels=None,
    color_values=None,
    color_label: str = "",
):
    """Scatter with node-index labels and optional color gradient."""
    n = len(x_values)
    if labels is None:
        labels = [str(i) for i in range(n)]
    marker = dict(size=12, line=dict(color="black", width=1))
    if color_values is not None:
        marker["color"] = color_values
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
        marker["colorbar"] = dict(title=color_label)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(size=10),
                marker=marker,
                hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=340,
        margin=dict(l=40, r=10, t=40, b=40),
    )
    return fig


def plot_centrality_bars(centrality, label: str = "centrality"):
    """Bar chart of a centrality vector, sorted descending."""
    n = len(centrality)
    idx = np.argsort(-centrality)
    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(i) for i in idx],
                y=centrality[idx],
                marker_color="#7f5fbf",
            )
        ]
    )
    fig.update_layout(
        title=f"{label} (sorted)",
        xaxis_title="agent",
        yaxis_title=label,
        height=300,
        margin=dict(l=40, r=10, t=40, b=40),
    )
    return fig
