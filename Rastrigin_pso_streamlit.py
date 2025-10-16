# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:53:21 2025

@author: siva1
"""

# app.py
# Streamlit app: Interactive PSO on Rastrigin with a 3D animated GIF output

import io
import time
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation

# -----------------------------
# Objective (Rastrigin, 2D -> 3D surface)
# -----------------------------
def rastrigin(X):
    # X: (N, 2)
    return 20 + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)

# -----------------------------
# PSO + Animation
# -----------------------------
def acor_gif_continuous_fast(
    objective="Rastrigin",   # "Rastrigin" or "Ackley"
    max_iters=70,
    bounds=(-5.12, 5.12),
    grid_res=48,
    alpha=0.25,
    fps=12,
    orbit_speed=2,
    seed=42,
    # ACOR (lean settings)
    k=18,
    m=18,
    q=0.25,
    xi=0.85,
    use_global_sigma=True,
    tail_len=0,
):
    import io, tempfile, os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rng = np.random.default_rng(seed)

    # ----- Objectives -----
    def rastrigin_func(X):
        return 20 + np.sum(X**2 - 10*np.cos(2*np.pi*X), axis=1)

    def ackley_func(X):
        # X shape: (N,2)
        x, y = X[:, 0], X[:, 1]
        a, b, c = 20.0, 0.2, 2*np.pi
        term1 = -a * np.exp(-b * np.sqrt(0.5 * (x*x + y*y)))
        term2 = -np.exp(0.5 * (np.cos(c*x) + np.cos(c*y)))
        return term1 + term2 + a + np.e

    def grid_ackley(Xg, Yg):
        a, b, c = 20.0, 0.2, 2*np.pi
        r2 = Xg**2 + Yg**2
        term1 = -a * np.exp(-b * np.sqrt(0.5 * r2))
        term2 = -np.exp(0.5 * (np.cos(c*Xg) + np.cos(c*Yg)))
        return term1 + term2 + a + np.e

    if objective == "Ackley":
        f_obj = ackley_func
        grid_fn = grid_ackley
        obj_label = "Ackley"
    else:
        f_obj = rastrigin_func
        grid_fn = lambda Xg, Yg: 20 + (Xg**2 - 10*np.cos(2*np.pi*Xg)) + (Yg**2 - 10*np.cos(2*np.pi*Yg))
        obj_label = "Rastrigin"

    lbv = np.array([bounds[0], bounds[0]])
    ubv = np.array([bounds[1], bounds[1]])

    def clamp_to_bounds(x):
        return np.minimum(ubv, np.maximum(lbv, x))

    # ----- Grid / figure -----
    xs = np.linspace(lbv[0], ubv[0], grid_res)
    ys = np.linspace(lbv[1], ubv[1], grid_res)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = grid_fn(Xg, Yg)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    ax3d.set_xlim(lbv[0], ubv[0]); ax3d.set_ylim(lbv[1], ubv[1])
    ax3d.set_zlim(0, float(np.percentile(Zg, 99)))
    stride = max(1, grid_res // 20)
    ax3d.plot_surface(
        Xg, Yg, Zg, rstride=stride, cstride=stride,
        linewidth=0.1, alpha=alpha, cmap="viridis",
        edgecolor="none", antialiased=False
    )

    ax2d.contourf(Xg, Yg, Zg, levels=24, cmap="viridis", alpha=alpha)
    ax2d.set_xlim(lbv[0], ubv[0]); ax2d.set_ylim(lbv[1], ubv[1])
    ax2d.set_aspect("equal", adjustable="box")

    # ----- Init archive -----
    A = lbv + rng.random((k, 2)) * (ubv - lbv)
    fA = f_obj(A)
    order = np.argsort(fA); A, fA = A[order], fA[order]
    best_x, best_f = A[0].copy(), fA[0]

    # Artists
    scat3d_A = ax3d.scatter(A[:,0], A[:,1], fA, s=16, c="black")
    scat3d_S = ax3d.scatter([], [], [], s=8,  c="gray", alpha=0.5)
    (best3d,) = ax3d.plot([best_x[0]],[best_x[1]],[best_f], marker="*", ms=9, color="red")
    ax3d.plot([0],[0],[0], marker="o", ms=5, color="blue")

    scat2d_A = ax2d.scatter(A[:,0], A[:,1], s=14, c="black")
    scat2d_S = ax2d.scatter([], [], s=10, c="gray", alpha=0.5)
    best2d = ax2d.scatter([best_x[0]],[best_x[1]], s=36, c="red", marker="D")
    ax2d.scatter([0],[0], s=28, c="blue", marker="o")

    if tail_len > 0:
        lines3d = [ax3d.plot([], [], [], lw=1, color="gray", alpha=0.4)[0] for _ in range(k)]
        lines2d = [ax2d.plot([], [], lw=1, color="gray", alpha=0.4)[0] for _ in range(k)]
        hist_x = [[A[i,0]] for i in range(k)]
        hist_y = [[A[i,1]] for i in range(k)]
        hist_z = [[fA[i]]   for i in range(k)]
    else:
        lines3d = lines2d = hist_x = hist_y = hist_z = None

    # Rank weights
    ranks = np.arange(k)
    sigma_rank = max(1e-9, 0.25 * k)
    w = np.exp(-(ranks**2) / (2.0 * (sigma_rank**2))); w /= w.sum()

    def set_view(frame):
        ax3d.view_init(elev=30, azim=(frame * orbit_speed) % 360)

    def update(frame):
        nonlocal A, fA, best_x, best_f

        if use_global_sigma:
            sigma_global = xi * (np.std(A, axis=0) + 1e-9)
            idxs = rng.choice(k, size=m, p=w)
            means = A[idxs]
            stds  = np.broadcast_to(sigma_global, means.shape)
        else:
            diffs = np.abs(A[:, None, :] - A[None, :, :])
            sigmas = xi * (np.sum(diffs, axis=1) / max(1, k-1))
            idxs = rng.choice(k, size=m, p=w)
            means = A[idxs]
            stds  = sigmas[idxs] + 1e-9

        S = rng.normal(loc=means, scale=stds)
        S = clamp_to_bounds(S)
        fS = f_obj(S)

        A_new = np.vstack([A, S])
        f_new = np.concatenate([fA, fS])
        order = np.argsort(f_new)
        A, fA = A_new[order][:k], f_new[order][:k]

        if fA[0] < best_f:
            best_f = fA[0]; best_x = A[0].copy()

        scat3d_A._offsets3d = (A[:,0], A[:,1], fA)
        scat3d_S._offsets3d = (S[:,0], S[:,1], fS)
        best3d.set_data_3d([best_x[0]], [best_x[1]], [best_f])

        scat2d_A.set_offsets(A)
        scat2d_S.set_offsets(S)
        best2d.set_offsets([[best_x[0], best_x[1]]])

        fig.suptitle(f"ACOR (fast) on {obj_label} — 3D + 2D | Iter {frame} | Best f: {best_f:.4f}")
        set_view(frame)
        return (scat3d_A, scat3d_S, best3d, scat2d_A, scat2d_S, best2d)

    ani = animation.FuncAnimation(fig, update, frames=max_iters, interval=60, blit=False)

    tmp_path = None
    try:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_path = tmp.name
        ani.save(tmp_path, writer=writer)
        plt.close(fig)
        with open(tmp_path, "rb") as f:
            data = f.read()
        return io.BytesIO(data)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass

# -----------------------------
# Streamlit UI
# -----------------------------
# --- Sidebar + run (FAST ACOR) ---
# --- Sidebar + run (FAST ACOR, multi-objective) ---
st.set_page_config(page_title="ACOR Animator (Rastrigin/Ackley)", page_icon="🐜", layout="centered")
st.title("🐜 ACOR (Fast) — 3D + 2D on Rastrigin / Ackley")
st.write("Choose a function and generate an animated GIF. Global minimum for both is at (0, 0).")

with st.sidebar:
    st.header("Objective & Rendering")
    objective = st.selectbox("Objective function", ["Rastrigin", "Ackley"])

    # Typical bounds per function
    typical_bounds = {"Rastrigin": (-5.12, 5.12), "Ackley": (-5.0, 5.0)}
    use_typical_bounds = st.toggle("Use typical bounds for this function", value=True)

    # Render & camera
    max_iters   = st.slider("Iterations (frames)", 10, 400, 70, step=5)
    grid_res    = st.slider("Surface resolution", 30, 150, 48, step=2)
    alpha       = st.slider("Surface transparency (alpha)", 0.0, 1.0, 0.25, step=0.05)
    fps         = st.slider("GIF FPS", 5, 30, 12, step=1)
    orbit_speed = st.slider("Camera orbit speed", 0, 10, 2, step=1)

    seed        = st.number_input("Random seed", value=42, min_value=0, step=1)

    if use_typical_bounds:
        bounds_low, bounds_high = typical_bounds[objective]
        st.caption(f"Bounds set to typical {objective} range: [{bounds_low}, {bounds_high}]")
    else:
        bounds_low  = st.number_input("Lower bound", value=-5.12, step=0.5, format="%.2f")
        bounds_high = st.number_input("Upper bound", value= 5.12, step=0.5, format="%.2f")

    st.markdown("**ACOR-specific (fast mode)**")
    k  = st.slider("Archive size (k)", 5, 200, 18, step=1)
    m  = st.slider("Samples per iter (m)", 5, 300, 18, step=1)
    q  = st.slider("Rank sharpness (q)", 0.05, 0.9, 0.25, step=0.01)
    xi = st.slider("Exploration scale (xi)", 0.1, 2.0, 0.85, step=0.05)

    use_global_sigma = st.toggle("Use global σ (O(k)) — fastest", value=True,
                                 help="If off, uses classic O(k²) per-archive sigmas.")
    tail_len = st.slider("Trail length (0 = off)", 0, 60, 0, step=1)

run = st.button("🚀 Run ACOR (Fast)")

if run:
    if bounds_high <= bounds_low:
        st.error("Upper bound must be greater than lower bound.")
    else:
        t0 = time.perf_counter()
        with st.spinner(f"Running ACOR on {objective} and rendering GIF..."):
            gif_buf = acor_gif_continuous_fast(
                objective=objective,
                max_iters=max_iters,
                bounds=(bounds_low, bounds_high),
                grid_res=grid_res,
                alpha=alpha,
                fps=fps,
                orbit_speed=orbit_speed,
                seed=seed,
                k=k, m=m, q=q, xi=xi,
                use_global_sigma=use_global_sigma,
                tail_len=tail_len,
            )
        elapsed = time.perf_counter() - t0
        fps_eff = (max_iters / elapsed) if elapsed > 0 else float("nan")

        st.success(f"Done! 🎉 Runtime: {elapsed:.2f} s  •  Effective gen speed: {fps_eff:.1f} frames/s")
        st.image(gif_buf, caption=f"ACOR (Fast) on {objective} — Animated GIF", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download GIF", data=gif_buf.getvalue(),
                               file_name=f"acor_{objective.lower()}_fast.gif", mime="image/gif")
        with col2:
            st.metric(label="Runtime (s)", value=f"{elapsed:.2f}", delta=None)

st.caption("Tip: For speed, keep grid_res ≈ 40–50, FPS ≈ 10–12, and enable global σ. Rastrigin bounds: ±5.12; Ackley bounds: ±5.0.")
