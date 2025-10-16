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
def acor_gif_rastrigin_fast(
    max_iters=70,
    bounds=(-5.12, 5.12),
    grid_res=48,
    alpha=0.25,
    fps=12,
    orbit_speed=2,
    seed=42,
    # ACOR (lean settings)
    k=18,          # smaller archive
    m=18,          # fewer samples per iteration
    q=0.25,
    xi=0.85,
    use_global_sigma=False,    # key speed-up
    tail_len=0,               # no trails (set >0 if you want)
):
    import io, tempfile, os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rng = np.random.default_rng(seed)

    def rastrigin(X):
        return 20 + np.sum(X**2 - 10*np.cos(2*np.pi*X), axis=1)

    lbv = np.array([bounds[0], bounds[0]])
    ubv = np.array([bounds[1], bounds[1]])

    def clamp_to_bounds(x):
        return np.minimum(ubv, np.maximum(lbv, x))

    # Grid (smaller & coarser surface)
    xs = np.linspace(lbv[0], ubv[0], grid_res)
    ys = np.linspace(lbv[1], ubv[1], grid_res)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = 20 + (Xg**2 - 10*np.cos(2*np.pi*Xg)) + (Yg**2 - 10*np.cos(2*np.pi*Yg))

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    # Static surface once
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

    # Init archive
    A = lbv + rng.random((k, 2)) * (ubv - lbv)
    fA = rastrigin(A)
    order = np.argsort(fA); A, fA = A[order], fA[order]
    best_x, best_f = A[0].copy(), fA[0]

    # Artists (no trails by default)
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

    # Rank weights (unchanged)
    ranks = np.arange(k)
    sigma_rank = max(1e-9, 0.25 * k)  # keep constant; q folded into sigma_rank if you like
    w = np.exp(-(ranks**2) / (2.0 * (sigma_rank**2))); w /= w.sum()

    def set_view(frame):
        ax3d.view_init(elev=30, azim=(frame * orbit_speed) % 360)

    def update(frame):
        nonlocal A, fA, best_x, best_f

        # ---- Fast sigma model
        if use_global_sigma:
            sigma_global = xi * (np.std(A, axis=0) + 1e-9)  # (2,)
            idxs = rng.choice(k, size=m, p=w)
            means = A[idxs]                                  # (m,2)
            stds  = np.broadcast_to(sigma_global, means.shape)
        else:
            # (slower) classic O(k^2) version
            diffs = np.abs(A[:, None, :] - A[None, :, :])    # (k,k,2)
            sigmas = xi * (np.sum(diffs, axis=1) / max(1, k-1))  # (k,2)
            idxs = rng.choice(k, size=m, p=w)
            means = A[idxs]
            stds  = sigmas[idxs] + 1e-9

        S = rng.normal(loc=means, scale=stds)
        S = clamp_to_bounds(S)
        fS = rastrigin(S)

        # Merge & keep best k
        A_new = np.vstack([A, S]); f_new = np.concatenate([fA, fS])
        order = np.argsort(f_new); A, fA = A_new[order][:k], f_new[order][:k]

        if fA[0] < best_f:
            best_f = fA[0]; best_x = A[0].copy()

        # Update artists
        scat3d_A._offsets3d = (A[:,0], A[:,1], fA)
        scat3d_S._offsets3d = (S[:,0], S[:,1], fS)
        best3d.set_data_3d([best_x[0]], [best_x[1]], [best_f])

        scat2d_A.set_offsets(A)
        scat2d_S.set_offsets(S)
        best2d.set_offsets([[best_x[0], best_x[1]]])

        if tail_len > 0:
            for i in range(k):
                hist_x[i].append(A[i,0]); hist_y[i].append(A[i,1]); hist_z[i].append(fA[i])
                hist_x[i] = hist_x[i][-tail_len:]; hist_y[i] = hist_y[i][-tail_len:]; hist_z[i] = hist_z[i][-tail_len:]
                lines3d[i].set_data_3d(hist_x[i], hist_y[i], hist_z[i])
                lines2d[i].set_data(hist_x[i], hist_y[i])

        fig.suptitle(f"ACOR (fast) — 3D + 2D | Iter {frame} | Best f: {best_f:.4f}")
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
st.set_page_config(page_title="ACOR Animator (Rastrigin, Fast)", page_icon="🐜", layout="centered")
st.title("🐜 ACOR (Fast) — 3D + 2D Rastrigin Animation")
st.write("Tune parameters, then click **Run ACOR (Fast)** to generate a GIF. The goal of the ants is to find the global minimum at (0,0) with f(0,0)=0.")

with st.sidebar:
    st.header("ACOR (Fast) Parameters")
    seed = st.number_input("Random seed", value=42, min_value=0, step=1)

    # Lean rendering defaults for speed
    max_iters   = st.slider("Iterations (frames)", 10, 400, 70, step=5)
    grid_res    = st.slider("Surface resolution", 30, 150, 48, step=2)
    alpha       = st.slider("Surface transparency (alpha)", 0.0, 1.0, 0.25, step=0.05)
    fps         = st.slider("GIF FPS", 5, 30, 12, step=1)
    orbit_speed = st.slider("Camera orbit speed", 0, 10, 2, step=1)

    bounds_low  = st.number_input("Lower bound", value=-5.12, step=0.5, format="%.2f")
    bounds_high = st.number_input("Upper bound", value= 5.12, step=0.5, format="%.2f")

    st.markdown("**ACOR-specific (fast mode)**")
    k  = st.slider("Archive size (k)", 5, 200, 18, step=1)
    m  = st.slider("Samples per iter (m)", 5, 300, 18, step=1)
    #q  = st.slider("Rank sharpness (q)", 0.05, 0.9, 0.25, step=0.01)
    #xi = st.slider("Exploration scale (xi)", 0.1, 2.0, 0.85, step=0.05)

    #use_global_sigma = st.toggle("Use global σ (O(k)) — fastest", value=True,
    #                             help="If off, uses classic O(k²) per-archive sigmas.")
    tail_len = st.slider("Trail length (0 = off)", 0, 60, 0, step=1)

run = st.button("🚀 Run ACOR (Fast)")

if run:
    if bounds_high <= bounds_low:
        st.error("Upper bound must be greater than lower bound.")
    else:
        t0 = time.perf_counter()
        with st.spinner("Running ACOR (fast) and rendering GIF..."):
            gif_buf = acor_gif_rastrigin_fast(
                max_iters=max_iters,
                bounds=(bounds_low, bounds_high),
                grid_res=grid_res,
                alpha=alpha,
                fps=fps,
                orbit_speed=orbit_speed,
                seed=seed,
                k=k, m=m, #q=q, xi=xi,
                use_global_sigma=False,
                tail_len=tail_len,
            )
        elapsed = time.perf_counter() - t0
        fps_eff = (max_iters / elapsed) if elapsed > 0 else float("nan")

        st.success(f"Done! 🎉 Runtime: {elapsed:.2f} s  •  Effective gen speed: {fps_eff:.1f} frames/s")
        st.image(gif_buf, caption="ACOR (Fast) on Rastrigin — Animated GIF", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download GIF", data=gif_buf.getvalue(),
                               file_name="acor_rastrigin_fast.gif", mime="image/gif")
        with col2:
            st.metric(label="Runtime (s)", value=f"{elapsed:.2f}", delta=None)


st.caption("Tip: For maximum speed, keep grid_res low (≈40–50), FPS ≈ 10–12, and enable global σ.") 

