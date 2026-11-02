"""Figures for the week 3 D1 deck and notes. Writes PNGs into ../fig/.

Run from this folder:  python make_figures.py      (matplotlib only; ~3 s)

  fig/gd_steps.png        gradient descent on f = z1^2 + 10 z2^2 for three fixed step sizes, and Newton's one step
  fig/armijo.png          the 1-D line-search function along -grad f at z0 with the Armijo line and the backtracking trials
  fig/ik_iterates.png     left: the inverse-kinematics cost over joint space with the Gauss-Newton and gradient-descent paths;
                          right: residuals for Gauss-Newton, fixed damping, ratio-based adaptive LM and gradient descent
  fig/stationary_straight.png  the stretched arm at q = (0, 0) with the reachable target (0.5, 0) on its own line: the gradient is
                          zero, the gradient and Gauss-Newton updates are zero, and bending still reaches the target (notes section 5 (p))
  fig/lbfgs_rosenbrock.png slide 33: three accepted L-BFGS updates (m = 3) from the Classic start with the explorer's exact line
                          search: the latest move, the negative gradients at its ends, the gradient change in a side panel, and
                          the L-BFGS direction beside the negative gradient (history_directions.py; notes section 3.5)
  fig/cg_rosenbrock.png    slide 34: three accepted CG updates from the same start; the previous step on the landscape and,
                          beside it, -g_new, beta d_previous tip to tail and their sum, the next nonlinear-CG direction
  fig/bfgs_rosenbrock.png  the optional reference for slide 32's demo: the explorer's BFGS after its first (strong-Wolfe) update
  fig/lbfgs_direction.png, fig/cg_direction.png  the earlier quadratic example (optional reference in notes section 3.5)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from descent import A, Z0, f, grad, gd_fixed, gd_armijo
from gauss_newton_ik import L1, L2, phi, jac, run_gn, run_lm, cost
import history_directions as hd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig")
os.makedirs(OUT, exist_ok=True)
MAROON, TEAL, GOLD, GREY = "#9b1b4b", "#1f6f8b", "#c9a400", "#888888"
HIST = "#5b3fa6"           # the history (path, moves, previous direction) on the Rosenbrock slides: not gray, so it stands apart from the contours


def contours(ax):
    x = np.linspace(-3.5, 3.5, 300); y = np.linspace(-2.5, 2.5, 300)
    X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, X**2 + 10*Y**2, levels=[0.5, 2, 5, 10, 20, 35, 49, 70], colors="#bbb", linewidths=0.8)
    ax.plot(0, 0, "k*", ms=9)
    ax.set_aspect("equal"); ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.5, 2.5)


def fig_gd():
    # A 2 x 2 grid with large labels, because the slide shows this figure alone and the room reads it
    # from the back (readability repair, 2026-10-06). The four panels and their numbers are unchanged.
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6))
    axes = axes.ravel()
    for ax, alpha, col in zip(axes[:3], (0.05, 0.08, 0.10), (TEAL, GOLD, MAROON)):
        contours(ax)
        p = gd_fixed(Z0, alpha, 12)
        ax.plot(p[:, 0], p[:, 1], "-o", color=col, ms=4.5, lw=1.8)
        fac = 1 - alpha * np.array([2.0, 20.0])
        ax.set_title(f"gradient descent, $\\alpha = {alpha}$: factors ({fac[0]:.2f}, {fac[1]:.2f})", fontsize=15)
    ax = axes[3]; contours(ax)
    zN = Z0 - np.linalg.solve(2*A, grad(Z0))
    ax.annotate("", xy=zN, xytext=Z0, arrowprops=dict(arrowstyle="->", color="k", lw=2.5))
    ax.plot(*Z0, "o", color="k", ms=5)
    ax.set_title("Newton: $z - H^{-1}\\nabla f$, one step, exactly", fontsize=15)
    for k, ax in enumerate(axes):
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(15); ax.yaxis.label.set_size(15)
        if k % 2: ax.set_ylabel("")
        if k < 2: ax.set_xlabel("")
    # the function, start and lesson are the slide's caption, so the panels get the whole height
    fig.tight_layout(pad=0.6); fig.savefig(os.path.join(OUT, "gd_steps.png"), dpi=170); plt.close(fig)


def fig_armijo():
    d = -grad(Z0); slope = float(grad(Z0) @ d); c = 0.01
    al = np.linspace(0, 0.13, 400)
    ph = np.array([f(Z0 + a*d) for a in al])
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(al, ph, color=TEAL, lw=2, label=r"$f(z^{(0)} + \alpha d)$")
    ax.plot(al, f(Z0) + al*slope, "--", color=GREY, lw=1.2, label="tangent: $f(z^{(0)}) + \\alpha\\,\\nabla f^\\top d$")
    ax.plot(al, f(Z0) + c*al*slope, "-.", color=MAROON, lw=1.5, label=f"Armijo line, $c = {c}$")
    trials = [1.0, 0.5, 0.25, 0.125, 0.0625]
    for a in trials:
        if a <= al[-1]:
            ok = f(Z0 + a*d) <= f(Z0) + c*a*slope
            ax.plot(a, f(Z0 + a*d), "o", color=("green" if ok else "red"), ms=7)
            ax.annotate(f"{a}", (a, f(Z0 + a*d)), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.text(0.075, 104, "trials 1, 0.5, 0.25 are off the chart\n(f = 14449, 3240, 642): rejected", fontsize=8.5, ha="center", color="red")
    ax.set_xlabel(r"step size $\alpha$"); ax.set_ylabel(r"$f(z^{(0)} + \alpha d)$"); ax.set_ylim(0, 120); ax.set_xlim(0, 0.135)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, color="#eee")
    ax.set_title(r"Halving from $\alpha = 1$: 0.125 rejected, 0.0625 accepted", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "armijo.png"), dpi=170); plt.close(fig)


def draw_arm(ax, q, color, alpha=1.0, lw=4):
    p1 = np.array([L1*np.cos(q[0]), L1*np.sin(q[0])]); p2 = phi(q)
    ax.plot([0, p1[0]], [0, p1[1]], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def fig_ik():
    """Left: the arm's cost 1/2|phi(q) - y*|^2 over joint space, with its two minima, the saddle between them,
    and the Gauss-Newton and gradient-descent paths from q0. Right: residuals for five update choices."""
    q0 = np.deg2rad([30.0, 45.0]); y = np.array([0.5, 0.2])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # left: the landscape in joint space
    g1 = np.linspace(-180, 180, 361); g2 = np.linspace(-180, 180, 361)
    Q1, Q2 = np.meshgrid(np.deg2rad(g1), np.deg2rad(g2))
    F = 0.5*((L1*np.cos(Q1) + L2*np.cos(Q1+Q2) - y[0])**2 + (L1*np.sin(Q1) + L2*np.sin(Q1+Q2) - y[1])**2)
    cs = ax1.contour(g1, g2, F, levels=[0.002, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6], colors="#bbb", linewidths=0.8)
    ax1.clabel(cs, levels=[0.01, 0.1, 0.3], fontsize=7, fmt="%.2f")
    hist, _ = run_gn(q0, y, rho=0.0)
    P = np.degrees(np.array([h["q"] for h in hist]))
    ax1.plot(P[:, 0], P[:, 1], "-o", color=TEAL, ms=4, lw=1.6, label="Gauss-Newton (5 steps)")
    q, a, G = q0.copy(), 1.0, [q0.copy()]
    for _ in range(46):
        g = jac(q).T @ (phi(q) - y); d = -g
        while cost(q + a*d, y) > cost(q, y) + 0.01*a*float(g @ d): a *= 0.5
        q = q + a*d; a = 1.2*a; G.append(q.copy())
    G = np.degrees(np.array(G)); ax1.plot(G[:, 0], G[:, 1], "-", color=GREY, lw=1.4, label="gradient descent + Armijo (46 steps)")
    ax1.plot(*np.degrees(q0), "o", color="k", ms=5); ax1.text(33, 40, "$q^{(0)}$", fontsize=9)
    for qm, lab, dx, dy in ((np.degrees(hist[-1]["q"]), "elbow-up", 4, 6), ((55.12, -80.41), "elbow-down", 4, 6)):
        ax1.plot(qm[0], qm[1], "*", color=GOLD, ms=14, mec="k"); ax1.text(qm[0]+dx, qm[1]+dy, lab, fontsize=8)
    ax1.plot(21.80, 0.0, "x", color=MAROON, ms=9, mew=2); ax1.text(26, -12, "saddle (21.8°, 0°)", fontsize=8, color=MAROON)
    ax1.set_xlabel("$q_1$ [deg]"); ax1.set_ylabel("$q_2$ [deg]"); ax1.set_xlim(-180, 180); ax1.set_ylim(-180, 180); ax1.set_aspect("equal")
    ax1.set_xticks(range(-180, 181, 60)); ax1.set_yticks(range(-180, 181, 60)); ax1.grid(True, color="#eee"); ax1.legend(fontsize=8, loc="lower left")
    ax1.set_title("$f(q) = \\frac{1}{2}\\|\\varphi(q) - y^*\\|^2$ over joint space, $y^* = (0.5, 0.2)$: two valleys, one saddle", fontsize=9.5)
    # right: residual curves
    for rho, col, lab in ((0.0, TEAL, "Gauss-Newton ($\\rho = 0$)"), (0.05, GOLD, "fixed $\\rho = 0.05$"), (0.5, MAROON, "fixed $\\rho = 0.5$")):
        h, _ = run_gn(q0, y, rho=rho, iters=60)
        ax2.semilogy([e["k"] for e in h], [max(e["res"], 1e-16) for e in h], "-o", color=col, ms=3, label=lab)
    lm_hist, _, _ = run_lm(q0, y, rho0=0.05)
    ax2.semilogy([e["k"] for e in lm_hist], [e["res"] for e in lm_hist],
                 "--D", color="#6f4c9b", ms=4, lw=1.7,
                 label="adaptive LM ($\\rho$ starts at 0.05)")
    q, a, res = q0.copy(), 1.0, []
    for k in range(61):
        res.append(np.linalg.norm(phi(q) - y))
        if res[-1] < 1e-6:
            break
        g = jac(q).T @ (phi(q) - y); d = -g
        while cost(q + a*d, y) > cost(q, y) + 0.01*a*float(g @ d): a *= 0.5
        q = q + a*d; a = 1.2*a
    ax2.semilogy(range(len(res)), res, "-", color=GREY, label="gradient descent + Armijo")
    ax2.axhline(1e-6, color=GREY, lw=0.8, ls=":")
    ax2.text(31, 1.5e-6, "task tolerance", color="#666", fontsize=8)
    ax2.set_xlabel("accepted update"); ax2.set_ylabel(r"$\|\varphi(q) - y^*\|$ [m]"); ax2.grid(True, color="#eee", which="both"); ax2.legend(fontsize=8.5)
    ax2.set_ylim(1e-13, 1); ax2.set_title("Residuals: fixed damping versus adaptive LM", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ik_iterates.png"), dpi=170); plt.close(fig)



def fig_stationary():
    """The reachable stationary point of notes section 5 (p): the straight arm, the target on its line, the
    collapsed velocity ellipse (a vertical segment), and the elbow-bent solution the target actually has."""
    q_s = np.zeros(2); y = np.array([0.5, 0.0]); p2 = phi(q_s); J = jac(q_s)
    q_sol = np.array([-np.arctan2(L2, L1), np.pi/2])                       # (-36.87 deg, 90 deg): phi = (0.5, 0)
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    draw_arm(ax, q_sol, "#bbbbbb", lw=4)
    draw_arm(ax, q_s, "#444", lw=6)
    ax.plot(0, 0, "o", color="white", mec="#444", mew=2, ms=8); ax.plot(L1, 0, "o", color="white", mec="#444", mew=2, ms=8)
    ax.plot(*p2, "o", color=TEAL, ms=8)
    t = np.linspace(0, 2*np.pi, 200); E = 0.35 * (J @ np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(p2[0] + E[0], p2[1] + E[1], color=TEAL, lw=2.5)
    ax.plot(*y, "*", color=GOLD, ms=17, mec="k", mew=0.8, zorder=6)
    ax.annotate("", xy=p2, xytext=y, arrowprops=dict(arrowstyle="->", color=MAROON, lw=2.2, shrinkA=6, shrinkB=4), zorder=7)   # r = phi - y*: from y* to the tip
    ax.text(0.25, -0.07, r"$\mathbf{r} = \varphi(\mathbf{q}) - \mathbf{y}^* = (0.2, 0)$", color=MAROON, fontsize=9.5, ha="center")
    ax.text(0.73, -0.19, "velocity ellipse: a\nvertical segment\n(both columns of $J$\npoint across the arm)", color=TEAL, fontsize=8.5, va="center")
    ax.text(0.5, 0.05, r"$\mathbf{y}^* = (0.5, 0)$, reachable", color="#7a6400", fontsize=9.5, ha="center")
    ax.text(0.02, 0.33, "ghost: the solution $(-36.9°, 90°)$\nthat bending the elbow reaches", color="#777", fontsize=8.5)
    ax.text(0.02, -0.33, r"at $\mathbf{q} = (0, 0)$: $\nabla f = \mathbf{J}^\top\mathbf{r} = \mathbf{0}$", fontsize=9.5)
    ax.set_xlim(-0.05, 0.95); ax.set_ylim(-0.38, 0.45); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True, color="#eee")
    ax.set_title("Reachable target, zero gradient: stationary, not solved, not unreachable", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stationary_straight.png"), dpi=170); plt.close(fig)


def _history_axes(title):
    """One panel of the optional quadratic reference (notes section 3.5): the elongated bowl, a gold star at the minimizer."""
    fig, ax = plt.subplots(figsize=(5.8, 4.35))
    x = np.linspace(-3.4, 0.8, 400); y = np.linspace(-1.6, 1.6, 400)
    X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, X**2 + 10*Y**2, levels=[0.25, 1, 2.5, 5, 9, 14, 20, 28], colors="#bbb", linewidths=0.9)
    ax.plot(0, 0, "*", color=GOLD, ms=19, mec="k", mew=0.8, zorder=6)
    ax.text(0.08, -0.22, "minimum", color="#7a6400", fontsize=12, ha="center", va="top")
    ax.set_aspect("equal"); ax.set_xlim(-3.4, 0.8); ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("$z_1$", fontsize=15); ax.set_ylabel("$z_2$", fontsize=15); ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=15)
    return fig, ax


def _arrow(ax, start, vec, color, lw=2.6, ls="-", alpha=1.0, z=5):
    ax.annotate("", xy=np.asarray(start) + np.asarray(vec), xytext=start, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls, alpha=alpha, shrinkA=0, shrinkB=0, mutation_scale=16))


def fig_history():
    """The earlier quadratic example, kept as optional reference: f = z1^2 + 10 z2^2 from (-2.5, 0.8) after one exact
    line-minimized step; left the one-pair L-BFGS direction, right the CG vector addition at one scale."""
    q = hd.quadratic_example()
    z0 = np.array([-2.5, 0.8]); z1, g1 = q["z1"], q["g1"]; g0 = 2 * np.diag([1.0, 10.0]) @ z0; d0 = -g0
    unit = lambda v: v / np.linalg.norm(v)
    fig, ax = _history_axes("L-BFGS: one stored move and gradient change")
    _arrow(ax, z0, z1 - z0, GREY, lw=3.0)
    ax.text(-3.3, 0.30, r"previous move $\mathbf{s}$", color="#666", fontsize=12.5)
    side = 0.09 * unit(np.array([-(z1 - z0)[1], (z1 - z0)[0]]))
    _arrow(ax, z0 + side, np.linalg.norm(z1 - z0) * unit(-g0), MAROON, lw=1.8, ls="--", alpha=0.4)
    ax.text(-2.08, 0.72, r"$-\mathbf{g}_{\mathrm{old}}$", color=MAROON, alpha=0.6, fontsize=12.5)
    _arrow(ax, z1, 1.5 * unit(-g1), MAROON)
    ax.text(-0.82, 0.55, r"$-\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=14)
    _arrow(ax, z1, 1.5 * unit(q["d_lbfgs"]), TEAL, lw=3.0)
    ax.text(-1.2, -0.36, "L-BFGS direction", color=TEAL, fontsize=14, ha="center")
    ax.plot(*z0, "o", color="#666", ms=6, zorder=6); ax.plot(*z1, "o", color="k", ms=7, zorder=7)
    ax.text(-2.5, 0.95, r"$\mathbf{z}^{(0)}$", fontsize=12.5, ha="center"); ax.text(-2.26, -0.33, r"$\mathbf{z}^{(1)}$", fontsize=12.5, ha="center")
    ax.text(-3.3, -0.72, "The gradient changed after this move.", fontsize=13, color="#333", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "lbfgs_direction.png"), dpi=170); plt.close(fig)
    k = 0.35; mg, bd = k * (-g1), k * q["beta"] * d0; d = k * q["d_cg"]
    fig, ax = _history_axes("nonlinear CG: add a scaled previous direction")
    _arrow(ax, z0, z1 - z0, GREY, lw=1.8, ls="--", alpha=0.45)
    ax.text(-2.38, 0.86, "previous move", color="#888", fontsize=11.5)
    _arrow(ax, z1, mg, MAROON); ax.text(-1.75, 0.45, r"$-\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=14)
    _arrow(ax, z1 + mg, bd, GREY, lw=3.0); ax.text(-0.55, 0.30, r"$\beta\,\mathbf{d}_{\mathrm{previous}}$", color="#666", fontsize=14)
    _arrow(ax, z1, d, TEAL, lw=3.2, z=4); ax.text(-1.1, -0.36, "next direction", color=TEAL, fontsize=14, ha="center")
    ax.plot(*z0, "o", color="#888", ms=5, zorder=6); ax.plot(*z1, "o", color="k", ms=7, zorder=7)
    ax.text(-2.26, -0.33, r"$\mathbf{z}^{(1)}$", fontsize=12.5, ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cg_direction.png"), dpi=170); plt.close(fig)


ROSEN_LEVELS = [0.3, 1, 3, 10, 30, 100, 300, 1000]
LOCAL_LEVELS = [3.3, 3.5, 3.75, 4, 4.5, 5, 6, 8, 11, 15, 20, 30, 45]
NUM = ["①", "②", "③", "④"]


def _rosen_overview(ax, pts, box):
    """The whole valley: contours at log-spaced levels, the path so far, the minimizer and the local view's box."""
    x = np.linspace(-2, 2, 500); y = np.linspace(-1, 3, 500); X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, (1 - X) ** 2 + 100 * (Y - X**2) ** 2, levels=ROSEN_LEVELS, colors="#bbb", linewidths=0.8)
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=HIST, ms=3.5, lw=1.6)
    ax.plot(*pts[-1], "o", color="k", ms=5, zorder=6)
    ax.plot(1, 1, "*", color=GOLD, ms=15, mec="k", mew=0.7, zorder=6)
    ax.text(1.0, 0.72, "minimum", color="#7a6400", fontsize=10.5, ha="center", va="top")
    (x0, x1), (y0, y1) = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#333", lw=1.2, ls="--"))
    ax.text(x0 - 0.05, y1 + 0.08, "local view", fontsize=10.5, color="#333")
    ax.set_xlim(-2, 2); ax.set_ylim(-1, 3); ax.set_aspect("equal")
    ax.set_xlabel("$z_1$", fontsize=12); ax.set_ylabel("$z_2$", fontsize=12); ax.tick_params(labelsize=10)
    ax.set_title("the whole valley", fontsize=11.5)


def _rosen_local(ax, box, title):
    (x0, x1), (y0, y1) = box
    x = np.linspace(x0, x1, 500); y = np.linspace(y0, y1, 500); X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, (1 - X) ** 2 + 100 * (Y - X**2) ** 2, levels=LOCAL_LEVELS, colors="#bbb", linewidths=0.9)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("$z_1$", fontsize=13); ax.set_ylabel("$z_2$", fontsize=13); ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=13.5)


def _shared_axes(fig):
    """The shared layout of slides 33 and 34: a small overview, a large local view and a side panel beside it."""
    axo = fig.add_axes([0.035, 0.27, 0.20, 0.50])
    ax = fig.add_axes([0.285, 0.11, 0.43, 0.80])
    side = fig.add_axes([0.765, 0.15, 0.22, 0.66])
    return axo, ax, side


def fig_rosenbrock_history():
    """Slides 33 and 34 use three accepted updates from the Classic start with the explorer's exact line search (the
    L-BFGS and CG traces coincide there, which history_directions.py checks). The optional BFGS figure for slide 32's
    demo uses the explorer's BFGS after its first strong-Wolfe update."""
    lb, cg = hd.trace_lbfgs(), hd.trace_cg()
    pts = lb["points"]; z2, z3 = pts[-2], pts[-1]
    g2, g3 = hd.grad(z2), hd.grad(z3)
    unit = lambda v: v / np.linalg.norm(v)
    box = ((-1.27, -0.30), (0.33, 1.13))
    # --- slide 33: L-BFGS with three pairs: overview | local view | the gradients at one origin -----------------------
    fig = plt.figure(figsize=(13.4, 4.6)); axo, ax, side = _shared_axes(fig)
    _rosen_overview(axo, pts, box)
    _rosen_local(ax, box, "after three accepted updates from the Classic start")
    ax.plot(pts[:3, 0], pts[:3, 1], "-o", color=HIST, alpha=0.45, ms=4, lw=1.4)                 # the recent path
    _arrow(ax, z2, z3 - z2, HIST, lw=3.4)                                                      # ① the latest move
    ax.text(-0.905, 0.72, NUM[0] + r" latest move $\mathbf{s}$", color=HIST, fontsize=12.5)
    kg = 0.012                                                                                 # ② negative gradients, one scale
    _arrow(ax, z2, kg * (-g2), MAROON, lw=2.0, alpha=0.55)
    ax.text(-0.72, 0.905, NUM[1] + r" $-\mathbf{g}_{\mathrm{old}}$", color=MAROON, alpha=0.75, fontsize=12.5)
    _arrow(ax, z3, kg * (-g3), MAROON, lw=2.6)
    ax.text(-0.53, 0.71, NUM[1] + r" $-\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=12.5)
    ax.text(-1.25, 0.37, "negative gradients cross the contours\ntoward lower cost (one scale for both)", fontsize=10.5, color=MAROON)
    L = np.linalg.norm(kg * g3)                                                                # ④ the L-BFGS direction, same length
    _arrow(ax, z3, L * unit(lb["d_next"]), TEAL, lw=3.2, z=6)
    ax.text(-0.62, 0.48, NUM[3] + " L-BFGS direction\n(drawn as long as $-\\mathbf{g}_{\\mathrm{new}}$)", color=TEAL, fontsize=11.5)
    ax.plot(*z2, "o", color=HIST, ms=6, zorder=6); ax.plot(*z3, "o", color="k", ms=7, zorder=7)
    ax.text(z2[0] - 0.03, z2[1] + 0.03, r"$\mathbf{z}_{\mathrm{old}}$", fontsize=12, ha="right")
    ax.text(z3[0] - 0.03, z3[1] - 0.06, r"$\mathbf{z}_{\mathrm{new}}$", fontsize=12, ha="right")
    y = g3 - g2; ki = 0.012                                                                    # ③ the gradients at one origin
    _arrow(side, (0, 0), ki * g2, MAROON, lw=2.0, alpha=0.55); _arrow(side, (0, 0), ki * g3, MAROON, lw=2.6)
    _arrow(side, ki * g2, ki * y, "#222", lw=2.6)
    side.text(-0.12, 0.012, r"$\mathbf{g}_{\mathrm{old}}$", color=MAROON, alpha=0.8, fontsize=12.5, ha="center", va="bottom")
    side.text(-0.30, -0.16, r"$\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=12.5, ha="center", va="top")
    side.text(-0.19, -0.105, r"$\mathbf{y}$", color="#222", fontsize=14, ha="left", va="top")
    side.text(-0.35, -0.265, r"$\mathbf{y}=\mathbf{g}_{\mathrm{new}}-\mathbf{g}_{\mathrm{old}}$" + ", the gradient change", color="#333", fontsize=10.5, va="bottom")
    side.set_title(NUM[2] + " the two gradients at one origin", fontsize=12)
    side.set_xlim(-0.36, 0.08); side.set_ylim(-0.28, 0.07); side.set_aspect("equal"); side.set_xticks([]); side.set_yticks([])
    side.axhline(0, color="#ddd", lw=0.8); side.axvline(0, color="#ddd", lw=0.8)
    fig.savefig(os.path.join(OUT, "lbfgs_rosenbrock.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    # --- slide 34: nonlinear CG: overview | local view | the tip-to-tail construction beside it --------------------
    fig = plt.figure(figsize=(13.4, 4.6)); axo, ax, side = _shared_axes(fig)
    _rosen_overview(axo, pts, box)
    _rosen_local(ax, box, "after three accepted updates from the Classic start")
    ax.plot(pts[:3, 0], pts[:3, 1], "-o", color=HIST, alpha=0.45, ms=4, lw=1.4)
    _arrow(ax, z2, z3 - z2, HIST, lw=3.4)                                                      # ① the previous step
    ax.text(-1.25, 0.42, NUM[0] + r" previous step $\alpha\,\mathbf{d}_{\mathrm{previous}}$" + "\n(its length came from the line search)", color=HIST, fontsize=11.5)
    kg3 = 0.012
    _arrow(ax, z3, kg3 * (-g3), MAROON, lw=2.6)                                                # ② -g_new on the landscape
    ax.text(-0.53, 0.71, NUM[1] + r" $-\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=12.5)
    ax.plot(*z2, "o", color=HIST, ms=6, zorder=6); ax.plot(*z3, "o", color="k", ms=7, zorder=7)
    ax.text(z2[0] - 0.03, z2[1] + 0.03, r"$\mathbf{z}_{\mathrm{old}}$", fontsize=12, ha="right")
    ax.text(z3[0] - 0.03, z3[1] - 0.06, r"$\mathbf{z}_{\mathrm{new}}$", fontsize=12, ha="right")
    kv = 0.004                                                                                 # the side panel: one scale for all
    mg, bd = (kv * c for c in cg["parts"]); d = kv * cg["d_next"]; dp = kv * cg["d_prev"]
    _arrow(side, (0, 0), dp, HIST, lw=1.6, ls="--", alpha=0.55)                                # the previous direction itself
    side.text(dp[0] * 0.55 + 0.03, dp[1] * 0.55, r"$\mathbf{d}_{\mathrm{previous}}$", color=HIST, alpha=0.8, fontsize=11.5)
    _arrow(side, (0, 0), mg, MAROON, lw=2.6)                                                   # ② -g_new
    side.text(mg[0] + 0.015, mg[1] + 0.035, NUM[1] + r" $-\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=12.5, ha="left")
    _arrow(side, mg, bd, HIST, lw=3.0)                                                         # ③ beta d_previous, tip to tail
    side.text(mg[0] + bd[0] * 0.5 + 0.025, mg[1] + bd[1] * 0.5, NUM[2] + r" $\beta\,\mathbf{d}_{\mathrm{previous}}$", color=HIST, fontsize=12.5)
    _arrow(side, (0, 0), d, TEAL, lw=3.4, z=6)                                                  # ④ the sum
    side.text(d[0] * 0.45 - 0.03, d[1] * 0.45 - 0.02, NUM[3] + r" $\mathbf{d}_{\mathrm{new}}$", color=TEAL, fontsize=13, ha="right", va="top")
    side.plot(0, 0, "o", color="k", ms=6, zorder=7)
    side.text(-0.015, -0.04, r"$\mathbf{z}_{\mathrm{new}}$", fontsize=11, ha="right", va="top")
    side.set_title("tip to tail at one scale", fontsize=12)
    side.text(-0.09, -0.80, "the previous step was α d_previous,\nwith α = 0.00156 from the line search", fontsize=9, color=HIST, va="bottom")
    side.set_xlim(-0.10, 0.48); side.set_ylim(-0.81, 0.13); side.set_aspect("equal"); side.set_xticks([]); side.set_yticks([])
    fig.savefig(os.path.join(OUT, "cg_rosenbrock.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    # --- optional reference (slide 32's demo): the explorer's BFGS after its first accepted update --------------------
    bf = hd.trace_bfgs(1); z0, z1 = bf["points"][0], bf["points"][1]
    g0, g1 = hd.grad(z0), hd.grad(z1); s1, y1 = bf["pairs"][-1]
    fig = plt.figure(figsize=(13.4, 4.6)); axo, ax, side = _shared_axes(fig)
    side.set_position([0.735, 0.15, 0.25, 0.66])                                              # a wider side panel for the wide y
    box1 = ((1.2, 1.8), (1.8, 2.4))
    x = np.linspace(-2, 2, 500); yy = np.linspace(-1, 3, 500); X, Y = np.meshgrid(x, yy)
    axo.contour(X, Y, (1 - X) ** 2 + 100 * (Y - X**2) ** 2, levels=ROSEN_LEVELS, colors="#bbb", linewidths=0.8)
    _arrow(axo, z0, z1 - z0, HIST, lw=3.0)                                                     # ① the first move: the jump
    axo.text(-0.15, 1.95, NUM[0] + r" move $\mathbf{s}$", color=HIST, fontsize=11.5, ha="right")
    kg = 0.004                                                                                 # ② negative gradients, one scale
    _arrow(axo, z0, kg * (-g0), MAROON, lw=2.0, alpha=0.6); _arrow(axo, z1, kg * (-g1), MAROON, lw=2.4)
    axo.text(-1.15, 0.55, NUM[1] + r" $-\mathbf{g}_{\mathrm{old}}$", color=MAROON, alpha=0.8, fontsize=11)
    axo.text(1.12, 2.3, NUM[1] + r" $-\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=11, ha="right")
    axo.plot(*z0, "o", color=HIST, ms=5, zorder=6); axo.plot(*z1, "o", color="k", ms=5, zorder=7)
    axo.plot(1, 1, "*", color=GOLD, ms=15, mec="k", mew=0.7, zorder=6); axo.text(1.0, 0.72, "minimum", color="#7a6400", fontsize=10.5, ha="center", va="top")
    (bx0, bx1), (by0, by1) = box1
    axo.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec="#333", lw=1.2, ls="--"))
    axo.text(bx1 + 0.05, by0 - 0.05, "local\nview", fontsize=10, color="#333", va="top")
    axo.set_xlim(-2, 2); axo.set_ylim(-1, 3); axo.set_aspect("equal")
    axo.set_xlabel("$z_1$", fontsize=12); axo.set_ylabel("$z_2$", fontsize=12); axo.tick_params(labelsize=10)
    axo.set_title("the first accepted update: a jump", fontsize=11.5)
    x = np.linspace(bx0, bx1, 500); yy = np.linspace(by0, by1, 500); X, Y = np.meshgrid(x, yy)
    ax.contour(X, Y, (1 - X) ** 2 + 100 * (Y - X**2) ** 2, levels=[0.3, 0.7, 1.2, 2.07, 3, 5, 8, 12, 20, 35, 60], colors="#bbb", linewidths=0.9)
    ax.set_xlim(bx0, bx1); ax.set_ylim(by0, by1); ax.set_aspect("equal")
    ax.set_xlabel("$z_1$", fontsize=13); ax.set_ylabel("$z_2$", fontsize=13); ax.tick_params(labelsize=11)
    ax.set_title("at the new point, after that one update", fontsize=13.5)
    t_in = (bx0 - z0[0]) / (z1 - z0)[0]                                                           # the move arrives from the left
    _arrow(ax, z0 + t_in * (z1 - z0), (1 - t_in) * (z1 - z0), HIST, lw=3.4)
    ax.text(1.215, 1.93, NUM[0] + r" the move $\mathbf{s}$ arrives", color=HIST, fontsize=12, rotation=22, rotation_mode="anchor")
    L = 0.26
    _arrow(ax, z1, L * unit(-g1), MAROON, lw=2.6)                                              # ② -g_new
    ax.text(1.255, 2.235, NUM[1] + r" $-\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=12.5)
    _arrow(ax, z1, L * unit(bf["d_next"]), TEAL, lw=3.2, z=6)                                  # ④ the BFGS direction
    ax.text(1.515, 2.30, NUM[3] + " BFGS direction\n(drawn as long as $-\\mathbf{g}_{\\mathrm{new}}$)", color=TEAL, fontsize=11.5)
    ax.text(1.22, 1.82, "the negative gradient points straight at the floor;\nthe turned direction approaches it more gently", fontsize=10.5, color="#333")
    ax.plot(*z1, "o", color="k", ms=7, zorder=7); ax.text(z1[0] + 0.02, z1[1] - 0.035, r"$\mathbf{z}_{\mathrm{new}}$", fontsize=12)
    ki = 0.004; yv = g1 - g0                                                                   # ③ the gradients at one origin
    _arrow(side, (0, 0), ki * g0, MAROON, lw=2.0, alpha=0.6); _arrow(side, (0, 0), ki * g1, MAROON, lw=2.6)
    _arrow(side, ki * g0, ki * yv, "#222", lw=2.6)
    side.text(-0.66, -0.20, r"$\mathbf{g}_{\mathrm{old}}$", color=MAROON, alpha=0.85, fontsize=12.5, ha="center", va="bottom")
    side.text(0.36, -0.07, r"$\mathbf{g}_{\mathrm{new}}$", color=MAROON, fontsize=12.5, ha="left", va="top")
    side.text(-0.27, -0.27, r"$\mathbf{y}$", color="#222", fontsize=14, ha="center", va="top")
    side.text(-0.98, -0.56, r"$\mathbf{y}=\mathbf{g}_{\mathrm{new}}-\mathbf{g}_{\mathrm{old}}$" + ", the gradient change", color="#333", fontsize=10.5, va="bottom")
    side.set_title(NUM[2] + " the two gradients at one origin", fontsize=12)
    side.set_xlim(-1.0, 0.62); side.set_ylim(-0.6, 0.2); side.set_aspect("equal"); side.set_xticks([]); side.set_yticks([])
    side.axhline(0, color="#ddd", lw=0.8); side.axvline(0, color="#ddd", lw=0.8)
    fig.savefig(os.path.join(OUT, "bfgs_rosenbrock.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    cross = lb["d_next"][0] * cg["d_next"][1] - lb["d_next"][1] * cg["d_next"][0]
    print(f"traces coincide: {np.allclose(lb['points'], cg['points'], atol=1e-9)}; proposed directions parallel to "
          f"{abs(cross) / np.linalg.norm(lb['d_next']) / np.linalg.norm(cg['d_next']):.1e} (sine of the angle)")


if __name__ == "__main__":
    fig_gd(); fig_armijo(); fig_ik(); fig_stationary(); fig_history(); fig_rosenbrock_history()
    print("wrote gd_steps, armijo, ik_iterates, stationary_straight, lbfgs_direction, cg_direction, bfgs_rosenbrock, cg_rosenbrock, lbfgs_rosenbrock (.png) to", os.path.abspath(OUT))
