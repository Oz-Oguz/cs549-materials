"""Figures for the kinematics reading that its two decks do not already draw. Writes PNGs into ../fig/.

Run from anywhere:  python lectures/02-reading-kinematic-foundations/code/make_figures.py   (matplotlib; ~5 s)

Every number plotted comes from the same functions that `kinematics_note.py` asserts against the
reading's frozen digits: the two-link arm from lectures/02-D2-transforms-fk-jacobian/code/two_link.py
and the closed-form IK, fixed-sign SVD and Gauss-Newton iteration of kinematics_note.py itself.

  fig/change_of_frame.png    frames A and B and a point P: the three arrows of equation (7), on the tray scene
  fig/workspace_ik.png       the reachable annulus and the three targets of §I.5: two, one and no solution
  fig/ellipse_sweep.png      the manipulability ellipse at q2 = 90°, 45°, 10°, 0°: the approach to a singularity
  fig/joint_speed.png        joint speed for a 0.1 m/s tip command against q2, J^-1 versus the damped step,
                             and the two gain curves 1/sigma and sigma/(sigma^2 + rho)
  fig/gauss_newton.png       the Gauss-Newton residual from both starts and the arm's five snapshots

The two figures the reading already had, fig/kinematic_tree.png and fig/nullspace_line.png, are
drawn by kinematics_note.py and are not repeated here.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge

_HERE = Path(__file__).resolve().parent
_TWO_LINK_DIR = _HERE.parents[1] / "02-D2-transforms-fk-jacobian" / "code"
for _d in (_HERE, _TWO_LINK_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from two_link import L1, L2, fk, jacobian                              # noqa: E402
from kinematics_note import ik_two_link, elbow, svd_fixed, gauss_newton_ik   # noqa: E402

OUT = _HERE.parent / "fig"
OUT.mkdir(exist_ok=True)
MAROON, TEAL, GREY, GOLD, INK = "#9b1b4b", "#1f6f8b", "#888888", "#c9a400", "#222222"
Q = np.deg2rad([30.0, 45.0])
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def draw_arm(ax, q, color="#555555", lw=6, alpha=1.0, tip_color=TEAL, zorder=3):
    p0, p1, p2 = np.zeros(2), elbow(q), fk(q)
    ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=zorder)
    for p in (p0, p1):
        ax.plot(*p, "o", color="white", mec=color, mew=2, ms=8, alpha=alpha, zorder=zorder + 1)
    ax.plot(*p2, "o", color=tip_color, ms=8, alpha=alpha, zorder=zorder + 1)


def draw_frame(ax, origin, theta, name, color, length=0.09, lw=1.8):
    o = np.asarray(origin)
    ex = length * np.array([np.cos(theta), np.sin(theta)])
    ey = length * np.array([-np.sin(theta), np.cos(theta)])
    for e, lab in ((ex, "x"), (ey, "y")):
        ax.annotate("", xy=o + e, xytext=o, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
        ax.text(*(o + 1.22 * e), f"${lab}_{{{name}}}$", color=color, ha="center", va="center", fontsize=10)
    ax.plot(*o, "o", color=color, ms=4)


# ---------------------------------------------------------------------------------------------------
def fig_change_of_frame(path):
    """Equation (7) on the tray scene: frame A at the origin, frame B at ^A p^B turned by 30°, point P."""
    pAB = np.array([0.40, 0.20]); th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pBP_B = np.array([0.15, 0.05])
    pAP = pAB + R @ pBP_B
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    draw_frame(ax, (0, 0), 0.0, "A", INK, length=0.10)
    draw_frame(ax, pAB, th, "B", TEAL, length=0.10)
    ax.plot(*pAP, "o", color=MAROON, ms=8, zorder=5); ax.text(pAP[0] + 0.012, pAP[1] + 0.012, "$P$", color=MAROON, fontsize=12)

    def arrow(a, b, color, label, dx=0.0, dy=0.0, ls="-"):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, ls=ls, shrinkA=0, shrinkB=0))
        m = (np.asarray(a) + np.asarray(b)) / 2
        ax.text(m[0] + dx, m[1] + dy, label, color=color, fontsize=12, ha="center", va="center")

    arrow((0, 0), pAB, TEAL, r"${}^{A}\mathbf{p}^{B}$", dx=0.02, dy=-0.035)
    arrow(pAB, pAP, GOLD, r"${}^{A}\mathbf{R}^{B}\,{}^{B}\mathbf{p}^{P}$", dx=0.075, dy=-0.02)
    arrow((0, 0), pAP, MAROON, r"${}^{A}\mathbf{p}^{P}$", dx=-0.06, dy=0.03, ls="--")
    ax.text(0.02, 0.42, r"${}^{A}\mathbf{p}^{P} = {}^{A}\mathbf{p}^{B} + {}^{A}\mathbf{R}^{B}\,{}^{B}\mathbf{p}^{P}$",
            fontsize=13, color=INK)
    ax.text(0.02, 0.375, r"the arrow from $B$ to $P$ is $(0.15,\ 0.05)$ in $B$'s axes and $(0.1049,\ 0.1183)$ in $A$'s",
            fontsize=10.5, color=GREY)
    ax.set_xlim(-0.06, 0.66); ax.set_ylim(-0.08, 0.46); ax.set_aspect("equal")
    ax.set_xlabel("$x_A$ [m]"); ax.set_ylabel("$y_A$ [m]")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig_workspace_ik(path):
    """The annulus l1 - l2 <= |p| <= l1 + l2 and the three targets of §I.5.2."""
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.add_patch(Wedge((0, 0), L1 + L2, 0, 360, width=(L1 + L2) - (L1 - L2), color="#f3e9ed", zorder=0))
    for r in (L1 - L2, L1 + L2):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=MAROON, lw=1.2, ls="--", zorder=1))
    for y in ((0.5, 0.2), (0.7, 0.0), (0.8, 0.1)):
        sols = ik_two_link(np.array(y))
        for q in sols:
            col = GREY if len(sols) == 1 else (TEAL if q[1] > 0 else GOLD)
            draw_arm(ax, q, color=col, lw=5, alpha=0.95, tip_color=MAROON)
        marker = "x" if not sols else "o"
        ax.plot(*y, marker, color=MAROON, ms=11 if marker == "x" else 8, mew=2.5, zorder=6)
    ax.text(-0.72, -0.84, r"$\mathbf{y}^* = (0.5,\ 0.2)$, inside the ring: two solutions,", fontsize=10, color=INK)
    ax.text(-0.72, -0.905, r"$q_2 = +80.41^\circ$ (teal, elbow below the base–tip line) and $q_2 = -80.41^\circ$ (gold, above)", fontsize=10, color=INK)
    ax.text(-0.72, -0.97, r"$\mathbf{y}^* = (0.7,\ 0)$, on the outer circle: one solution, the arm stretched (gray), singular", fontsize=10, color=INK)
    ax.text(-0.72, -1.035, r"$\mathbf{y}^* = (0.8,\ 0.1)$, outside the ring: no solution ($\times$)", fontsize=10, color=INK)
    ax.text(-0.72, 0.74, "reachable annulus: radii $l_1 - l_2 = 0.1$ m and $l_1 + l_2 = 0.7$ m", fontsize=10, color=MAROON)
    ax.set_xlim(-0.75, 0.95); ax.set_ylim(-1.09, 0.82); ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig_ellipse_sweep(path):
    """The manipulability ellipse at four q2, q1 fixed at 30°: it thins to a segment at q2 = 0."""
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9))
    for ax, q2 in zip(axes, (90.0, 45.0, 10.0, 0.0)):
        q = np.deg2rad([30.0, q2])
        J = jacobian(q); U, s, V = svd_fixed(J)
        p = fk(q)
        draw_arm(ax, q, lw=5)
        ang = np.rad2deg(np.arctan2(U[1, 0], U[0, 0]))
        scale = 0.45                                              # meters per (m/s): draws sigma as a length
        if s[1] > 1e-9:
            ax.add_patch(Ellipse(p, 2 * scale * s[0], 2 * scale * s[1], angle=ang, fill=True,
                                 color=MAROON, alpha=0.18, zorder=2))
        ax.add_patch(Ellipse(p, 2 * scale * s[0], 2 * scale * max(s[1], 1e-3), angle=ang, fill=False,
                             color=MAROON, lw=1.8, zorder=4))
        for i, col in ((0, MAROON), (1, GOLD)):
            if s[i] > 1e-9:
                ax.annotate("", xy=p + scale * s[i] * U[:, i], xytext=p,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax.set_title(f"$q_2 = {q2:g}^\\circ$\n$\\sigma_1 = {s[0]:.4f}$,  $\\sigma_2 = {s[1]:.4f}$", fontsize=11)
        ax.set_xlim(-0.45, 0.85); ax.set_ylim(-0.25, 0.85); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    fig.suptitle("The image of the unit circle of joint rates, $\\{\\mathbf{J}\\dot{\\mathbf{q}} : |\\dot{\\mathbf{q}}| = 1\\}$, at $q_1 = 30^\\circ$", fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig_joint_speed(path):
    """|q_dot| for x_dot = (0, 0.1) against q2, J^-1 versus damped (rho = 0.01); and the two gain curves."""
    rho = 0.01
    xdot = np.array([0.0, 0.1])
    q2s = np.concatenate([np.linspace(0.25, 10, 200), np.linspace(10, 90, 200)])
    exact, damped = [], []
    for q2 in q2s:
        J = jacobian(np.deg2rad([30.0, q2]))
        exact.append(np.linalg.norm(np.linalg.solve(J, xdot)))
        damped.append(np.linalg.norm(np.linalg.solve(J.T @ J + rho * np.eye(2), J.T @ xdot)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax1.semilogy(q2s, exact, color=MAROON, lw=2, label=r"$\dot{\mathbf{q}} = \mathbf{J}^{-1}\dot{\mathbf{x}}$")
    ax1.semilogy(q2s, damped, color=TEAL, lw=2, label=r"damped, $(\mathbf{J}^\top\mathbf{J} + \rho\mathbf{I})^{-1}\mathbf{J}^\top\dot{\mathbf{x}}$,  $\rho = 0.01$")
    for q2 in (45.0, 10.0, 1.0):
        J = jacobian(np.deg2rad([30.0, q2])); n = np.linalg.norm(np.linalg.solve(J, xdot))
        ax1.plot(q2, n, "o", color=MAROON, ms=6)
        ax1.annotate(f"{n:.2f} rad/s", (q2, n), textcoords="offset points", xytext=(8, -2), fontsize=9.5, color=MAROON)
    ax1.axhline(0.5, color=GREY, lw=1, ls=":")
    ax1.text(1.5, 0.38, r"damped bound $0.1 \cdot 1/(2\sqrt{\rho}) = 0.5$ rad/s", color=GREY, fontsize=9.5)
    ax1.set_xlabel("$q_2$ [deg]  ($q_1 = 30^\\circ$)"); ax1.set_ylabel(r"$|\dot{\mathbf{q}}|$ [rad/s] for $\dot{\mathbf{x}} = (0,\ 0.1)$ m/s")
    ax1.set_xlim(0, 90); ax1.legend(loc="upper right", fontsize=9.5, frameon=False)
    ax1.set_title("Joint speed for one tip command, approaching the stretched pose", fontsize=11)

    sig = np.linspace(0.005, 0.8, 600)
    ax2.plot(sig, 1 / sig, color=MAROON, lw=2, label=r"$1/\sigma$  (pseudo-inverse)")
    ax2.plot(sig, sig / (sig ** 2 + rho), color=TEAL, lw=2, label=r"$\sigma/(\sigma^2 + \rho)$  (damped)")
    ax2.plot(np.sqrt(rho), 1 / (2 * np.sqrt(rho)), "o", color=TEAL, ms=6)
    ax2.annotate(r"maximum $1/(2\sqrt{\rho}) = 5$ at $\sigma = \sqrt{\rho} = 0.1$", (np.sqrt(rho), 5),
                 textcoords="offset points", xytext=(12, 4), fontsize=9.5, color=TEAL)
    for s_, lab in ((0.1206, r"$\sigma_2$ at $45^\circ$"), (0.0275, r"$\sigma_2$ at $10^\circ$")):
        ax2.axvline(s_, color=GREY, lw=1, ls=":"); ax2.text(s_ + 0.008, 22, lab, fontsize=9, color=GREY, rotation=90, va="top")
    ax2.set_ylim(0, 25); ax2.set_xlim(0, 0.8)
    ax2.set_xlabel(r"singular value $\sigma$"); ax2.set_ylabel("gain along that direction [rad/s per m/s]")
    ax2.legend(loc="upper right", fontsize=9.5, frameon=False)
    ax2.set_title("What each inverse does to one singular direction", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def fig_gauss_newton(path):
    """Residual against iteration from both starts, and the arm's snapshots from (30°, 45°)."""
    y_star = np.array([0.5, 0.2])
    h1 = gauss_newton_ik(np.deg2rad([30.0, 45.0]), y_star, steps=5)
    h2 = gauss_newton_ik(np.deg2rad([60.0, -45.0]), y_star, steps=5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    for h, col, lab in ((h1, MAROON, r"from $\mathbf{q}^{(0)} = (30^\circ, 45^\circ)$"), (h2, TEAL, r"from $\mathbf{q}^{(0)} = (60^\circ, -45^\circ)$")):
        ks = [k for k, *_ in h]; res = [r for _, _, r, _ in h]      # no floor: every residual is plotted
        ax1.semilogy(ks, res, "o-", color=col, lw=2, ms=6, label=lab)
    ax1.set_xlabel("iteration $k$", fontsize=11); ax1.set_ylabel(r"$\|\varphi(\mathbf{q}^{(k)}) - \mathbf{y}^*\|$ [m]", fontsize=11)
    ax1.set_ylim(1e-15, 1); ax1.tick_params(labelsize=10); ax1.legend(frameon=False, fontsize=10.5)
    ax1.set_title("Gauss-Newton on the two-link arm:\nthe residual per step", fontsize=11)
    ax1.text(2.3, 3e-2, "each step re-linearizes;\nnear the answer the digits double", fontsize=10, color=GREY)

    n = len(h1)
    for i, (k, q, r, _) in enumerate(h1):
        a = 0.25 + 0.75 * i / (n - 1)
        draw_arm(ax2, q, color=MAROON, lw=4.5, alpha=a, tip_color=MAROON)
        p = fk(q)
        if k in (0, 1):
            ax2.text(p[0] + 0.015, p[1] + 0.015, f"$k = {k}$", fontsize=10, color=MAROON, alpha=max(a, 0.6))
    ax2.plot(*y_star, "*", color=GOLD, ms=16, zorder=8)
    ax2.text(0.47, 0.26, r"$\mathbf{y}^* = (0.5,\ 0.2)$;  $k \geq 2$ land here", color=INK, fontsize=10)
    ax2.set_xlim(-0.15, 0.8); ax2.set_ylim(-0.3, 0.6); ax2.set_aspect("equal")
    ax2.set_xlabel("$x$ [m]", fontsize=11); ax2.set_ylabel("$y$ [m]", fontsize=11); ax2.tick_params(labelsize=10)
    ax2.set_title("The arm at $k = 0, 1, \\dots, 5$ from $(30^\\circ, 45^\\circ)$;\nstep 1 overshoots, then converges", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


FIGURES = {
    "change_of_frame.png": fig_change_of_frame,
    "workspace_ik.png": fig_workspace_ik,
    "ellipse_sweep.png": fig_ellipse_sweep,
    "joint_speed.png": fig_joint_speed,
    "gauss_newton.png": fig_gauss_newton,
}

if __name__ == "__main__":
    for name, fn in FIGURES.items():
        fn(OUT / name)
        print("wrote", OUT / name)
