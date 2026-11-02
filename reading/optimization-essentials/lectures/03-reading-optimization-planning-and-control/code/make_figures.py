"""Generate Chapter 02's figures from its runnable examples (no simulation).

Run: .venv/bin/python lectures/03-reading-optimization-planning-and-control/code/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from local_step import L1, example, phi

OUT = Path(__file__).resolve().parents[1] / "fig"
INK, TEAL, MAROON, GRAY = "#333333", "#1f6f8b", "#9b1b4b", "#888888"
GOLD = "#b58d00"


def arm(ax, q, color, label, **kwargs):
    elbow = [L1*np.cos(q[0]), L1*np.sin(q[0])]
    points = np.vstack(([0, 0], elbow, phi(q)))
    ax.plot(points[:, 0], points[:, 1], "o-", color=color, lw=3,
            markersize=5, label=label, **kwargs)


def local_prediction():
    r = example()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), sharex=True, sharey=True)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-0.055, 0.635)
        ax.set_ylim(-0.235, 0.64)
        ax.set_xlabel("World x [m]", fontsize=15)
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=13)
        ax.spines[["right", "top"]].set_visible(False)
        ax.plot(*r["target"], "*", ms=17, color=GOLD, zorder=6,
                label="Target (0.50, 0.20) m")
    axes[0].set_ylabel("World y [m]", fontsize=15)
    arm(axes[0], r["q_initial"], GRAY, "Initial configuration")
    axes[0].annotate("", xy=r["p_predicted"], xytext=r["p_initial"],
                     arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5,
                                     linestyle="--", mutation_scale=18))
    axes[0].plot(*r["p_predicted"], "o", mfc="none", mec=TEAL, ms=23, mew=2)
    axes[0].text(0.02, 0.56, "Initial tip\n(0.4241, 0.4898) m", fontsize=14)
    axes[0].text(0.60, 0.36, "$\\mathbf{J}\\boldsymbol{\\delta}$",
                 color=TEAL, ha="right", fontsize=17)
    axes[0].text(0.60, 0.08, "Predicted tip = target", color=TEAL,
                 ha="right", fontsize=14)
    axes[0].set_title("1. Solve the local equation", fontsize=16, pad=13)
    arm(axes[1], r["q_initial"], GRAY, "Initial configuration", alpha=0.30)
    arm(axes[1], r["q_checked"], MAROON, "Proposed configuration")
    axes[1].plot([r["p_checked"][0], r["target"][0]],
                 [r["p_checked"][1], r["target"][1]], "--",
                 color=MAROON, lw=2)
    axes[1].annotate("Remaining error\n0.1737 m", xy=(0.425, 0.17),
                     xytext=(0.60, 0.32), ha="right", color=MAROON,
                     fontsize=14, arrowprops=dict(arrowstyle="-", color=MAROON))
    axes[1].text(0.015, 0.52, "FK-checked tip\n(0.3357, 0.1437) m", fontsize=14,
                 color=MAROON)
    axes[1].set_title("2. Evaluate forward kinematics", fontsize=16, pad=13)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(pad=1.1, w_pad=2, rect=(0, 0.055, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "local_prediction_vs_fk.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    local_prediction()
    print(f"Wrote {OUT / 'local_prediction_vs_fk.png'}")
