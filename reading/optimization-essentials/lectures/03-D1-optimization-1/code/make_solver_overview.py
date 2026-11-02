"""Plot the line-search example already used by opt-explorer's 1D panel.

Run with .venv/bin/python -B lectures/03-D1-optimization-1/code/make_solver_overview.py.
The panel uses f(x)=x^2, x=2, d=-4, c1=0.2 and c2=0.5.
The bands show the complete acceptance conditions, including Armijo.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent.parent / "fig"
C1, C2 = 0.2, 0.5
MAROON, TEAL, GOLD = "#9b1b4b", "#1f6f8b", "#c9a400"


def phi(alpha):
    return (2 - 4 * np.asarray(alpha)) ** 2


def slope(alpha):
    return -16 + 32 * np.asarray(alpha)


def accepted(alpha):
    alpha = np.asarray(alpha)
    armijo = (alpha > 0) & (phi(alpha) <= 4 + C1 * alpha * (-16) + 1e-12)
    wolfe = armijo & (slope(alpha) >= C2 * (-16) - 1e-12)
    strong = armijo & (np.abs(slope(alpha)) <= C2 * 16 + 1e-12)
    return armijo, wolfe, strong


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17})
    fig, (curve, bands) = plt.subplots(
        2, 1, figsize=(8.0, 5.1), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.2], "hspace": 0.25},
    )
    alpha = np.linspace(0, 1, 401)
    curve.plot(alpha, phi(alpha), color=MAROON, linewidth=3)
    curve.scatter([0], [4], color=MAROON, s=40, zorder=3)
    curve.annotate("Start", (0, 4), xytext=(10, -3), textcoords="offset points")
    curve.axvline(0.5, color="#aaaaaa", linestyle=":", linewidth=1)
    curve.set_ylabel("Cost along the line")
    curve.set_ylim(-0.2, 4.65)
    curve.set_yticks([0, 2, 4])
    curve.grid(axis="y", color="#eeeeee")
    for ax in (curve, bands):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.015, 1.015)
    rows = [("Armijo", 2, 0, 0.8, TEAL),
            ("Wolfe", 1, 0.25, 0.8, GOLD),
            ("Strong Wolfe", 0, 0.25, 0.75, MAROON)]
    for label, y, lo, hi, color in rows:
        bands.plot([lo, hi], [y, y], color=color, linewidth=9, solid_capstyle="butt")
        bands.scatter([hi], [y], s=38, color=color, zorder=3)
        bands.scatter([lo], [y], s=38, facecolors="white" if lo == 0 else color,
                      edgecolors=color, linewidths=1.5, zorder=4)
    bands.set_yticks([2, 1, 0], [r[0] for r in rows])
    bands.tick_params(axis="y", length=0, pad=9)
    bands.set_ylim(-0.45, 2.45)
    bands.set_xticks([0, 0.25, 0.5, 0.75, 1])
    bands.set_xlabel(r"Step size $\alpha$")
    bands.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.96, bottom=0.14)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "line_search_overview.png", dpi=180, facecolor="white")
    plt.close(fig)
    a, w, s = accepted(np.array([0.1, 0.5, 0.78]))
    assert a.tolist() == [True, True, True]
    assert w.tolist() == [False, True, True]
    assert s.tolist() == [False, True, False]
    print("Wrote line_search_overview.png; the three probe classifications pass.")


if __name__ == "__main__":
    main()
