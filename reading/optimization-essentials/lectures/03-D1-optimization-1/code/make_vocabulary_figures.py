"""Draw the two introductory bowl figures for Deck 04.

Run from any directory:
    .venv/bin/python lectures/03-D1-optimization-1/code/make_vocabulary_figures.py

The objective, derivatives, and current point come from descent.py. This script
regenerates only bowl_landscape.png and bowl_derivatives.png in ../fig/.
The first figure shows the objective before introducing derivatives. The second
shows the uphill gradient and two coordinate slices through the same point.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import proj3d
import numpy as np

from descent import Z0, f, grad, hess


OUT = Path(__file__).resolve().parent.parent / "fig"
MAROON, TEAL, GOLD = "#9b1b4b", "#1f6f8b", "#c9a400"
INK, CONTOUR, GRID = "#333333", "#b8bec3", "#e8ecef"
DPI = 180
LEVELS = [1, 5, 10, 20, 35, 49, 70, 90]
XLIM, YLIM = (-4.0, 4.0), (-3.0, 3.0)
ARROW_SCALE = 0.0225  # Drawing scale only; this is not an optimization step.

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#888888",
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": "white",
    }
)


def mesh(xlim=XLIM, ylim=YLIM, count=181):
    """Evaluate the authoritative scalar objective on a plotting grid."""
    x = np.linspace(*xlim, count)
    y = np.linspace(*ylim, count)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = np.fromiter((f(z) for z in points), dtype=float, count=len(points))
    return X, Y, values.reshape(X.shape)


def contours(ax, label_levels=(10, 35, 70)):
    X, Y, values = mesh()
    cs = ax.contour(X, Y, values, levels=LEVELS, colors=CONTOUR, linewidths=1.0)
    ax.clabel(
        cs,
        levels=list(label_levels),
        manual=[point for level, point in [(10, (-3.0, -0.32)), (35, (3.0, -1.61)), (70, (-3.0, 2.47))] if level in label_levels],
        inline=True,
        fontsize=16,
        fmt="%g",
    )
    ax.set(xlim=XLIM, ylim=YLIM, xlabel=r"$z_1$", ylabel=r"$z_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-2, 0, 2])
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    return cs


def fig_landscape():
    fig = plt.figure(figsize=(12.8, 4.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.16, 1.0], wspace=0.07)
    surface = fig.add_subplot(grid[0, 0], projection="3d", computed_zorder=False)
    flat = fig.add_subplot(grid[0, 1])

    X, Y, values = mesh(count=101)
    cmap = LinearSegmentedColormap.from_list("bowl_teal", ["#d9edf1", TEAL])
    surface.plot_surface(
        X,
        Y,
        values,
        cmap=cmap,
        alpha=0.86,
        rcount=51,
        ccount=51,
        linewidth=0.25,
        edgecolor=(1, 1, 1, 0.18),
        antialiased=True,
        zorder=1,
    )
    surface.scatter(*Z0, f(Z0), color=MAROON, s=90, edgecolor="white", linewidth=1.4, depthshade=False, zorder=10)
    surface.scatter(0, 0, f(np.zeros(2)), color=GOLD, marker="*", s=180, edgecolor=INK, linewidth=0.7, depthshade=False, zorder=10)
    surface.set(xlim=XLIM, ylim=YLIM, zlim=(0, 110))
    surface.set_xlabel(r"$z_1$", labelpad=9)
    surface.set_ylabel(r"$z_2$", labelpad=9)
    surface.set_zlabel(r"$f(\mathbf{z})$", labelpad=8)
    surface.set_xticks([-4, 0, 4])
    surface.set_yticks([-2, 0, 2])
    surface.set_zticks([0, 49, 100])
    surface.tick_params(pad=0, labelsize=16)
    surface.view_init(elev=34, azim=-60)
    surface.set_box_aspect((8, 6, 3.8), zoom=1.13)
    for point, label, anchor, color in (
        ((*Z0, f(Z0)), r"$f(\mathbf{z}^{(0)})=49$", (0.43, 0.78), MAROON),
        ((0, 0, f(np.zeros(2))), r"$f(\mathbf{0})=0$", (0.06, 0.31), INK),
    ):
        x, y, _ = proj3d.proj_transform(*point, surface.get_proj())
        surface.annotate(
            label,
            xy=(x, y),
            xytext=anchor,
            textcoords="axes fraction",
            fontsize=18,
            color=color,
            ha="left",
            va="center",
            zorder=20,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.97),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=3, shrinkB=7),
        )
    for axis in (surface.xaxis, surface.yaxis, surface.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"]["color"] = "#dddddd"
    surface.set_title("Height gives the cost", pad=10)

    contours(flat, label_levels=(10, 35))
    flat.plot(*Z0, "o", color=MAROON, ms=9, mec="white", mew=1.2, zorder=6)
    flat.plot(0, 0, "*", color=GOLD, ms=15, mec=INK, mew=0.7, zorder=6)
    flat.annotate(
        r"$\mathbf{z}^{(0)}=(3,2)$" + "\n" + r"$f=49$",
        xy=Z0,
        xytext=(0.3, 1.65),
        fontsize=18,
        color=MAROON,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.0, color=MAROON, shrinkA=3, shrinkB=7),
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.93),
    )
    flat.annotate(
        "minimum " + r"$(0,0)$" + "\n" + r"$f=0$",
        xy=(0, 0),
        xytext=(-3.6, -1.75),
        fontsize=18,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.0, color=INK, shrinkA=3, shrinkB=7),
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.93),
    )
    flat.set_title("Each contour has one cost", pad=10)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.88, bottom=0.14)
    fig.savefig(OUT / "bowl_landscape.png", dpi=DPI)
    plt.close(fig)


def coordinate_slice(ax, coordinate, color):
    """Plot both branches, using the same axis scales for both slices."""
    values = np.linspace(-3.5, 3.5, 501)
    points = np.repeat(Z0[None, :], len(values), axis=0)
    points[:, coordinate] = values
    costs = np.fromiter((f(z) for z in points), dtype=float, count=len(points))
    current = float(Z0[coordinate])
    slope = float(grad(Z0)[coordinate])
    curvature = float(hess(Z0)[coordinate, coordinate])
    ax.plot(values, costs, color=color, lw=3.4)

    # A short tangent shows the first derivative at the marked current point.
    half_width = 0.7 if coordinate == 0 else 0.55
    tangent_x = np.linspace(current - half_width, min(current + half_width, 3.5), 2)
    tangent_y = f(Z0) + slope * (tangent_x - current)
    ax.plot(tangent_x, tangent_y, "--", color=MAROON, lw=2.3)
    ax.plot(current, f(Z0), "o", color=MAROON, ms=9, mec="white", mew=1.2, zorder=6)
    ax.axhline(f(Z0), color="#ccd1d4", ls=":", lw=1.1, zorder=0)
    ax.plot([current, current], [0, f(Z0)], color="#ccd1d4", ls=":", lw=1.1, zorder=0)
    ax.text(
        0.50,
        0.94,
        rf"slope $={slope:g}$" + "\n" + rf"curvature $={curvature:g}$",
        transform=ax.transAxes,
        fontsize=18,
        ha="center",
        va="top",
        linespacing=1.35,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.95),
    )
    ax.annotate(
        "current point",
        xy=(current, f(Z0)),
        xytext=(0, 65),
        ha="center",
        va="center",
        fontsize=16,
        color=MAROON,
        arrowprops=dict(arrowstyle="-", color=MAROON, lw=1.0, shrinkA=3, shrinkB=7),
    )
    ax.set(xlim=(-3.5, 3.5), ylim=(0, 140), xlabel=rf"$z_{coordinate + 1}$")
    ax.set_xticks([-3, 0, 3] if coordinate == 0 else [-3, 0, 2, 3])
    ax.set_yticks([0, 49, 100])
    ax.set_ylabel(r"$f$", labelpad=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(r"$f(z_1,2)=z_1^2+40$" if coordinate == 0 else r"$f(3,z_2)=9+10z_2^2$", pad=12, fontsize=18)


def fig_derivatives():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    flat, first, second = axes
    contours(flat, label_levels=(10, 35))
    flat.plot(*Z0, "o", color=MAROON, ms=9, mec="white", mew=1.2, zorder=6)
    flat.plot(0, 0, "*", color=GOLD, ms=13, mec=INK, mew=0.6, zorder=6)
    flat.annotate(
        "",
        xy=Z0 + ARROW_SCALE * grad(Z0),
        xytext=Z0,
        arrowprops=dict(arrowstyle="-|>", lw=3.0, color=MAROON, mutation_scale=19, shrinkA=0, shrinkB=0),
        zorder=7,
    )
    flat.text(
        0.035,
        0.965,
        r"$\nabla f(\mathbf{z}^{(0)})=(6,40)$",
        transform=flat.transAxes,
        va="top",
        color=MAROON,
        fontsize=18,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.95),
    )
    flat.text(0.8, 1.35, r"$\mathbf{z}^{(0)}$", color=MAROON, fontsize=18)
    flat.set_title("The gradient points uphill", pad=12, fontsize=18)
    coordinate_slice(first, 0, TEAL)
    coordinate_slice(second, 1, GOLD)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.28, wspace=0.38)
    # This label belongs to the drawing, so it stays with the figure when reused.
    fig.text(0.195, 0.055, "Arrow scaled for visibility.\nIt is not a step.", ha="center", va="center", fontsize=16, color=INK, linespacing=1.15)
    fig.text(0.687, 0.035, "Both slices use the same axis scales.", ha="center", fontsize=16, color=INK)
    fig.savefig(OUT / "bowl_derivatives.png", dpi=DPI)
    plt.close(fig)


def check_example():
    """Fail if the lecture example drifts, and verify derivatives numerically."""
    np.testing.assert_allclose(Z0, [3.0, 2.0])
    np.testing.assert_allclose(f(Z0), 49.0)
    np.testing.assert_allclose(grad(Z0), [6.0, 40.0])
    np.testing.assert_allclose(hess(Z0), [[2.0, 0.0], [0.0, 20.0]])
    origin = np.zeros_like(Z0)
    np.testing.assert_allclose(f(origin), 0.0)
    np.testing.assert_allclose(grad(origin), origin)
    assert np.linalg.eigvalsh(hess(Z0)).min() > 0
    eps = 1e-5
    basis = np.eye(2)
    numerical_gradient = np.array([(f(Z0 + eps * e) - f(Z0 - eps * e)) / (2 * eps) for e in basis])
    numerical_hessian = np.column_stack([(grad(Z0 + eps * e) - grad(Z0 - eps * e)) / (2 * eps) for e in basis])
    np.testing.assert_allclose(numerical_gradient, grad(Z0), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(numerical_hessian, hess(Z0), rtol=1e-8, atol=1e-8)
    slice_constants = np.array([f(np.array([0.0, Z0[1]])), f(np.array([Z0[0], 0.0]))])
    np.testing.assert_allclose(slice_constants, [40.0, 9.0])
    coefficients = 0.5 * np.diag(hess(Z0))
    for value in [-3.5, 0.0, 1.5, 3.5]:
        for coordinate in range(2):
            point = Z0.copy()
            point[coordinate] = value
            np.testing.assert_allclose(f(point), coefficients[coordinate] * value**2 + slice_constants[coordinate])
    endpoint = Z0 + ARROW_SCALE * grad(Z0)
    assert f(endpoint) > f(Z0)
    assert XLIM[0] < endpoint[0] < XLIM[1] and YLIM[0] < endpoint[1] < YLIM[1]
    print("current point:", Z0)
    print("cost:", f(Z0))
    print("gradient:", grad(Z0))
    print("Hessian:\n", hess(Z0))
    print("coordinate-slice slopes:", grad(Z0))
    print("coordinate-slice curvatures:", np.diag(hess(Z0)))
    print("coordinate-slice quadratic coefficients:", coefficients)
    print("coordinate-slice fixed costs:", slice_constants)
    print("minimum:", origin, "cost:", f(origin))
    print("gradient drawing scale:", ARROW_SCALE, "(visibility only)")
    print("analytic and numerical checks passed")


if __name__ == "__main__":
    check_example()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_landscape()
    fig_derivatives()
    print("wrote", OUT / "bowl_landscape.png")
    print("wrote", OUT / "bowl_derivatives.png")
