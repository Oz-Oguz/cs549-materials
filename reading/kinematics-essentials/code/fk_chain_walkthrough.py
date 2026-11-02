"""Numerical construction of the fixed two-link pose, one factor at a time.

The four factors are R(q1), T(l1, 0), R(q2), T(l2, 0), with q=(30,45) degrees
and lengths (0.4,0.3) metres. X^(s) labels the cumulative construction stage,
not a robot link or time. All figures show the same fixed robot configuration.
Physical frames are W, 1 (base after q1), 2 (elbow after q2), and E (tip).

Run: python fk_chain_walkthrough.py             # numerical checks; writes no figures
     python fk_chain_walkthrough.py --figures   # also writes fig/fk-chain-*.png
"""
import argparse
from pathlib import Path

import numpy as np

from two_link import fk, fk_transforms, jacobian, jacobian_fd


LENGTHS = (0.4, 0.3)
ANGLES = np.deg2rad([30.0, 45.0])
OUT = Path(__file__).resolve().parent.parent / "fig"
MAROON, TEAL, GREY = "#9b1b4b", "#1f6f8b", "#777777"


def rotation_block(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def pose2(theta=0.0, translation=(0.0, 0.0)):
    out = np.eye(3)
    out[:2, :2] = rotation_block(theta)
    out[:2, 2] = translation
    return out


def embed3(X):
    """Insert the physical z coordinate before the homogeneous coordinate."""
    out = np.eye(4)
    out[:2, :2] = X[:2, :2]
    out[:2, 3] = X[:2, 2]
    return out


def walkthrough():
    q1, q2 = ANGLES
    l1, l2 = LENGTHS
    names = ("R(30 deg)", "T(0.4, 0)", "R(45 deg)", "T(0.3, 0)")
    factors = (pose2(q1), pose2(translation=(l1, 0.0)),
               pose2(q2), pose2(translation=(l2, 0.0)))
    stages = [np.eye(3)]
    for factor in factors:
        stages.append(stages[-1] @ factor)
    coordinates = [np.array([0.0, 0.0, 1.0])]
    for factor in reversed(factors):
        coordinates.append(factor @ coordinates[-1])
    return names, factors, stages, coordinates


def check_and_print():
    names, factors, stages, coordinates = walkthrough()
    np.set_printoptions(precision=6, suppress=True)
    expected_origins = np.array([[0, 0], [0, 0], [.3464101615137755, .2],
                                 [.3464101615137755, .2], [.4240558750445317, .4897777478867205]])
    expected_headings = [0, 30, 30, 75, 75]
    expected_coordinates = np.array([[0, 0, 1], [.3, 0, 1],
                                     [.2121320343559643, .2121320343559643, 1],
                                     [.6121320343559643, .2121320343559643, 1],
                                     [.4240558750445317, .4897777478867205, 1]])
    print("Fixed robot: q = (30, 45) deg, lengths = (0.4, 0.3) m")
    print("X^(s) is a construction stage; the robot does not move between stages.")
    for s, X in enumerate(stages):
        assert np.allclose(X[:2, 2], expected_origins[s], atol=1e-14, rtol=0)
        assert np.allclose(X[:2, :2], rotation_block(np.deg2rad(expected_headings[s])), atol=1e-14)
        print(f"\nX^({s}), origin {X[:2, 2]}, heading {expected_headings[s]} deg:\n{X}")
        if s:
            old, factor = stages[s - 1], factors[s - 1]
            p_update = old[:2, 2] + old[:2, :2] @ factor[:2, 2]
            assert np.allclose(p_update, X[:2, 2], atol=1e-14)
            assert np.allclose(old[:2, :2] @ factor[:2, :2], X[:2, :2], atol=1e-14)
            print(f"appended {names[s - 1]}:\n{factor}")
            print("origin update: old origin", old[:2, 2], "+ old rotation times", factor[:2, 2])

    print("\nThe elbow rotation block, row times column (unrounded inputs):")
    old, next_rotation = stages[2][:2, :2], factors[2][:2, :2]
    for row in range(2):
        for col in range(2):
            terms = old[row] * next_rotation[:, col]
            print(f"R[{row+1},{col+1}]: {terms[0]: .8f} + {terms[1]: .8f} = {terms.sum(): .8f}")

    # Existing closed form and separately implemented factor chain remain identical.
    old_factors, old_stages, old_final = fk_transforms(ANGLES)
    for (_, old_factor), factor in zip(old_factors, factors):
        assert np.allclose(old_factor, factor, atol=1e-14)
    for (_, old_stage), stage in zip(old_stages, stages[1:]):
        assert np.allclose(old_stage, stage, atol=1e-14)
    assert np.allclose(stages[-1], old_final, atol=1e-14)
    assert np.allclose(stages[-1][:2, 2], fk(ANGLES), atol=1e-14)
    assert np.allclose(jacobian(ANGLES), jacobian_fd(ANGLES), atol=1e-9)
    grouped = factors[0] @ (factors[1] @ factors[2]) @ factors[3]
    assert np.allclose(grouped, stages[4], atol=1e-14)
    print("\nPhysical frame grouping: W->1 is factor 1; 1->2 is factors 2,3; 2->E is factor 4.")
    print("^1 X^2 = T(0.4, 0) R(45 deg):\n", factors[1] @ factors[2])

    print("\nTip origin, with factors applied to coordinates from the right:")
    for s, point in enumerate(coordinates):
        assert np.allclose(point, expected_coordinates[s], atol=1e-14, rtol=0)
        print("start in E" if s == 0 else f"after {names[-s]}", point)
    assert np.allclose(coordinates[-1], stages[-1] @ coordinates[0], atol=1e-14)

    X3 = np.eye(4)
    for factor in factors:
        X3 = X3 @ embed3(factor)
    assert np.allclose(X3, embed3(stages[-1]), atol=1e-14)
    assert np.allclose(X3 @ [0, 0, 0, 1], [*expected_origins[-1], 0, 1], atol=1e-14)
    print("\nSame pose embedded in 3D:\n", X3)
    print("All stage, coordinate-action, frame-grouping and 3D-embedding checks passed.")


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, _, stages, _ = walkthrough()
    points = np.array([stages[0][:2, 2], stages[2][:2, 2], stages[4][:2, 2]])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})

    def axes():
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        ax.set_aspect("equal")
        ax.set_xlim(-.19, .67)
        ax.set_ylim(-.17, .73)
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.set_xticks([0, .2, .4, .6])
        ax.set_yticks([0, .2, .4, .6])
        ax.grid(color="#ececec", lw=.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        return fig, ax

    def frame(ax, X, name, muted=False, length=.12):
        p = X[:2, 2]
        for col, letter, color in [(0, "x", MAROON), (1, "y", TEAL)]:
            direction = X[:2, col]
            color = "#a9a9a9" if muted else color
            ax.annotate("", xy=p + length * direction, xytext=p,
                        arrowprops={"arrowstyle": "-|>", "lw": 1.7 if muted else 2.5,
                                    "color": color, "linestyle": "--" if muted else "-"}, zorder=6)
            ax.text(*(p + 1.24 * length * direction), f"${letter}_{{{name}}}$",
                    color=color, fontsize=13, ha="center", va="center", zorder=7)

    def arm(ax, processed=0):
        # Full arm stays in the same pose, even before its chain has been evaluated.
        ax.plot(points[:, 0], points[:, 1], color="#d6d6d6", lw=9,
                solid_capstyle="round", zorder=2)
        if processed >= 2:
            ax.plot(points[:2, 0], points[:2, 1], color="#555555", lw=7,
                    solid_capstyle="round", zorder=3)
        if processed >= 4:
            ax.plot(points[1:, 0], points[1:, 1], color="#555555", lw=7,
                    solid_capstyle="round", zorder=3)
        for p in points[:2]:
            ax.plot(*p, "o", ms=10, mfc="white", mec="#555555", mew=2, zorder=5)
        ax.plot(*points[2], "o", color="#555555", ms=8, zorder=5)

    def save(fig, name):
        fig.tight_layout(pad=1.0)
        target = OUT / name
        fig.savefig(target, dpi=180, facecolor="white")
        plt.close(fig)
        print("wrote", target)

    fig, ax = axes()
    arm(ax, 4)
    frame(ax, stages[0], "W", muted=True, length=.155)
    for s, name in ((1, "1"), (3, "2"), (4, "E")):
        frame(ax, stages[s], name)
    ax.text(-.025, -.105, "joint 1 / base", ha="center", fontsize=12)
    ax.text(.405, .155, "joint 2", ha="left", fontsize=12)
    ax.text(.505, .49, "tip E", ha="left", fontsize=12)
    ax.text(.025, .31, "$l_1=0.4$ m", fontsize=13)
    ax.text(.46, .405, "$l_2=0.3$ m", fontsize=13)
    for origin, start, stop, radius, label, label_pos in (
        (points[0], 0, ANGLES[0], .20, "$q_1=30°$", (.215, -.045)),
        (points[1], ANGLES[0], ANGLES.sum(), .13, "$q_2=45°$", (.485, .285)),
    ):
        th = np.linspace(start, stop, 50)
        ax.plot(origin[0] + radius*np.cos(th), origin[1] + radius*np.sin(th), color=MAROON, lw=1.4)
        ax.text(*label_pos, label, color=MAROON, fontsize=13)
    ax.set_title("Physical frames W, 1, 2 and E", fontsize=15, pad=12)
    save(fig, "fk-chain-overview.png")

    labels = ["W", "1", "(2)", "2", "E"]
    purposes = ["Start at the world frame", "Orient along link 1", "Reach the elbow; keep link-1 axes",
                "Orient along link 2; keep the elbow origin", "Reach the tip; keep link-2 axes"]
    for s, X in enumerate(stages):
        fig, ax = axes()
        arm(ax, s)
        if s:
            frame(ax, stages[s - 1], labels[s - 1], muted=True, length=.145)
        frame(ax, X, labels[s], length=.13)
        p = X[:2, 2]
        ax.plot(*p, "o", ms=8, color=MAROON, zorder=8)
        if s in (2, 4):
            old_p = stages[s - 1][:2, 2]
            offset = np.array([.022, -.023])
            ax.annotate("", xy=p+offset, xytext=old_p+offset,
                        arrowprops={"arrowstyle": "->", "color": MAROON, "lw": 2}, zorder=6)
        ax.text(.015, .97, "Same fixed arm at every stage", transform=ax.transAxes,
                va="top", color=GREY, fontsize=12)
        ax.text(.015, .9, f"origin ({p[0]:.4f}, {p[1]:.4f}) m\nheading {np.degrees(np.arctan2(X[1,0], X[0,0])):.0f}°",
                transform=ax.transAxes, va="top", fontsize=13)
        if s == 2:
            ax.text(.015, .04, "(2) labels a construction stage.\nPhysical frame 2 comes next.",
                    transform=ax.transAxes, fontsize=12, color=GREY)
        ax.set_title(f"Stage {s}: {purposes[s]}", fontsize=14, pad=12)
        save(fig, f"fk-chain-stage-{s}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="also regenerate the six stage/overview figures")
    args = parser.parse_args()
    check_and_print()
    if args.figures:
        make_figures()


if __name__ == "__main__":
    main()
