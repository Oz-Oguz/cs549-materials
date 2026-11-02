"""Static explanatory figures for the opening of Deck 06.

Run from the repository root:
    .venv/bin/python lectures/04-D2-motion-planning/code/make_figures.py

The calculation is independent of the browser demonstration and of the KOMO
solver. The scene constants match Deck 05's code/komo_two_link.py. Here the
robot is two whole line segments, and validity requires an extra 5 mm of
obstacle clearance. There is no self-collision or dynamics model. The bounded
joint chart is [-pi, pi]^2 without periodic connections or a 60-degree limit.

The grid is a sampled illustration, not an exact configuration-space boundary.
The edge check has at most one degree of change per coordinate per subinterval;
passing it alone would not establish continuous collision freedom.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
import numpy as np


FIG = Path(__file__).resolve().parent.parent / "fig"
GREEN, BLUE, GREY, RED = "#2a7d2a", "#1f5fbf", "#777777", "#b00020"
FORBIDDEN = "#c9c9c9"
LENGTHS = np.array([0.4, 0.3])
START = np.deg2rad([30.0, 45.0])
TARGET = np.array([0.5, 0.2])
OBSTACLE_CENTER = np.array([0.49, 0.35])
OBSTACLE_RADIUS = 0.05
REQUIRED_CLEARANCE = 0.005
EDGE_RESOLUTION = np.deg2rad(1.0)
GRID_STEP_DEG = 0.5
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def positive_elbow_goal():
    """Return the fixed positive-elbow IK solution, in radians."""
    cosine = (TARGET @ TARGET - LENGTHS @ LENGTHS) / (2 * np.prod(LENGTHS))
    elbow = np.arccos(cosine)
    shoulder = np.arctan2(TARGET[1], TARGET[0]) - np.arctan2(
        LENGTHS[1] * np.sin(elbow), LENGTHS[0] + LENGTHS[1] * np.cos(elbow)
    )
    return np.array([shoulder, elbow])


def arm_vertices(q):
    """Return base, elbow and tip positions in metres; q[..., 2] is radians."""
    q = np.asarray(q, dtype=float)
    elbow = LENGTHS[0] * np.stack((np.cos(q[..., 0]), np.sin(q[..., 0])), axis=-1)
    tip = elbow + LENGTHS[1] * np.stack(
        (np.cos(q.sum(axis=-1)), np.sin(q.sum(axis=-1))), axis=-1
    )
    return np.stack((np.zeros_like(elbow), elbow, tip), axis=-2)


def point_segment_distance(point, a, b):
    """Euclidean distance to closed segments, with broadcastable input arrays."""
    direction = np.asarray(b) - a
    offset = np.asarray(point) - a
    fraction = np.clip(
        np.sum(offset * direction, axis=-1) / np.sum(direction * direction, axis=-1),
        0.0, 1.0,
    )
    return np.linalg.norm(offset - fraction[..., None] * direction, axis=-1)


def link_clearances(q):
    """Signed disc-surface clearance of each whole link, in metres."""
    vertices = arm_vertices(q)
    return point_segment_distance(
        OBSTACLE_CENTER, vertices[..., :-1, :], vertices[..., 1:, :]
    ) - OBSTACLE_RADIUS


def direct_edge_samples(a=START, b=None):
    """Return progress and configurations, including endpoints, at <=1 deg/joint."""
    if b is None:
        b = positive_elbow_goal()
    count = max(1, int(np.ceil(np.max(np.abs(b - a)) / EDGE_RESOLUTION)))
    progress = np.linspace(0.0, 1.0, count + 1)
    return progress, (1.0 - progress[:, None]) * a + progress[:, None] * b


def configuration_grid():
    """Return the 0.5-degree grid and its sampled clearance-violation mask."""
    angles = np.arange(-180.0, 180.0 + GRID_STEP_DEG / 2, GRID_STEP_DEG)
    shoulder, elbow = np.meshgrid(angles, angles)
    q = np.deg2rad(np.stack((shoulder, elbow), axis=-1))
    forbidden = link_clearances(q).min(axis=-1) < REQUIRED_CLEARANCE
    return angles, forbidden


def detour_waypoints():
    """A simple candidate route, used to motivate the path definition."""
    goal = positive_elbow_goal()
    return np.array([START, [START[0], np.deg2rad(95)],
                     [goal[0], np.deg2rad(95)], goal])


def detour_samples():
    """Apply the same one-degree subdivision to each candidate-route segment."""
    waypoints = detour_waypoints()
    return np.concatenate([direct_edge_samples(a, b)[1]
                           for a, b in zip(waypoints[:-1], waypoints[1:])])


def print_calculations():
    """Print reproducible values for the opening slides and presenter notes."""
    goal = positive_elbow_goal()
    progress, samples = direct_edge_samples()
    clearance = link_clearances(samples)
    row, link = np.unravel_index(np.argmin(clearance), clearance.shape)
    midpoint = (START + goal) / 2
    np.set_printoptions(precision=9, suppress=True)
    print("Model: two whole line segments; static disc; bounded joint chart; no wrapping")
    print("Lengths [m]:", LENGTHS)
    print("Disc center [m]:", OBSTACLE_CENTER, "radius [m]:", OBSTACLE_RADIUS)
    print("Required extra clearance [m]:", REQUIRED_CLEARANCE)
    print("Joint-coordinate bounds [deg]: [-180, 180] for both coordinates")
    print("Target tip [m]:", TARGET)
    print("Start [deg]:", np.rad2deg(START))
    print("Goal [deg]:", np.rad2deg(goal))
    print("Goal [rad]:", goal)
    print("Start tip [m]:", arm_vertices(START)[-1])
    print("Goal tip [m]:", arm_vertices(goal)[-1])
    print("Start link clearances [mm]:", 1000 * link_clearances(START))
    print("Goal link clearances [mm]:", 1000 * link_clearances(goal))
    print("Direct edge subintervals:", len(progress) - 1, "samples:", len(progress))
    print("Max coordinate increments [deg]:", np.rad2deg(np.max(np.abs(np.diff(samples, axis=0)), axis=0)))
    print("Midpoint [deg]:", np.rad2deg(midpoint))
    print("Midpoint link clearances [mm]:", 1000 * link_clearances(midpoint))
    print("Minimum sampled clearance [mm]:", 1000 * clearance[row, link])
    print("Minimum sample index:", row, "progress:", progress[row], "link:", link + 1)
    print("Samples below required clearance:", np.count_nonzero(clearance.min(axis=-1) < REQUIRED_CLEARANCE))
    failed = np.flatnonzero(clearance.min(axis=-1) < REQUIRED_CLEARANCE)
    first = failed[0]
    print("Failing sample indices:", failed)
    print("First failure [deg]:", np.rad2deg(samples[first]))
    print("First failure disc-surface clearance [mm]:", 1000 * clearance[first].min())
    print("Detour waypoints [deg]:", np.rad2deg(detour_waypoints()))
    print("Detour minimum sampled disc-surface clearance [mm]:", 1000 * link_clearances(detour_samples()).min())
    print("Illustration grid step [deg]:", GRID_STEP_DEG)

    # Check geometric cases with independent, closed-form expected distances.
    a, b = np.array([0., 0.]), np.array([2., 0.])
    points = np.array([[1., 3.], [-1., 0.], [3., 4.]])
    np.testing.assert_allclose(point_segment_distance(points, a, b), [3., 1., np.sqrt(17.)])
    np.testing.assert_allclose(arm_vertices(goal)[-1], TARGET, atol=1e-14, rtol=0)
    assert min(link_clearances(START)) > REQUIRED_CLEARANCE
    assert min(link_clearances(goal)) > REQUIRED_CLEARANCE
    assert min(link_clearances(midpoint)) < 0
    assert np.max(np.abs(np.diff(samples, axis=0))) <= EDGE_RESOLUTION + 1e-14
    assert link_clearances(detour_samples()).min() >= REQUIRED_CLEARANCE
    print("Checks: projection cases, IK endpoint, clear endpoints, colliding midpoint, edge spacing PASS")


def workspace(ax):
    """Draw the common physical scene, with axes expressed in world metres."""
    ax.add_patch(Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, facecolor="#dddddd",
                        edgecolor=GREY, linewidth=1.5, zorder=1))
    ax.add_patch(Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS + REQUIRED_CLEARANCE,
                        facecolor="none", edgecolor="black", linestyle="--",
                        linewidth=1.5, zorder=2))
    ax.plot(*TARGET, marker="*", markersize=17, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.5, zorder=7)
    ax.set(xlim=(-0.04, 0.64), ylim=(-0.13, 0.59), xlabel="World x [m]", ylabel="World y [m]")
    ax.set_aspect("equal")
    ax.set_xticks([0, .2, .4, .6])
    ax.set_yticks([0, .2, .4])
    ax.grid(color="#e8e8e8", linewidth=.7, zorder=0)


def draw_arm(ax, q, color, marker="o", linestyle="-", label=None):
    points = arm_vertices(q)
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=4,
            marker=marker, markersize=8, linestyle=linestyle, label=label, zorder=4)


def draw_configuration_grid(ax, angles, forbidden, detail=True, shade=True):
    if shade:
        ax.pcolormesh(angles, angles, forbidden, shading="nearest", rasterized=True,
                      cmap=ListedColormap(["#ffffff", FORBIDDEN]), vmin=0, vmax=1, zorder=0)
        ax.contour(angles, angles, forbidden.astype(float), levels=[.5],
                   colors=[GREY], linewidths=.8, zorder=1)
    if detail:
        ax.set_xlim(-60, 120)
        ax.set_ylim(-30, 150)
        ax.set_xticks([-60, 0, 60, 120])
        ax.set_yticks([-30, 30, 90, 150])
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel(r"Shoulder $q_1$ [deg]")
    ax.set_ylabel(r"Elbow $q_2$ [deg]")
    ax.set_aspect("equal")
    ax.grid(color="#dddddd", linewidth=.6)


def save(fig, filename):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / filename, dpi=180, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    print("Wrote", FIG / filename)


def make_endpoints():
    """Show only the two endpoints, preserving the prediction before the reveal."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    for ax, q, color, title, target_label in zip(
        axes, (START, positive_elbow_goal()), (BLUE, GREEN),
        ("Start", "Fixed goal"), ((.14, -.10), (.28, .20)),
    ):
        workspace(ax)
        draw_arm(ax, q, color)
        ax.set_title(title, color=color)
        ax.annotate("Disc + 5 mm", xy=OBSTACLE_CENTER + [0., .055], xytext=(.49, .53),
                    ha="center", fontsize=20, arrowprops=dict(arrowstyle="-", color=GREY))
        ax.annotate("Target", xy=TARGET, xytext=target_label, ha="center", fontsize=20,
                    arrowprops=dict(arrowstyle="-", color=GREY))
    save(fig, "endpoints.png")


def make_blocked_edge():
    """Reveal the colliding midpoint and the finite samples of the direct edge."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    goal = positive_elbow_goal()
    midpoint = (START + goal) / 2
    workspace(axes[0])
    draw_arm(axes[0], midpoint, RED, marker="x")
    axes[0].set_title("Halfway posture")
    axes[0].annotate("Link 2 intersects disc", xy=OBSTACLE_CENTER, xytext=(.29, .54),
                     ha="center", fontsize=20, arrowprops=dict(arrowstyle="-", color=RED))
    progress, samples = direct_edge_samples()
    clearance_mm = 1000 * link_clearances(samples).min(axis=-1)
    axes[1].axhspan(-65, 5, color=FORBIDDEN, zorder=0)
    axes[1].axhline(5, color=GREY, linestyle="--", linewidth=1.5)
    axes[1].axhline(0, color=GREY, linewidth=.8)
    axes[1].plot(progress, clearance_mm, color=GREY, linewidth=1.5,
                  marker=".", markersize=5, zorder=3)
    for index, color, marker, label, offset in (
        (0, BLUE, "o", "Start", (9, 12)),
        (len(progress) // 2, RED, "x", "Halfway", (-110, 5)),
        (-1, GREEN, "s", "Goal", (-50, 12)),
    ):
        point = (progress[index], clearance_mm[index])
        axes[1].plot(*point, marker=marker, color=color, markersize=10, markeredgewidth=2, zorder=5)
        axes[1].annotate(label, xy=point, xytext=offset, textcoords="offset points", color=color, fontsize=20)
    axes[1].set(xlim=(-.03, 1.03), ylim=(-65, 130), xlabel="Progress along the connection",
                ylabel="Clearance from disc [mm]")
    axes[1].set_title("Minimum across both links")
    axes[1].set_xticks([0, .25, .5, .75, 1])
    axes[1].grid(color="#dddddd", linewidth=.6)
    axes[1].text(.30, 12, "5 mm required", fontsize=20, va="bottom")
    axes[1].text(.12, 105, "43 checked postures", fontsize=20, va="bottom")
    save(fig, "blocked-edge.png")


def make_configuration_space(angles, forbidden, shade=True):
    """Match postures to points; reveal violations and a checked candidate detour."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    goal = positive_elbow_goal()
    cases = ((START, BLUE, "o", "A: start"),
             ((START + goal) / 2, RED, "x", "B: halfway"),
             (goal, GREEN, "s", "C: goal"))
    workspace(axes[0])
    for q, color, marker, label in cases:
        draw_arm(axes[0], q, color, marker=marker, label=label)
    axes[0].set_title("Three arm postures")
    axes[0].legend(loc="upper left", fontsize=19, frameon=True, framealpha=.95)
    draw_configuration_grid(axes[1], angles, forbidden, shade=shade)
    _, samples = direct_edge_samples()
    degrees = np.rad2deg(samples)
    axes[1].plot(degrees[:, 0], degrees[:, 1], color=GREY, linestyle="--", linewidth=2, zorder=2)
    if shade:
        bad = link_clearances(samples).min(axis=-1) < REQUIRED_CLEARANCE
        axes[1].plot(degrees[bad, 0], degrees[bad, 1], color=RED,
                     linestyle="none", marker=".", markersize=4, zorder=3)
        detour = np.rad2deg(detour_waypoints())
        axes[1].plot(detour[:, 0], detour[:, 1], color=BLUE, linewidth=2.5,
                     marker=".", markersize=7, zorder=2)
        axes[1].annotate("Detour", xy=(8, 95), xytext=(-35, 122), fontsize=20, color=BLUE,
                         arrowprops=dict(arrowstyle="-", color=BLUE))
    for (q, color, marker, label), offset in zip(cases, ((18, -20), (24, 5), (-28, -20))):
        point = np.rad2deg(q)
        axes[1].plot(*point, marker=marker, color=color, markersize=9, markeredgewidth=2, zorder=4)
        axes[1].annotate(label[0], xy=point, xytext=offset, textcoords="offset points", color=color,
                        fontsize=22, arrowprops=dict(arrowstyle="-", color=color, linewidth=1.1))
    axes[1].set_title("Postures in joint coordinates")
    grid_label = "Forbidden:\nclearance\n< 5 mm" if shade else "Bounded chart;\nno wrapping"
    axes[1].text(.03, .40, grid_label,
                 transform=axes[1].transAxes, fontsize=19, va="center")
    # Both reveal stages keep identical axes and an inset of the full joint box.
    inset = axes[1].inset_axes([.69, .67, .28, .28])
    if shade:
        inset.pcolormesh(angles, angles, forbidden, shading="nearest", rasterized=True,
                         cmap=ListedColormap(["white", FORBIDDEN]), vmin=0, vmax=1)
    inset.add_patch(Rectangle((-60, -30), 180, 180, fill=False, edgecolor=BLUE, linewidth=1.5))
    inset.set(xlim=(-180, 180), ylim=(-180, 180), xticks=[], yticks=[], aspect="equal")
    for spine in inset.spines.values():
        spine.set_visible(True)
    inset.set_title("Full chart", fontsize=16, pad=4)
    save(fig, "configuration-space.png" if shade else "configurations.png")


def main():
    print_calculations()
    angles, forbidden = configuration_grid()
    make_endpoints()
    make_blocked_edge()
    make_configuration_space(angles, forbidden, shade=False)
    make_configuration_space(angles, forbidden)


if __name__ == "__main__":
    main()
