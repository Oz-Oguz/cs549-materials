"""Slide-size variants for Deck 06's round-5 review.

Run from the repository root with:
  .venv/bin/python lectures/04-D2-motion-planning/code/make_review_figures.py --out lectures/04-D2-motion-planning/fig

Writes only eight new PNGs, into the --out directory. Without --out, or with
--check, it solves, asserts and draws but writes no deck PNG. --komo-only limits
the run to the four trajectory-optimization figures (the three KOMO slides and
the between-knot slide); --prm-only limits it to the lecture roadmap.
The KOMO slides are drawn from named runs of the three-point model [3P] (komo_two_link.py), each solved by the
NumPy port of RAI's loop (rai_nlp.py) under RAI's default options from the exact straight-line start:
3P-auglag-default-guessA, 3P-auglag-default-guessB and 3P-auglag-default-guessB-k1. Each run is compared with the
recorded real-RAI run of the same case ID (../fixtures/komo_two_link_rai.json, sos to 1e-6 relative), every KOMO slide
names its case IDs and 'exact start', and the JSON report is keyed by case ID. A default run stops on a step test, so
the multiplier bars and the closed-form check against the disc-free quadratic program are drawn only for a converged
('tight') case; with the default cases they are left out.
It imports the KOMO model and the sampling module's Python reference without changing either.
The separate browser counterexample is exported from its actual TypeScript
preset and geometry functions using the already installed esbuild and Node.
No browser screenshots, module assets, or Deck 05 figures are rewritten.

The fixed-size canvases deliberately have larger labels than the note figures.
Recommended display widths: costs 700 px; paths 960 px; band 500 px;
between-knot 560 px; RRT 850 px. The configuration grids are sampled illustrations.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"
PREVIEW = None
OUT = None  # set by --out; None draws and checks without writing a deck PNG
WIDTHS = {"komo-costs-slide.png": 700, "komo-paths-slide.png": 960,
          "komo-band-slide.png": 500, "between-knot-slide.png": 560,
          "rrt-paused-slide.png": 850, "rrt-connected-slide.png": 850,
          "prm-slide.png": 960, "connect-slide.png": 780}
MAROON, BLUE, GREEN, RED = "#9b1b4b", "#1f6f8b", "#2a7d2a", "#b00020"
GREY, LIGHT = "#6b6b6b", "#d2d2d2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
                     "axes.titlesize": 19, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vertices(q, lengths=(0.4, 0.3), base=(0.0, 0.0)):
    """Vectorized base, elbow, and tip; joint angles are in radians."""
    q = np.asarray(q)
    first = lengths[0] * np.stack([np.cos(q[..., 0]), np.sin(q[..., 0])], -1)
    second = lengths[1] * np.stack([np.cos(q.sum(-1)), np.sin(q.sum(-1))], -1)
    return np.stack([np.zeros_like(first), first, first + second], -2) + base


def clearances(q, center, radius, lengths=(0.4, 0.3), base=(0.0, 0.0)):
    """Smallest whole-segment clearance from the physical disc surface."""
    v = vertices(q, lengths, base)
    a, b = v[..., :-1, :], v[..., 1:, :]
    d, offset = b - a, np.asarray(center) - a
    u = np.clip(np.sum(offset * d, -1) / np.sum(d * d, -1), 0, 1)
    return np.min(np.linalg.norm(offset - u[..., None] * d, axis=-1), -1) - radius


def arm(ax, q, color, lengths=(0.4, 0.3), base=(0, 0), **kwargs):
    v = vertices(q, lengths, base)
    ax.plot(v[:, 0], v[:, 1], color=color, **kwargs)


def finish(fig, name):
    if OUT is not None:
        fig.savefig(OUT / name, dpi=180, facecolor="white")
    if PREVIEW:
        fig.savefig(PREVIEW / name, dpi=WIDTHS[name] / fig.get_figwidth(), facecolor="white")
    plt.close(fig)


# the [3P] runs behind the KOMO slides: (k, guess toward the q2 > 0 solution) -> case ID
RETAINED = {(2, True): "3P-auglag-default-guessA", (2, False): "3P-auglag-default-guessB", (1, False): "3P-auglag-default-guessB-k1"}


def solve_retained(km, cases=RETAINED):
    """Run each case with the port from its exact start and compare it with the fixture. Returns answers[(k, positive)] =
    (knots, lambda, cost), with answers["ids"] and answers["info"] keyed the same way."""
    answers = {"ids": {}, "info": {}}
    for key, cid in cases.items():
        z, lam, kap, info = km.solve_komo(cid, verbose=False)
        km.compare_with_fixture(cid, info)
        answers[key] = (km.knots(z), lam, km.cost(z))
        answers["ids"][key], answers["info"][key] = cid, info
        # Only a converged run can be compared with the disc-free quadratic program (equality-constrained, no obstacle
        # rows), and only when the disc is inactive at its answer.
        if km.rai_nlp.Case(cid).opts == "tight" and not key[1] and np.max(km.ineq_constraints(z)[0]) < -1e-3:
            zeros = np.zeros(km.N)
            offset, jac = km.acc_residuals(zeros)
            eq = np.zeros((4, km.N))
            eq[:2, -2:] = np.eye(2)
            eq[2:, -4:-2], eq[2:, -2:] = -np.eye(2), np.eye(2)
            rhs = np.r_[km.ik_two_link(km.Y_STAR, False), np.zeros(2)]
            system = np.block([[2 * jac.T @ jac, eq.T], [eq, np.zeros((4, 4))]])
            unconstrained = np.linalg.solve(system, np.r_[-2 * jac.T @ offset, rhs])[:km.N]
            assert np.max(np.abs(unconstrained - z)) < 2e-8, f"{cid}: differs from the disc-free quadratic program"
    km.T, km.N, km.K_ORDER = 20, 40, 2
    return answers


def cost_comparison(km, answers, *, goal=None, output_path=None):
    """Draw the shared two-panel comparison; defaults preserve the deck figure.

    A supplied goal and output_path allow a chapter comparison without changing
    the deck endpoint or overwriting its assets. Answers contain solved paths.
    """
    chapter = goal is not None
    ids = answers.get("ids", {})
    fig, (ax, work) = plt.subplots(1, 2, figsize=(9.4, 5.0),
                                  gridspec_kw={"width_ratios": [1.18, 1]})
    fig.subplots_adjust(left=.09, right=.97, bottom=.27, top=.84, wspace=.40)
    styles = [(1, MAROON, "s", "Velocity cost"), (2, BLUE, "o", "Acceleration cost")]
    if not chapter:
        # the deck's runs: the legend names each case, and the bottom margin holds the longer labels
        styles = [(o, c, m, f"{l}: {ids[o, False]}") for o, c, m, l in styles]
        fig.subplots_adjust(bottom=.30, top=.80)
    for order, color, marker, label in styles:
        Q = answers[order, False][0]
        steps = np.degrees(np.linalg.norm(np.diff(Q, axis=0), axis=1))
        ax.plot(np.arange(1, 21), steps, color=color, marker=marker, ms=4.8, lw=2.5, label=label)
    ax.set(xlabel="Knot $t$", ylabel="Joint step [deg]", ylim=(-.2, 10.5), xlim=(0, 20.5))
    ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yticks([0, 5, 10]); ax.grid(color="#e7e7e7")
    ax.set_title("Steps between knots", pad=14)
    # Both paths lie on the same joint-space segment. Draw its dense FK image
    # once, then plot corresponding knot times from each solved schedule.
    route_deviation = {}
    if chapter:
        # exact solutions of the disc-free program: both lie on the straight joint-space segment to the goal
        goal = np.asarray(goal)
        direction = goal - km.Q0
        for order in [1, 2]:
            Q = answers[order, False][0]
            progress = (Q - km.Q0) @ direction / (direction @ direction)
            np.testing.assert_allclose(Q, km.Q0 + progress[:, None] * direction, atol=2e-8)
            assert np.min(np.diff(progress)) > -2e-8
        Qroute = km.Q0 + np.linspace(0, 1, 401)[:, None] * (goal - km.Q0)
        curve = vertices(Qroute)[:, -1]
        work.plot(curve[:, 0], curve[:, 1], color="#555", lw=2, zorder=1)
    else:
        # the deck's runs: each tip path drawn from its own knots (joint-space interpolation between them), and the
        # distance of each run from the straight joint-space segment to its own last knot reported, not assumed
        for order, color, _, _ in styles:
            Q = answers[order, False][0]
            dense = np.vstack([(1 - s) * Q[t] + s * Q[t + 1] for t in range(len(Q) - 1) for s in np.linspace(0, 1, 21)])
            path = vertices(dense)[:, -1]
            work.plot(path[:, 0], path[:, 1], color=color, lw=2, zorder=1)
            direction = Q[-1] - km.Q0
            progress = (Q - km.Q0) @ direction / (direction @ direction)
            route_deviation[ids[order, False]] = float(np.degrees(np.max(np.linalg.norm(Q - (km.Q0 + progress[:, None] * direction), axis=1))))
        curve = vertices(np.vstack([km.Q0, answers[2, False][0][-1]]))[:, -1]
    marks = [4, 10, 16] if chapter else [4, 8, 12, 16]
    tips = {}
    for order, color, marker, label in styles:
        tips[order] = vertices(answers[order, False][0])[:, -1]
        work.plot(tips[order][marks, 0], tips[order][marks, 1], ls="none",
                  marker=marker, color=color, ms=8, zorder=3)
    for knot in marks:
        point = (tips[1][knot] + tips[2][knot]) / 2
        work.text(point[0] + .014, point[1], f"{knot}", fontsize=16, va="center")
    for point, label, dy in [(curve[0], "Start", .022), (curve[-1], "Goal", -.035)]:
        work.plot(*point, "o", color="black", ms=6)
        work.text(point[0] - .008, point[1] + dy, label, fontsize=16,
                  ha="left" if label == "Start" else "right")
    work.set(xlim=(.38, .65), ylim=(.15, .54), xlabel="World $x$ [m]", ylabel="World $y$ [m]")
    work.set_xticks([.4, .5, .6]); work.set_yticks([.2, .3, .4, .5]); work.set_aspect("equal")
    if chapter:
        ax.set(ylim=(-.12, 4.55), xlabel="Knot index $t$")
        ax.set_yticks([0, 1, 2, 3, 4])
        work.set(xlim=(.39, .62), ylim=(.15, .54))
    if chapter:
        work.set_title("Same geometric route\nDifferent timing", pad=12)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
                   frameon=False, fontsize=17, bbox_to_anchor=(.5, .015))
    else:
        work.set_title("Tip paths", pad=12)
        fig.suptitle(f"{ids[2, False]} and {ids[1, False]}, exact start", fontsize=13, y=.985)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1,
                   frameon=False, fontsize=13, bbox_to_anchor=(.5, .005))
    if output_path is None:
        finish(fig, "komo-costs-slide.png")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, facecolor="white")
        plt.close(fig)
    if chapter:
        return {"costs": {str(k): answers[k, False][2] for k in [1, 2]},
                "largest_matching_knot_tip_separation_m": float(np.max(np.linalg.norm(tips[1] - tips[2], axis=1)))}
    return {"cases": {str(k): ids[k, False] for k in [1, 2]}, "start": "exact start",
            "costs": {ids[k, False]: answers[k, False][2] for k in [1, 2]},
            "largest_deviation_from_straight_joint_segment_deg": route_deviation,
            "largest_matching_knot_tip_separation_m": {f"{ids[1, False]} vs {ids[2, False]}": float(np.max(np.linalg.norm(tips[1] - tips[2], axis=1)))}}


def solutions(km, answers):
    ids = answers["ids"]
    idA, idB = ids[2, True], ids[2, False]
    bars = km.rai_nlp.Case(idA).opts == "tight"      # multipliers are drawn only for a converged run
    if bars:
        fig, (work, ax) = plt.subplots(1, 2, figsize=(10.5, 4.9), gridspec_kw={"width_ratios": [1.1, 1]})
        fig.subplots_adjust(left=.07, right=.98, bottom=.28, top=.84, wspace=.30)
    else:
        fig, work = plt.subplots(figsize=(6.4, 5.6))
        fig.subplots_adjust(left=.14, right=.97, bottom=.27, top=.88)
    work.add_patch(Circle(km.OBS_C, km.OBS_R, fc=LIGHT, ec="#666", lw=1.4))
    work.annotate("Disc", xy=km.OBS_C + [.04, .025], xytext=(.70, .46), fontsize=17,
                  ha="center", arrowprops={"arrowstyle": "-", "color": "#666", "lw": 1.2})
    handles = []
    for branch, color, style in [(True, MAROON, "-"), (False, BLUE, "--")]:
        Q, lam, cost = answers[2, branch]
        for q in Q[::4]:
            arm(work, q, color, lw=1.6, alpha=.28)
        tip = vertices(Q)[:, -1]
        work.plot(tip[:, 0], tip[:, 1], style, marker="o", color=color, lw=2.8, ms=4)
        handles.append(Line2D([], [], color=color, marker="o", ls=style, lw=2.8, label=f"{ids[2, branch]}: cost {cost:.2f}"))
    start = km.p_tip(km.Q0)
    work.plot(*start, "s", color="black", ms=7)
    work.text(start[0] - .02, start[1] + .02, "Start", ha="right", fontsize=17)
    work.plot(*km.Y_STAR, "*", color="black", ms=14)
    work.text(km.Y_STAR[0] + .025, km.Y_STAR[1] - .035, "Target", fontsize=17)
    work.set(xlim=(-.03, .75), ylim=(-.075, .63), xlabel="World $x$ [m]", ylabel="World $y$ [m]")
    work.set_aspect("equal"); work.set_xticks([0, .2, .4, .6]); work.set_yticks([0, .2, .4, .6])
    work.set_title("Tip routes of two runs, exact start", pad=12)
    report = {"start": "exact start", "costs": {idA: answers[2, True][2], idB: answers[2, False][2]}}
    if bars:
        multipliers = answers[2, True][1].reshape(km.T, 3)
        bottom = np.zeros(km.T)
        for i, color in enumerate([MAROON, BLUE, GREY]):
            ax.bar(np.arange(1, km.T + 1), multipliers[:, i], bottom=bottom, color=color, width=.8)
            bottom += multipliers[:, i]
        active = np.flatnonzero(multipliers.sum(axis=1) > 1e-6)
        for i in active:
            ax.text(i + 1, bottom[i] + 5, f"{bottom[i]:.2f}", ha="center", fontsize=15)
        ax.set(xlim=(0, 21), ylim=(0, 1.15 * bottom.max()), xlabel="Knot $t$", ylabel="Multiplier $\\lambda$")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_title(f"{idA}: active rows", pad=12, fontsize=13)
        report["multipliers"] = {idA: {f"knot {int(i) + 1}": float(bottom[i]) for i in active}}
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=14,
               bbox_to_anchor=(.5, .005))
    finish(fig, "komo-paths-slide.png")
    return report


def sparsity(km, answers):
    cid, info = answers["ids"][2, True], answers["info"][2, True]
    z = km.linear_init(km.ik_two_link(km.Y_STAR, True))
    # The feature Jacobian itself, before multiplier shifts/row rescaling, at the exact start of the case.
    J = np.vstack([km.acc_residuals(z)[1], km.eq_constraints(z)[1], km.ineq_constraints(z)[1]])
    _, J_al = km.al_residual(z, np.ones(3 * km.T), np.zeros(4), info["mu_unscaled"])
    assert np.array_equal(J != 0, J_al != 0)
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    fig.subplots_adjust(left=.18, right=.62, bottom=.18, top=.90)
    ax.imshow(J != 0, cmap=ListedColormap(["white", MAROON]), interpolation="nearest", aspect="auto")
    for y in [39.5, 43.5]: ax.axhline(y, color=BLUE, lw=1.7)
    for y, text in [(19.5, "Acceleration\n40 rows\nUp to 3 knots"),
                    (41.5, "Target and rest\nRows 41–44\nLast 2 knots"),
                    (73.5, "Obstacle rows\n60 rows\n1 knot")]:
        ax.annotate(text, xy=(39.7, y), xytext=(43, y), va="center", fontsize=15,
                    annotation_clip=False, arrowprops={"arrowstyle": "-", "color": GREY})
    ax.set_xticks([.5, 18.5, 38.5], ["1", "10", "20"])
    ax.set_yticks([0, 39, 103], ["1", "40", "104"])
    ax.set(xlabel="Knot column blocks\n2 joint columns each", ylabel="Scalar rows")
    fig.suptitle("Feature Jacobian: 104 × 40", fontsize=18, y=.99)
    fig.text(.5, .015, f"{cid}, exact start", ha="center", fontsize=11, color=GREY)
    finish(fig, "komo-band-slide.png")
    return {"case": cid, "start": "exact start", "shape": list(J.shape), "nonzeros": int(np.count_nonzero(J))}


def browser_preset():
    """Execute original TypeScript functions; export raw poses and readouts."""
    app = ROOT / "modules/opt-explorer"
    code = f'''
import {{betweenKnotPreset, interpolateTrajectory, trajectoryClearances}} from {json.dumps(str(app / 'src/trajectorySampling.ts'))};
import {{APP_ARM, setArmModel, configToAngles, forwardKinematics, taskSpaceSignedDistance}} from {json.dumps(str(app / 'src/optimizer.ts'))};
setArmModel(APP_ARM);
const p = betweenKnotPreset();
const samples = interpolateTrajectory(p.trajectory);
const clearance = trajectoryClearances(p.trajectory, p.obstacles);
console.log(JSON.stringify({{arm: APP_ARM, ...p, samples, clearance,
  angles: p.trajectory.map(configToAngles),
  sampleAngles: samples.map(configToAngles),
  poses: p.trajectory.map(q => forwardKinematics(...configToAngles(q))),
  knotClearances: p.trajectory.map(q => taskSpaceSignedDistance(q, p.obstacles[0]))
}}));
'''
    with tempfile.TemporaryDirectory(prefix="cs549-review-figures-") as temp:
        entry = Path(temp) / "preset.ts"
        entry.write_text(code)
        bundle = subprocess.check_output([str(app / "node_modules/.bin/esbuild"), str(entry),
                                           "--bundle", "--platform=node", "--format=cjs", "--log-level=error"], text=True)
        return json.loads(subprocess.check_output(["node", "-e", bundle], text=True))


def shade_grid(ax, xlim, ylim, center, radius, lengths=(.4, .3), base=(0, 0), margin=0, step=1):
    x = np.arange(xlim[0], xlim[1] + step / 2, step)
    y = np.arange(ylim[0], ylim[1] + step / 2, step)
    a, b = np.meshgrid(x, y)
    invalid = clearances(np.deg2rad(np.stack([a, b], -1)), center, radius, lengths, base) < margin
    ax.pcolormesh(x, y, invalid, cmap=ListedColormap(["white", LIGHT]), shading="nearest", vmin=0, vmax=1, zorder=0)


def between_knots(data):
    lengths = (data["arm"]["l1"], data["arm"]["l2"])
    ob = data["obstacles"][0]; center = np.array([ob["x"], ob["y"]]); radius = ob["r"]
    Q, samples = np.array(data["angles"]), np.array(data["sampleAngles"])
    clearance = clearances(samples, center, radius, lengths, (.5, .5))
    np.testing.assert_allclose(clearances(Q, center, radius, lengths, (.5, .5)), data["knotClearances"], atol=1e-15)
    assert abs(clearance.min() - data["clearance"]["sampledMinimum"]) < 1e-15
    worst = samples[np.argmin(clearance)]
    fig, work = plt.subplots(figsize=(5.8, 4.9))
    fig.subplots_adjust(left=.17, right=.96, bottom=.28, top=.94)
    for i, q in enumerate(Q):
        arm(work, q, GREY, lengths, (.5, .5), lw=3, marker="o", ms=4, alpha=.75)
        tip = vertices(q, lengths, (.5, .5))[-1]
        work.annotate(str(i), xy=tip, xytext=(5, 6), textcoords="offset points", fontsize=17)
    curve = vertices(samples, lengths, (.5, .5))[:, -1]
    work.plot(curve[:, 0], curve[:, 1], "--", color=BLUE, lw=2)
    arm(work, worst, RED, lengths, (.5, .5), lw=3, ls="--")
    work.add_patch(Circle(center, radius, fc=LIGHT, ec=RED, lw=1.4))
    work.text(.94, .54, "Disc", ha="left", fontsize=17)
    work.set(xlim=(.44, 1.02), ylim=(.12, .90), xlabel="Workspace $x$", ylabel="Workspace $y$")
    work.set_xticks([.5, .7, .9]); work.set_yticks([.2, .4, .6, .8]); work.set_aspect("equal")
    fig.legend(handles=[Line2D([], [], marker="o", color=GREY, label="Knot posture"),
                        Line2D([], [], color=RED, ls="--", label="Between knots")],
               loc="lower center", ncol=2, fontsize=16, frameon=False, bbox_to_anchor=(.5, .025))
    finish(fig, "between-knot-slide.png")
    return {"arm": data["arm"], "angles_deg": np.degrees(Q).tolist(),
            "knot_minimum": data["clearance"]["knotMinimum"], "sampled_minimum": float(clearance.min()),
            "sample_count": len(samples), "subdivisions": data["clearance"]["subdivisions"],
            "worst_angles_deg": np.degrees(worst).tolist(), "obstacle": ob}


def rrt_figures(rrt):
    saved_path = ROOT / "lectures/05-motion-planning/sampling-demo/evidence/check/ref-default.json"
    saved = json.loads(saved_path.read_text())
    trace = []
    result = rrt.rrt(trace=trace)
    # The existing browser regression's saved raw states remain the source of
    # the pictured tree. The live reference must agree before we use them.
    np.testing.assert_allclose(result["nodes"], saved["result"]["nodes"], atol=1e-14, rtol=0)
    assert result["parents"] == saved["result"]["parents"] and result["path"] == saved["result"]["path"]
    nodes = np.asarray(saved["result"]["nodes"])
    path = saved["result"]["path"]
    worst, checked, fail = rrt.revalidate(nodes.tolist(), path)
    assert fail is None and checked == 74
    length = sum(np.linalg.norm(nodes[b] - nodes[a]) for a, b in zip(path, path[1:]))
    deep = rrt.deepest_along(rrt.Q_START, rrt.Q_GOAL)
    for connected in [False, True]:
        fig, (work, config) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.05]})
        fig.subplots_adjust(left=.07, right=.98, bottom=.28 if connected else .37, top=.86, wspace=.36)
        work.add_patch(Circle(rrt.OBS_C, rrt.OBS_R, fc=LIGHT, ec=GREY, lw=1.4))
        work.add_patch(Circle(rrt.OBS_C, rrt.OBS_R + rrt.MARGIN, fill=False, ec=GREY, ls="--", lw=1.3))
        for q, color, name in [(rrt.Q_START, BLUE, "Start"), (rrt.Q_GOAL, GREEN, "Goal")]:
            arm(work, q, color, lw=3, marker="o", ms=4)
            tip = vertices(q)[-1]
            work.text(tip[0] + .025, tip[1], name, color=color, fontsize=16)
        if connected:
            for idx in path[1:-1]: arm(work, nodes[idx], BLUE, lw=1.4, alpha=.35)
        else:
            arm(work, deep[3], RED, lw=2.5, ls="--")
        work.set(xlim=(-.04, .70), ylim=(-.13, .60), xlabel="World $x$ [m]", ylabel="World $y$ [m]")
        work.set_aspect("equal"); work.set_xticks([0, .3, .6]); work.set_yticks([0, .3, .6])
        work.set_title("Arm postures", pad=12)
        window = ((-60, 120), (-30, 150)) if connected else ((-180, 180), (-180, 180))
        shade_grid(config, *window, rrt.OBS_C, rrt.OBS_R, margin=rrt.MARGIN, step=1)
        if connected:
            for idx, parent in enumerate(result["parents"]):
                if parent >= 0:
                    pts = np.degrees(nodes[[parent, idx]])
                    config.plot(pts[:, 0], pts[:, 1], color="#aaa", lw=1.1, zorder=1)
            route = np.degrees(nodes[path])
            config.plot(route[:, 0], route[:, 1], "-o", color=BLUE, lw=3, ms=4, zorder=4)
        direct = np.degrees([rrt.Q_START, rrt.Q_GOAL])
        config.plot(direct[:, 0], direct[:, 1], "--", color=RED, lw=1.6, zorder=2)
        for q, color, name, marker, offset in [(rrt.Q_START, BLUE, "Start", "o", (8, -10)),
                                                (rrt.Q_GOAL, GREEN, "Goal", "D", (-14, 12))]:
            xy = np.degrees(q)
            config.plot(*xy, marker, color=color, ms=7, zorder=5)
            config.annotate(name, xy=xy, xytext=offset, textcoords="offset points", color=color,
                            fontsize=16, ha="left" if name == "Start" else "right")
        config.set(xlim=window[0], ylim=window[1], xlabel=r"$q_1$ [deg]", ylabel=r"$q_2$ [deg]")
        ticks = [-180, 0, 180] if not connected else [-60, 0, 60, 120]
        config.set_xticks(ticks); config.set_yticks([-180, 0, 180] if not connected else [0, 60, 120])
        config.set_aspect("equal"); config.grid(color="#ececec", lw=.7)
        config.set_title("Joint-space window" if connected else "Full joint chart", pad=12)
        handles = [Line2D([], [], color=RED, ls="--", label="Invalid direct edge"),
                   Patch(color=LIGHT, label="Sampled forbidden region")]
        if connected:
            handles.insert(0, Line2D([], [], color=BLUE, marker="o", label="Returned route"))
        else:
            handles[0].set_label("Invalid direct edge (joint chart)")
            handles.insert(0, Line2D([], [], color=RED, ls="--",
                label="Deepest posture on direct edge: 47.7 mm inside disc (workspace)"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles) if connected else 1, frameon=False,
                   fontsize=14.5, bbox_to_anchor=(.5, .012))
        finish(fig, "rrt-connected-slide.png" if connected else "rrt-paused-slide.png")
    return {"iterations": result["iterations"], "node_count": len(nodes), "path": path,
            "length_deg": float(np.degrees(length)), "checked": checked,
            "minimum_disc_clearance_m": worst + rrt.MARGIN,
            "direct_deepest_index": deep[1], "direct_subintervals": deep[2],
            "direct_deepest_disc_clearance_m": deep[0] + rrt.MARGIN,
            "first_step": trace[0], "second_step": trace[1]}


def prm_figure(rrt):
    """A denser lecture roadmap; the module's original 60-sample run is retained."""
    stage = load_module("lecture_prm", "modules/motion-planning/staged_two_link.py")
    data = stage.prm(seed=7, n_samples=400, k=5)
    nodes = np.asarray(data["nodes"])
    route = nodes[data["path"]]
    assert len(route) == 6
    validation = stage.validate_path(route.tolist())
    assert validation["valid"]

    # Independently relax the complete edge list, rather than repeat Dijkstra.
    edges = [(i, j, w) for i, neighbors in data["edges"].items() for j, w in neighbors]
    src, dst = np.array([(i, j) for i, j, _ in edges]).T
    weights = np.array([w for _, _, w in edges])
    np.testing.assert_allclose(weights, np.linalg.norm(nodes[src] - nodes[dst], axis=1), atol=1e-14)
    distances = np.full(len(nodes), np.inf)
    distances[0] = 0
    for _ in range(len(nodes) - 1):
        updated = distances.copy()
        np.minimum.at(updated, dst, distances[src] + weights)
        if np.array_equal(updated, distances):
            break
        distances = updated
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    assert abs(distances[1] - lengths.sum()) < 1e-12

    # Independently reconstruct every checked posture, including the start.
    checks, dense = [route[0]], [route[0]]
    for a, b in zip(route[:-1], route[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / rrt.SUB)))
        checks.extend(a + np.arange(1, n + 1)[:, None] / n * (b - a))
        dense.extend(a + np.linspace(0, 1, 101)[1:, None] * (b - a))
    checked_clearance = clearances(np.asarray(checks), rrt.OBS_C, rrt.OBS_R)
    minimum = float(checked_clearance.min())
    assert minimum >= rrt.MARGIN
    assert abs(minimum - validation["deepest"]["c"] - rrt.MARGIN) < 1e-12
    certificate = minimum - .5 * (rrt.L1 + 2 * rrt.L2) * rrt.SUB
    assert certificate > rrt.MARGIN

    amber = "#f3ca84"
    fig, (ax, work) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    fig.subplots_adjust(left=.08, right=.98, bottom=.32, top=.86, wspace=.34)
    # Local view only; sampling still covers the full allowed joint box.
    x, y = np.arange(-35, 55.25, .25), np.arange(25, 105.25, .25)
    a, b = np.meshgrid(x, y)
    gap = clearances(np.deg2rad(np.stack([a, b], -1)), rrt.OBS_C, rrt.OBS_R)
    ax.pcolormesh(x, y, gap < rrt.MARGIN, cmap=ListedColormap(["white", amber]),
                  shading="nearest", vmin=0, vmax=1, zorder=0)
    ax.contour(x, y, gap, levels=[rrt.MARGIN], colors=["#a56800"], linewidths=1)
    deg = np.degrees(nodes)
    for i, neighbors in data["edges"].items():
        for j, _ in neighbors:
            if i < j:
                ax.plot(*deg[[i, j]].T, color="#b7b7b7", lw=.7, zorder=1)
    ax.plot(*deg[2:].T, "o", color="#888", ms=2.8, zorder=2)
    for pair in data["rejected"]:
        ax.plot(*deg[list(pair)].T, color=RED, ls=":", lw=1.4, zorder=2)
    ax.plot(*np.degrees(route).T, "-o", color=BLUE, lw=3.2, ms=5, zorder=5)
    labels = ["Start", "1", "2", "3", "4", "Goal"]
    offsets = [(8, -15), (9, 0), (9, 4), (-2, -20), (-4, 10), (-28, 9)]
    for q, label, offset in zip(np.degrees(route), labels, offsets):
        ax.annotate(label, q, xytext=offset, textcoords="offset points", color=BLUE,
                    fontsize=15, zorder=7)
    ax.set(xlim=(-35, 55), ylim=(25, 105), xlabel=r"$q_1$ [deg]", ylabel=r"$q_2$ [deg]")
    ax.set_xticks([-30, 0, 30]); ax.set_yticks([30, 60, 90]); ax.set_aspect("equal")
    ax.set_title("Roadmap near the route", pad=12)

    work.add_patch(Circle(rrt.OBS_C, rrt.OBS_R, fc=amber, ec="#a56800", lw=1.4))
    work.add_patch(Circle(rrt.OBS_C, rrt.OBS_R + rrt.MARGIN, fill=False, ec="#a56800", ls="--", lw=1.2))
    for i, q in enumerate(route):
        arm(work, q, BLUE if i < len(route)-1 else GREEN, lw=2,
            alpha=1 if i in (0, len(route)-1) else .32)
    tip_curve = vertices(np.asarray(dense))[:, -1]
    work.plot(*tip_curve.T, color=BLUE, lw=2, zorder=4)
    tips = vertices(route)[:, -1]
    work.plot(*tips.T, "o", color=BLUE, ms=4, zorder=5)
    for xy, label, offset in zip(tips, labels, [(8, 3), (7, 6), (-19, 5), (-25, -5), (-23, -1), (8, -12)]):
        work.annotate(label, xy, xytext=offset, textcoords="offset points", fontsize=15,
                      color=GREEN if label == "Goal" else BLUE)
    work.set(xlim=(-.04, .68), ylim=(-.13, .60), xlabel="World $x$ [m]", ylabel="World $y$ [m]")
    work.set_xticks([0, .3, .6]); work.set_yticks([0, .3, .6]); work.set_aspect("equal")
    work.set_title("Arm postures and tip route", pad=12)
    handles = [Line2D([], [], color="#aaa", label="Graph edge"),
               Line2D([], [], color=RED, ls=":", label="Rejected edge"),
               Line2D([], [], color=BLUE, lw=3, label="Selected route"),
               Patch(color=amber, label="Below 5 mm (left)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=15,
               bbox_to_anchor=(.5, .01))
    finish(fig, "prm-slide.png")
    return {"prm_nodes": len(nodes), "prm_tested": data["tested"], "prm_accepted": data["accepted"],
            "prm_path": data["path"], "prm_checked": data["checked_configurations"],
            "seed": 7, "n_samples": 400, "k": 5, "route_degrees": np.degrees(route).tolist(),
            "edge_lengths_degrees": np.degrees(lengths).tolist(),
            "settled_before_goal": data["settled_before_goal"], "validation": validation,
            "minimum_surface_clearance_m": minimum, "continuous_clearance_bound_m": certificate,
            "independent_shortest_length_rad": float(distances[1]), "data": data}


def staged_figures(rrt):
    """Draw the denser lecture PRM and preserve the staged Connect example."""
    stage = load_module("review_staged", "modules/motion-planning/staged_two_link.py")
    source = ROOT / "lectures/05-motion-planning/sampling-demo/evidence/staged-seed7.json"
    saved = json.loads(source.read_text())
    prm, connect = stage.prm(), stage.rrt_connect()
    np.testing.assert_allclose(prm["nodes"], saved["prm"]["nodes"], atol=1e-14, rtol=0)
    assert prm["path"] == saved["prm"]["path"]
    assert prm["tested"] == 191 and prm["accepted"] == 187
    for name in ["A", "B"]:
        np.testing.assert_allclose(connect[name]["nodes"], saved["rrt_connect"][name]["nodes"], atol=1e-14, rtol=0)
        assert connect[name]["parents"] == saved["rrt_connect"][name]["parents"]
    np.testing.assert_allclose(connect["path"], saved["rrt_connect"]["path"], atol=1e-14, rtol=0)
    assert connect["iterations"] == 14 and connect["checked_configurations"] == 186

    def poly(ax, points, **kw):
        p = np.degrees(points)
        ax.plot(p[:, 0], p[:, 1], **kw)

    def cspace(ax, full=False):
        bounds = ((-180, 180), (-180, 180)) if full else ((-60, 120), (-30, 150))
        shade_grid(ax, *bounds, rrt.OBS_C, rrt.OBS_R, margin=rrt.MARGIN, step=1)
        ax.set(xlim=bounds[0], ylim=bounds[1], xlabel=r"$q_1$ [deg]", ylabel=r"$q_2$ [deg]")
        ax.set_xticks([-180, 0, 180] if full else [-60, 0, 60, 120])
        ax.set_yticks([-180, 0, 180] if full else [0, 60, 120])
        ax.set_aspect("equal"); ax.grid(color="#ededed", lw=.5)
        for q, color, name, offset in [(rrt.Q_START, BLUE, "Start", (7, -10)),
                                      (rrt.Q_GOAL, GREEN, "Goal", (-7, 10))]:
            xy = np.degrees(q)
            ax.plot(*xy, "o" if name == "Start" else "D", color=color, ms=6, zorder=6)
            ax.annotate(name, xy=xy, xytext=offset, textcoords="offset points", color=color,
                        fontsize=15, ha="left" if name == "Start" else "right", zorder=7)

    prm_report = prm_figure(rrt)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))
    fig.subplots_adjust(left=.08, right=.98, bottom=.31, top=.87, wspace=.34)
    for ax in axes:
        cspace(ax)
        poly(ax, [rrt.Q_START, rrt.Q_GOAL], color=RED, ls="--", lw=1.5, zorder=2)
    halo = "#adc9ec"
    poly(axes[0], connect["path"], color=halo, lw=9, zorder=2, solid_capstyle="round")
    for key, color in [("A", BLUE), ("B", GREEN)]:
        tree = connect[key]
        for idx, parent in enumerate(tree["parents"]):
            if parent >= 0: poly(axes[0], [tree["nodes"][parent], tree["nodes"][idx]], color=color, lw=2, zorder=3)
    meet = np.degrees(connect["meet"]["q"])
    axes[0].plot(*meet, "o", mfc="none", mec="black", ms=11, mew=1.8, zorder=8)
    axes[0].annotate("Meet", xy=meet, xytext=(24, 30), textcoords="offset points", fontsize=16,
                     arrowprops={"arrowstyle": "-", "color": "black"})
    one = saved["rrt"]; one_nodes = np.asarray(one["nodes"])
    for idx, parent in enumerate(one["parents"]):
        if parent >= 0: poly(axes[1], one_nodes[[parent, idx]], color="#999", lw=1, zorder=2)
    poly(axes[1], one_nodes[one["path"]], color=halo, lw=9, zorder=3, solid_capstyle="round")
    poly(axes[1], one_nodes[one["path"]], color=BLUE, lw=2.6, zorder=4)
    axes[0].set_title("RRT-Connect · 14 iterations", pad=15)
    axes[1].set_title("One tree · 42 iterations", pad=15)
    handles = [Line2D([], [], color=BLUE, lw=2, label="Start tree"),
               Line2D([], [], color=GREEN, lw=2, label="Goal tree"),
               Line2D([], [], color=halo, lw=7, label="Route halo"),
               Line2D([], [], color="#999", lw=1.4, label="Other tree edges"),
               Line2D([], [], color=RED, ls="--", label="Invalid direct edge"),
               Patch(color=LIGHT, label="Forbidden samples")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=15,
               bbox_to_anchor=(.5, .015))
    finish(fig, "connect-slide.png")
    return {**prm_report,
            "connect_iterations": connect["iterations"], "connect_attempts": connect["attempts"],
            "connect_checked": connect["checked_configurations"], "connect_source": str(source.relative_to(ROOT))}


def note_unwritten():
    if OUT is None:
        print("no --out given: nothing written" if not PREVIEW else
              "no --out given: only the --preview-dir previews were written")


def main():
    global PREVIEW, OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help=f"Write the PNGs to this directory (the deck uses {FIG}).")
    parser.add_argument("--check", action="store_true", help="Solve, assert and draw; write no deck PNG unless --out is given.")
    parser.add_argument("--preview-dir", type=Path, help="Also render directly from Matplotlib at each intended slide width.")
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--prm-only", action="store_true", help="Render and verify only the 400-sample lecture roadmap.")
    only.add_argument("--komo-only", action="store_true",
                      help="Render and verify only the KOMO cost, path and band slides and the between-knot slide.")
    args = parser.parse_args()
    PREVIEW = args.preview_dir
    if PREVIEW: PREVIEW.mkdir(parents=True, exist_ok=True)
    OUT = args.out
    if OUT is not None: OUT.mkdir(parents=True, exist_ok=True)
    if args.prm_only:
        rrt = load_module("review_rrt", "modules/motion-planning/rrt_two_link.py")
        print(json.dumps(prm_figure(rrt), indent=2))
        note_unwritten()
        if args.check: print("check passed")
        return
    km = load_module("review_retained_komo", "lectures/04-D2-motion-planning/code/komo_two_link.py")
    rrt = None if args.komo_only else load_module("review_rrt", "modules/motion-planning/rrt_two_link.py")
    answers = solve_retained(km)
    report = {"cost_comparison": cost_comparison(km, answers),
              "solutions": solutions(km, answers), "sparsity": sparsity(km, answers),
              "between_knots": between_knots(browser_preset())}
    if not args.komo_only:
        report.update({"rrt": rrt_figures(rrt), "staged_layouts": staged_figures(rrt)})
    print(json.dumps(report, indent=2))
    note_unwritten()
    if args.check: print("check passed")


if __name__ == "__main__":
    main()
