"""Slide-sized figures proposed for Deck 06's refinement block (2026-11-01), from the existing numerical sources.

Three figures, each from staged_two_link.py stage 4 (the whole-link worked example [W], RAI's loop through the NumPy
port, exact starts) and the sampling demo's motion check (rrt_two_link.py):
  refine-34-initialize.png    the seed-7 sampled route, the 21 knots placed along it and their new straight connections;
                              a 6° close-up of connection 14->15, which cuts the route's corner 0.30 mm into the buffer
  refine-35-compare.png       the direct and sampled initial guesses and their two results under the tighter-accuracy
                              settings (every constraint row times 10, tight stops), which coincide in this example
  refine-36-cost-clearance.png  the sampled initial knots against their optimized result: acceleration cost and how far
                              the deepest posture between knots comes inside the 5 mm buffer (two panels, one measure each)
Every plotted number is computed here and asserted: the optimized knots, cost and between-knot deficit against the
native RAI record of the same cases (modules/opt-explorer/fixtures/refine_rai.json, written in the Colima container);
the initial cost and the route and initial-knot checks against the Python reference
(modules/opt-explorer/fixtures/refine_reference.json); the footnote states only an order of magnitude for the knot
residual, which differs between engines in its first digit. No existing figure is replaced: the
files are new and are written only with --out.

Run:  .venv/bin/python modules/motion-planning/refine_figures.py                 print the numbers and the checks
      .venv/bin/python modules/motion-planning/refine_figures.py --out DIR       also write the three figures to DIR
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rrt_two_link as R        # noqa: E402
import staged_two_link as S     # noqa: E402

DEG = R.DEG
NATIVE = HERE.parents[0] / "opt-explorer" / "fixtures" / "refine_rai.json"
C = {"start": "#1f5fbf", "goal": "#2a7d2a", "route": "#6a3d9a", "direct": "#b00020", "init": "#333333", "result": "#0b6fb8",
     "other": "#e08a00", "bad": "#b00020", "disc": "#a8a8a8", "band": "#f2d39b", "ink": "#222222", "muted": "#666666"}


def compute():
    with contextlib.redirect_stdout(io.StringIO()):
        rr = R.rrt(7, 3000, "links")
        route, _ = S.shortcut([rr["nodes"][i] for i in rr["path"]], 7, 100)
        x_line, x_sc = S.linear_init(), S.shortcut_init(7)
        runs = {cid: S.solve_komo(cid) for cid in ("W-auglag-tight-line-s10", "W-auglag-tight-shortcut-s10")}
    k_sc = [tuple(q) for q in S.knots(x_sc)]
    k_line = [tuple(q) for q in S.knots(x_line)]
    res = {cid: [tuple(q) for q in S.knots(z)] for cid, (z, *_rest) in runs.items()}
    v_route, v_init, v_res = S.validate_path(route), S.validate_path(k_sc), S.validate_path(res["W-auglag-tight-shortcut-s10"])
    d = {"route": route, "k_sc": k_sc, "k_line": k_line, "res": res, "v_route": v_route, "v_init": v_init, "v_res": v_res,
         "cost_init": S.cost(x_sc), "cost_res": S.cost(runs["W-auglag-tight-shortcut-s10"][0]),
         "deficit_init_mm": max(0.0, -v_init["deepest"]["c"]) * 1000, "deficit_res_mm": max(0.0, -v_res["deepest"]["c"]) * 1000}
    g_res, _ = S.ineq_constraints(runs["W-auglag-tight-shortcut-s10"][0])
    d["knot_deficit_res_mm"] = max(0.0, g_res.max()) * 1000
    d["same_deg"] = max(R.dist(a, b) for a, b in zip(res["W-auglag-tight-line-s10"], res["W-auglag-tight-shortcut-s10"])) / DEG
    return d


def check_against_native(d):
    """The plotted numbers against RAI's own runs of the same cases (both programs, exact starts)."""
    nat = json.loads(NATIVE.read_text())["cases"]
    fails = []
    for cid in ("W-auglag-tight-line-s10", "W-auglag-tight-shortcut-s10"):
        for prog in ("twin", "komo"):
            rec = nat[f"{cid}|{prog}"]
            dx = max(abs(a - b) for q, p in zip(d["res"][cid][1:], np.array(rec["x"]).reshape(-1, 2)) for a, b in zip(q, p))
            if dx > 1e-7:
                fails.append(f"{cid} [{prog}]: the plotted knots differ from RAI's by {dx:.1e} rad")
    ref = json.loads((HERE.parents[0] / "opt-explorer" / "fixtures" / "refine_reference.json").read_text())
    if abs(d["cost_init"] - ref["init_cost"]["sampled"]) > 1e-12 * d["cost_init"]:
        fails.append("the initial cost differs from the Python reference")
    for name, v in (("sampled knots", d["v_init"]), ("route", d["v_route"])):
        rv = ref["checks"][name]
        if (rv["valid"], rv["deepest"]["edge"], rv["deepest"]["j"]) != (v["valid"], v["deepest"]["edge"], v["deepest"]["j"]) or abs(rv["deepest"]["c"] - v["deepest"]["c"]) > 1e-12:
            fails.append(f"the {name} check differs from the Python reference")
    for prog in ("twin", "komo"):
        if nat[f"W-auglag-tight-shortcut-s10|{prog}"]["residuals_at_x"]["max_knot_deficit_mm"] >= 1e-6:
            fails.append(f"native {prog}: the knot clearance deficits are not below 1e-6 mm")
    r = nat["W-auglag-tight-shortcut-s10|komo"]["residuals_at_x"]
    if abs(d["cost_res"] - r["cost"]) > 1e-7 * r["cost"]:
        fails.append(f"cost {d['cost_res']} against RAI's {r['cost']}")
    if abs(d["deficit_res_mm"] - r["between_knot_deficit_mm"]) > 1e-5:
        fails.append(f"between-knot deficit {d['deficit_res_mm']} against RAI's {r['between_knot_deficit_mm']}")
    if nat["W-auglag-tight-shortcut-s10|komo"]["nominal_x"] != [v for q in d["k_sc"][1:] for v in q]:
        fails.append("the sampled initial knots differ from the start RAI received")
    return fails


def figures(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.patheffects import withStroke
    HALO = [withStroke(linewidth=4, foreground="white")]
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 15, "axes.edgecolor": "#888888", "axes.titlepad": 10})
    written = []

    def raster(ax, x0, x1, y0, y1, step):
        q1, q2 = np.arange(x0, x1 + step / 2, step), np.arange(y0, y1 + step / 2, step)
        c = np.array([[R.clearance((a * DEG, b * DEG), "links")[0] for a in q1] for b in q2])
        img = np.ones(c.shape + (4,))
        img[..., 3] = 0
        inside, buffer = c < -R.MARGIN, (c >= -R.MARGIN) & (c < 0)
        img[inside] = matplotlib.colors.to_rgba(C["disc"])
        img[buffer] = matplotlib.colors.to_rgba(C["band"])
        ax.imshow(img, extent=(x0 - step / 2, x1 + step / 2, y0 - step / 2, y1 + step / 2), origin="lower", interpolation="nearest", zorder=0)
        ax.set(xlim=(x0, x1), ylim=(y0, y1))
        ax.set_aspect("equal")
        ax.grid(color="#e6e6e6", linewidth=.6, zorder=1)
        ax.set_axisbelow(True)

    def line(ax, pts, **kw):
        ax.plot([p[0] / DEG for p in pts], [p[1] / DEG for p in pts], **kw)

    def ends(ax):
        ax.plot(S.Q_START[0] / DEG, S.Q_START[1] / DEG, "o", color=C["start"], markersize=11, zorder=8)
        ax.plot(S.Q_GOAL[0] / DEG, S.Q_GOAL[1] / DEG, "D", color=C["goal"], markersize=10, zorder=8)
        box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.9)
        ax.annotate("start", (S.Q_START[0] / DEG, S.Q_START[1] / DEG), xytext=(10, -20), textcoords="offset points", color=C["start"], bbox=box, zorder=9)
        ax.annotate("goal", (S.Q_GOAL[0] / DEG, S.Q_GOAL[1] / DEG), xytext=(-6, 12), textcoords="offset points", color=C["goal"], bbox=box, zorder=9)

    xl, yl = "$q_1$, shoulder angle [deg]", "$q_2$, elbow angle [deg]"
    regions = [Patch(color=C["disc"], label="a link enters the disc"), Patch(color=C["band"], label="a link is within 5 mm")]

    # ---------------------------------------------------------------- slide 34: prepare the initial guess
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    raster(axes[0], -20, 40, 35, 95, 0.25)
    line(axes[0], d["route"], color=C["route"], linewidth=6, alpha=.55, solid_capstyle="round", zorder=3)
    line(axes[0], d["k_sc"], color=C["init"], linewidth=1.2, marker="o", markersize=4.5, zorder=4)
    ends(axes[0])
    dq = d["v_init"]["deepest"]["q"]
    axes[0].add_patch(plt.Rectangle((dq[0] / DEG - 3, dq[1] / DEG - 3), 6, 6, fill=False, edgecolor=C["ink"], linewidth=1.2, linestyle="--", zorder=6))
    axes[0].set(xlabel=xl, ylabel=yl, title="Sampled route and 21 knots")
    raster(axes[1], dq[0] / DEG - 3, dq[0] / DEG + 3, dq[1] / DEG - 3, dq[1] / DEG + 3, 0.02)
    line(axes[1], d["route"], color=C["route"], linewidth=6, alpha=.55, solid_capstyle="round", zorder=3)
    line(axes[1], d["k_sc"], color=C["init"], linewidth=1.6, marker="o", markersize=7, zorder=4)
    e = d["v_init"]["deepest"]["edge"]
    for t in (e, e + 1):
        q = d["k_sc"][t]
        axes[1].annotate(f"knot {t}", (q[0] / DEG, q[1] / DEG), xytext=(10, 10), textcoords="offset points", fontsize=14, path_effects=HALO, zorder=9)
    axes[1].plot(dq[0] / DEG, dq[1] / DEG, "x", color=C["bad"], markersize=14, markeredgewidth=3, zorder=8, path_effects=HALO)
    axes[1].annotate(f"{d['deficit_init_mm']:.2f} mm inside\nthe 5 mm buffer", (dq[0] / DEG, dq[1] / DEG), xytext=(.05, .08), textcoords="axes fraction",
                     color=C["bad"], fontsize=15, arrowprops=dict(arrowstyle="-", color=C["bad"], linewidth=.9), path_effects=HALO, zorder=9)
    axes[1].set(xlabel=xl, title="Close-up: a new connection cuts a corner")
    fig.legend(handles=regions + [Line2D([], [], color=C["route"], linewidth=6, alpha=.55, label="sampled route (passes the check)"),
                                  Line2D([], [], color=C["init"], marker="o", linewidth=1.2, label="knots and their straight connections"),
                                  Line2D([], [], color=C["bad"], marker="x", linewidth=0, markeredgewidth=3, markersize=11, label="deepest scanned posture")],
               loc="lower center", ncol=3, frameon=False, fontsize=14, columnspacing=1.4, handlelength=2.2)
    fig.tight_layout(rect=[0, .16, 1, .97])
    fig.savefig(Path(out) / "refine-34-initialize.png", dpi=180)
    plt.close(fig)
    written.append("refine-34-initialize.png")

    # ---------------------------------------------------------------- slide 35: compare initial guesses
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    raster(ax, -20, 40, 35, 95, 0.25)
    line(ax, [S.Q_START, S.Q_GOAL], color=C["direct"], linewidth=1.8, linestyle="--", zorder=3)
    line(ax, d["k_sc"], color=C["init"], linewidth=1.2, linestyle=":", marker="o", markersize=3.5, zorder=4)
    line(ax, d["res"]["W-auglag-tight-shortcut-s10"], color=C["result"], linewidth=3.2, marker="o", markersize=7, zorder=5)
    line(ax, d["res"]["W-auglag-tight-line-s10"], color=C["other"], linewidth=0, marker="o", markersize=15, markerfacecolor="none", markeredgewidth=2.0, zorder=6)
    ends(ax)
    ax.set(xlabel=xl, ylabel=yl)
    ax.legend(handles=regions + [Line2D([], [], color=C["direct"], linestyle="--", linewidth=1.8, label="direct initial guess"),
                                 Line2D([], [], color=C["init"], linestyle=":", marker="o", markersize=4, linewidth=1.2, label="sampled initial guess"),
                                 Line2D([], [], color=C["result"], marker="o", linewidth=3.2, label="result from the sampled guess"),
                                 Line2D([], [], color=C["other"], marker="o", markerfacecolor="none", markeredgewidth=2, markersize=12, linewidth=0, label="result from the direct guess")],
              loc="center left", bbox_to_anchor=(1.03, .5), frameon=False, fontsize=15, handlelength=2.4, labelspacing=.9)
    fig.tight_layout()
    fig.savefig(Path(out) / "refine-35-compare.png", dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    written.append("refine-35-compare.png")

    # ---------------------------------------------------------------- slide 36: cost and clearance, one measure per panel
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    cats = ["sampled\ninitial knots", "optimized\nknots"]
    for ax, vals, title, fmt, notes in (
            (axes[0], [d["cost_init"], d["cost_res"]], "Acceleration cost", "{:.2f}", None),
            (axes[1], [d["deficit_init_mm"], d["deficit_res_mm"]], "Deepest posture inside the\n5 mm buffer, between knots [mm]", "{:.2f} mm",
             [f"knots {d['v_init']['deepest']['edge']}–{d['v_init']['deepest']['edge'] + 1}", f"knots {d['v_res']['deepest']['edge']}–{d['v_res']['deepest']['edge'] + 1}"])):
        bars = ax.bar([0, 1], vals, width=.5, color=C["result"], zorder=3)
        top = max(vals) * 1.3
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, v + top * .02, fmt.format(v), ha="center", va="bottom", fontsize=17, color=C["ink"], zorder=4)
            if notes:
                ax.text(b.get_x() + b.get_width() / 2, v + top * .11, notes[i], ha="center", va="bottom", fontsize=13, color=C["muted"], zorder=4)
        ax.set_xticks([0, 1], cats)
        ax.set_ylim(0, top)
        ax.set_xlim(-.6, 1.6)
        ax.set_title(title)
        ax.grid(axis="y", color="#e6e6e6", linewidth=.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    assert 0 < d["knot_deficit_res_mm"] < 1e-6, "the footnote's order-of-magnitude statement no longer holds"
    fig.text(.5, .015, f"Knot clearance deficits are below $10^{{-6}}$ mm; the {d['deficit_res_mm']:.2f} mm deficit lies between knots.",
             ha="center", fontsize=13, color=C["muted"])
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(Path(out) / "refine-36-cost-clearance.png", dpi=180)
    plt.close(fig)
    written.append("refine-36-cost-clearance.png")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write the three figures to this directory")
    a = ap.parse_args()
    d = compute()
    print(f"route: {len(d['route']) - 1} connections, {d['v_route']['length_deg']:.2f}°, valid {d['v_route']['valid']}, smallest clearance "
          f"{(d['v_route']['deepest']['c'] + R.MARGIN) * 1000:.2f} mm from the disc surface")
    print(f"sampled initial knots: cost {d['cost_init']:.4f}; deepest posture {d['deficit_init_mm']:.4f} mm inside the buffer between knots "
          f"{d['v_init']['deepest']['edge']} and {d['v_init']['deepest']['edge'] + 1}")
    print(f"optimized (W-auglag-tight-shortcut-s10, exact start): cost {d['cost_res']:.4f}; largest knot deficit {d['knot_deficit_res_mm']:.2e} mm; deepest posture "
          f"{d['deficit_res_mm']:.4f} mm inside the buffer between knots {d['v_res']['deepest']['edge']} and {d['v_res']['deepest']['edge'] + 1}")
    print(f"the two tighter-accuracy results differ by at most {d['same_deg']:.1e}° at a knot")
    fails = check_against_native(d)
    for f in fails:
        print("FAIL", f)
    print("native RAI check:", "passed (knots within 1e-7 rad on both programs, cost and deficit agree)" if not fails else f"{len(fails)} failure(s)")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        print("wrote", ", ".join(figures(d, a.out)), "to", a.out)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
