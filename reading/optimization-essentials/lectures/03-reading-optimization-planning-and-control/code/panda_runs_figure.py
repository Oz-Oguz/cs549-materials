"""Chapter 02 essentials, Figure 2.3: the four damped Gauss-Newton runs on the Panda, drawn for the
reading's text width with larger labels.

The data come from drake/02-gauss-newton-ik/gn_ik_panda.py (its `build`, `make_fk` and `record_runs`),
the same runs that the Deck 04 slide figure fig/drake-gn-replay.png shows. Only the layout differs:
short column titles, larger fonts and no plot title, because the reading's caption defines the three
quantities. The script checks the update counts and run 3b's final error before drawing.

Run from the repository root:  .venv/bin/python lectures/03-reading-optimization-planning-and-control/code/panda_runs_figure.py
Writes lectures/03-reading-optimization-planning-and-control/fig/panda_gn_runs.png.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "drake/02-gauss-newton-ik"))
import gn_ik_panda as panda  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fig" / "panda_gn_runs.png"
TITLES = {"run1": "Run 1\nready start, nearby target",
          "run2": "Run 2\nstretched start, weak direction",
          "run3a": "Run 3a\nunreachable, full steps",
          "run3b": "Run 3b\nunreachable, Armijo"}


def main():
    _, plant, pc, _, panda_instance, ee = panda.build()
    rec = panda.record_runs(*panda.make_fk(plant, pc, panda_instance, ee))
    runs = rec["runs"]
    counts = {k: runs[k]["n_updates"] for k in runs}
    assert counts == {"run1": 4, "run2": 18, "run3a": 40, "run3b": 40}, counts
    assert abs(runs["run3b"]["iterates"][-1]["residual"] - 0.24621) < 1e-5

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(3, 4, figsize=(8.6, 6.4), sharex="col")
    for c, key in enumerate(["run1", "run2", "run3a", "run3b"]):
        run = runs[key]
        it, ups = run["iterates"], run["accepted_updates"]
        # Runs 3a and 3b stay within one decade of task error (3a also of prediction gap): linear axes there.
        err_plot = axes[0, c].plot if key in ("run3a", "run3b") else axes[0, c].semilogy
        gap_plot = axes[2, c].plot if key == "run3a" else axes[2, c].semilogy
        err_plot([x["i"] for x in it], [x["residual"] for x in it], "o-", color="#7a1f3d", ms=2.5)
        axes[1, c].plot([u["i"] for u in ups], [u["step_norm"] for u in ups], "o-", color="#1f6f8b", ms=2.5)
        gap_plot([u["i"] for u in ups], [max(u["gap"], 1e-16) for u in ups], "o-", color="#b8860b", ms=2.5)
        axes[0, c].set_title(TITLES[key])
        axes[2, c].set_xlabel("accepted update")
        for r in range(3):
            axes[r, c].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("task error (m)")
    axes[1, 0].set_ylabel("joint step (rad)")
    axes[2, 0].set_ylabel("prediction gap (m)")
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}; update counts {counts}; run 3b final error "
          f"{runs['run3b']['iterates'][-1]['residual']:.5f} m")


if __name__ == "__main__":
    main()
