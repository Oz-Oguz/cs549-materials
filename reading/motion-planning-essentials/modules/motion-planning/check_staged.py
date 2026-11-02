"""Regression check of the staged figures' headline numbers (modules/motion-planning/staged.md), seed 7.
Runs staged_two_link.py's computations in memory and exits 1 on any drift.

The KOMO stage [W] is checked against the recorded real-RAI fixtures, never against remembered numbers. The port's
exact-start runs of W-auglag-default-line (RAI's untouched options) and W-auglag-tight-line-s10 (the converged
reference: every constraint row times 10, tight stops) are compared with the records of the same case IDs in
fixtures/stage4_rai.json. The seed sweep fixtures/stage4_seed_sweep.json is read, not re-run. A missing fixture or a
missing case is a failure.
Run:  .venv/bin/python modules/motion-planning/check_staged.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrt_two_link as R
import staged_two_link as S

DEG = R.DEG
failed = 0


def check(msg, ok):
    global failed
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        failed += 1


near = lambda a, b, tol: abs(a - b) <= tol

rr = R.rrt(7, 3000, "links")
pts = [rr["nodes"][i] for i in rr["path"]]
check("demo RRT: iteration 42, 8 edges, 78.71°", rr["iterations"] == 42 and len(pts) == 9 and near(S.path_length(pts) / DEG, 78.71, 0.01))

P = S.prm(7)
v1 = S.validate_path([P["nodes"][i] for i in P["path"]])
check("PRM: 59 valid of 60, 187 accepted of 191, Dijkstra settles 7 nodes before the goal, route of 2 edges, 70.05°, valid, 17.6 mm",
      P["n_valid"] == 59 and P["tested"] == 191 and P["accepted"] == 187 and P["settled_before_goal"] == 7 and len(P["path"]) == 3 and near(v1["length_deg"], 70.05, 0.01) and v1["valid"] and near((v1["deepest"]["c"] + S.MARGIN) * 1000, 17.6, 0.1))

C = S.rrt_connect(7)
v2 = S.validate_path(C["path"])
check("RRT-Connect: iteration 14, 27 attempts (14 added, 13 blocked), 186 configurations evaluated, 7 + 9 nodes, 7 edges, 69.47°, valid, 12.9 mm",
      C["outcome"] == "connected" and C["iterations"] == 14 and C["attempts"] == 27 and C["added"] == 14 and C["checked_configurations"] == 186 and len(C["A"]["nodes"]) == 7 and len(C["B"]["nodes"]) == 9 and len(C["path"]) == 8 and near(v2["length_deg"], 69.47, 0.01) and v2["valid"] and near((v2["deepest"]["c"] + S.MARGIN) * 1000, 12.9, 0.1))
rc = S.rrt_counts(7)
check("one-tree RRT counts: 43 attempts, 291 configurations evaluated", rc["attempts"] == 43 and rc["checked_configurations"] == 291)
fewer = sum(S.rrt_connect(sd)["iterations"] < S.rrt_counts(sd)["iterations"] for sd in range(1, 13))
check("seeds 1 to 12: RRT-Connect needs fewer iterations in 11 of 12", fewer == 11)

after, log = S.shortcut(pts, 7, 100)
v3 = S.validate_path(after)
acc = sum(a["result"] == "accepted" for a in log)
check("shortcut: 37 accepted, 38 chords rejected, 0 candidates rejected, 25 same-edge; 10 edges, 58.34°, valid at 1°, 5.1 mm",
      acc == 37 and sum(a["result"] == "chord rejected" for a in log) == 38 and sum(a["result"] == "candidate rejected" for a in log) == 0 and len(after) == 11 and near(v3["length_deg"], 58.34, 0.01) and v3["valid"] and near((v3["deepest"]["c"] + S.MARGIN) * 1000, 5.1, 0.1))
check("shortcut: every accepted edit's candidate was validated (37 validations, 2241 configurations) apart from the 735 chord evaluations",
      all(a["candidate_evaluated"] == a["candidate_scheduled"] for a in log if a["result"] == "accepted") and sum(a.get("candidate_evaluated", 0) for a in log) == 2241 and sum(a.get("chord_evaluated", 0) for a in log) == 735)
# the complete-candidate rule: the cases of Codex's review STG-S01 (seed, attempt, failing edge, check, depth inside the ring in mm)
for sd, t_at, edge, j, n, mm in ((15, 35, 4, 5, 11, 0.011083), (46, 85, 2, 1, 2, 0.013477), (71, 74, 3, 2, 3, 0.002330)):
    rs = R.rrt(sd, 3000, "links"); pts_s = [rs["nodes"][i] for i in rs["path"]]
    after_s, log_s = S.shortcut(pts_s, sd, 100)
    cr = [a for a in log_s if a["result"] == "candidate rejected"]
    first = cr[0] if cr else None
    vs = S.validate_path(after_s)
    check(f"seed {sd}: the chord of attempt {t_at} is valid but the candidate fails on edge {edge}->{edge + 1}, check {j} of {n}, {mm} mm inside the ring, so it is rejected and the final path is valid at 1°",
          first is not None and first["t"] == t_at and first["fail_edge"] == edge and first["fail_j"] == j and first["fail_n"] == n and near(-first["fail_c"] * 1000, mm, 1e-4) and vs["valid"])
saved = R.SUB
R.SUB = 0.05 * DEG
try:
    v3f = S.validate_path(after)
finally:
    R.SUB = saved
check("shortcut: the 0.05° diagnostic finds 0.1 mm inside the ring (invalid) over 1004 configurations",
      (not v3f["valid"]) and v3f["scheduled"] == 1004 and near(-v3f["deepest"]["c"] * 1000, 0.1, 0.05))
check("the direct edge is never an accepted shortcut", all(not (a["result"] == "accepted" and a["edges"] == 1) for a in log))

# ---- the KOMO stage [W]: the port's exact-start runs against the real-RAI fixtures of the same case IDs
FIX = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name):
    path = FIX / name
    if not path.exists():
        check(f"fixture {path.name} exists (recorded in the container by stage4_fixture_rai.py)", False)
        return {}
    return json.loads(path.read_text())["cases"]


def rel(a, b):
    return abs(a - b) / abs(b)


cases = load_fixture("stage4_rai.json")
DEF, REF = "W-auglag-default-line", "W-auglag-tight-line-s10"
for cid in (DEF, REF):
    if cid not in cases and cases:
        check(f"{cid} is recorded in stage4_rai.json", False)
if DEF in cases:
    rec = cases[DEF]
    z, lam, _, info = S.solve_komo(DEF)
    rep = S.physical_report(DEF, z, lam)
    dx = float(np.abs(z - np.array(rec["x"])).max())
    check(f"{DEF} (exact start): {info['evals']} evaluations (RAI {rec['ret']['evals']}), sos {info['errors_buffered']['sos']:.9f} (RAI {rec['ret']['sos']:.9f}, "
          f"relative {rel(info['errors_buffered']['sos'], rec['ret']['sos']):.1e} <= 1e-9), max |x - ret.x| {dx:.1e} rad (<= 1e-8), RAI's feasible flag {rec['ret']['feasible']}",
          info["evals"] == rec["ret"]["evals"] == 28 and rel(info["errors_buffered"]["sos"], rec["ret"]["sos"]) <= 1e-9 and dx <= 1e-8)
    check(f"{DEF}: knot deficit {rep['knot_deficit_mm']:.3f} mm at knot {rep['worst_row'][0]} link {rep['worst_row'][1]} (the record {rec['residuals_at_x']['max_knot_deficit_mm']:.3f} mm) "
          f"beside RAI's feasible flag {rec['ret']['feasible']}; between knots {rep['between_knot_deficit_mm']:.3f} mm inside the ring",
          near(rep["knot_deficit_mm"], 2.003, 5e-4) and near(rep["knot_deficit_mm"], rec["residuals_at_x"]["max_knot_deficit_mm"], 1e-6) and rec["ret"]["feasible"])
if REF in cases:
    rec = cases[REF]
    z, lam, _, info = S.solve_komo(REF)
    rep = S.physical_report(REF, z, lam)
    act = {(t, l): v for t, l, v in rep["active"]}
    check(f"{REF} (exact start): sos {info['errors_buffered']['sos']:.9f} (RAI {rec['ret']['sos']:.9f}, relative {rel(info['errors_buffered']['sos'], rec['ret']['sos']):.1e} <= 1e-6; "
          f"16.138216); {info['evals']} evaluations (RAI {rec['ret']['evals']}; check_rai_nlp.py qualifies the difference as a rounding-level Wolfe decision)",
          rel(info["errors_buffered"]["sos"], rec["ret"]["sos"]) <= 1e-6 and near(info["errors_buffered"]["sos"], 16.138216, 5e-7))
    check(f"{REF}: knot deficit {rep['knot_deficit_mm']:.1e} mm (< 1e-6 mm); between knots {rep['between_knot_deficit_mm']:.3f} mm inside the ring "
          f"(edge {rep['validation']['deepest']['edge']}->{rep['validation']['deepest']['edge'] + 1})",
          rep["knot_deficit_mm"] < 1e-6 and near(rep["between_knot_deficit_mm"], 1.844, 1e-3) and not rep["validation"]["valid"])
    check(f"{REF}: physical multipliers (10 x RAI's dual) " + ", ".join(f"knot {t} link {l}: {v:.3f} per m" for (t, l), v in act.items())
          + " = 146.955 and 87.136 at (12, 2) and (13, 2) within 1e-2",
          sorted(act) == [(12, 2), (13, 2)] and near(act[(12, 2)], 146.955, 1e-2) and near(act[(13, 2)], 87.136, 1e-2))
sweep = load_fixture("stage4_seed_sweep.json")
if sweep:
    tight = {cid: r for cid, r in sweep.items() if "-tight-" in cid}
    worst = max((abs(r["ret"]["sos"] - 16.138216) for r in tight.values()), default=float("inf"))
    check(f"stage4_seed_sweep.json (read, not re-run): {len(sweep)} cases (24 expected); the {len(tight)} W-auglag-tight-shortcut-s10-rrt<n>-seed1 cases end within "
          f"{worst:.1e} of sos 16.138216 (<= 1e-6)",
          len(sweep) == 24 and len(tight) == 12 and worst <= 1e-6)
print(f"{'all checks passed' if not failed else str(failed) + ' checks FAILED'}")
sys.exit(1 if failed else 0)
