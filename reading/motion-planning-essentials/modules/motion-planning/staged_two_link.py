"""Staged figures and numbers for Deck 06's sampling block, on the sampling demo's exact scene and contract
(modules/motion-planning/rrt_two_link.py: the two-link arm, whole links as segments, the 5 mm requirement from the
disc surface, the joint box [-pi, pi]^2 without wrapping, the fixed positive-elbow goal, extensions of at most 10 deg,
edge checks with configurations at most 1 deg apart per joint and the first invalid one rejecting the edge,
mulberry32 seeded with 7, restarted from the seed at the start of every stage).

Four stages, each a reproducible data set and a static figure:
  1. a probabilistic roadmap (PRM) and the shortest path through it (Dijkstra),
  2. RRT-Connect: two trees, one from each endpoint, against the one-tree RRT of the demo (seed 7, then seeds 1-12),
  3. shortcutting of the demo's RRT path: the chord is checked, then the whole candidate path is validated with the
     same 1 deg rule before it replaces the path (the split edges get new check grids), and the result is revalidated,
  4. the sampled path as the initialization of a KOMO problem whose clearance rows use the same whole-link geometry
     and the same 5 mm requirement, against the straight-line initialization. This is the whole-link worked example [W]
     of Chapter 03 and Deck 06. The solver is RAI's own loop (ry.NLP_Solver, augmented Lagrangian with RAI's default
     penalty schedule mu = 1, 5, 25, ... up to 1e4, damped Gauss-Newton steps capped at 0.2 rad, RAI's Wolfe line
     search) through its NumPy port lectures/04-D2-motion-planning/code/rai_nlp.py. Every run is named by a case ID of
     that port's grammar and starts exactly from the nominal vector ("exact start"); RAI's own seeded calls exist only
     in the recorded fixtures (fixtures/stage4_rai.json, fixtures/stage4_seed_sweep.json). The main runs use RAI's
     untouched options, so they stop on RAI's default step test (a step below 0.01 rad) and their physical residuals
     are reported in task units. The converged reference, named W-auglag-tight-...-s10, keeps RAI's schedule but
     multiplies every constraint row by 10 and uses the tight stops stopTolerance 1e-8, stopFTolerance 1e-12 and
     stopEvals 5000. KOMO scales the feature, so the physical multiplier of a scaled row is 10 times RAI's dual.
Every path, before and after any modification, is revalidated with the demo's edge check (rrt_two_link.check_edge,
counting the evaluations of the repeated edge tests, each failing edge stopping at its first invalid configuration)
and scanned completely for its smallest clearance (rrt_two_link.deepest_along). No RAI execution happens here.

Run:  .venv/bin/python modules/motion-planning/staged_two_link.py                            print every number (a few seconds)
      .venv/bin/python modules/motion-planning/staged_two_link.py --figures --out DIR         also write the four staged-*.png to DIR
      .venv/bin/python modules/motion-planning/staged_two_link.py --figures --stage 4 --out DIR   write only staged-4-komo.png
      .venv/bin/python modules/motion-planning/staged_two_link.py --json out.json             also dump the data
--figures without --out writes nothing and says so: new figures are staged outside modules/motion-planning/fig/
(planning/rai-defaults-2026-10-30/staging-fig/) until they are swapped in with their text.
"""
import argparse
import heapq
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrt_two_link as R                                                   # the demo's scene and contract

KOMO_CODE = Path(__file__).resolve().parents[2] / "lectures/04-D2-motion-planning/code"


def _load_rai_nlp():
    """RAI's NLP_Solver ported to NumPy, loaded by path from Deck 06's code folder (one module object per process)."""
    if "rai_nlp" in sys.modules:
        return sys.modules["rai_nlp"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("rai_nlp", KOMO_CODE / "rai_nlp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rai_nlp"] = mod
    spec.loader.exec_module(mod)
    return mod


rai_nlp = _load_rai_nlp()

DEG = R.DEG
STEP, SUB, P_GOAL, MARGIN = R.STEP, R.SUB, R.P_GOAL, R.MARGIN
Q_START, Q_GOAL = R.Q_START, R.Q_GOAL
FIG = Path(__file__).resolve().parent / "fig"                              # where staged.md embeds the figures; written only by hand from --out


# ----------------------------------------------------------------------------- shared helpers
def deg(q):
    return R.deg(q)


def path_length(pts):
    return sum(R.dist(a, b) for a, b in zip(pts, pts[1:]))


def validate_path(pts, checker="links"):
    """The demo's contract on a list of configurations: the repeated edge tests (each edge checked in order at <= 1 deg
    per joint, the first invalid configuration rejecting it; `evaluated` counts what the tests evaluated) and a complete
    scan of every scheduled configuration of every edge, j = 0..n, for the smallest clearance (`deepest`)."""
    evaluated, scheduled, fail, deepest = 0, 0, None, None
    for e, (a, b) in enumerate(zip(pts, pts[1:])):
        ok, n, j, q_fail, part, c_fail, _ = R.check_edge(a, b, checker)
        evaluated += n if ok else j
        scheduled += n
        if not ok and fail is None:
            fail = {"edge": e, "j": j, "n": n, "q": q_fail, "part": part, "c": c_fail}
        c, jj, nn, qq, pp = R.deepest_along(a, b, checker)
        if deepest is None or c < deepest["c"]:
            deepest = {"c": c, "edge": e, "j": jj, "n": nn, "q": qq, "part": pp}
    return {"valid": fail is None, "fail": fail, "evaluated": evaluated, "scheduled": scheduled, "deepest": deepest,
            "length_deg": path_length(pts) / DEG}


def clearance_words(c):
    """c is the clearance beyond the 5 mm requirement; print the clearance from the disc surface or the depth."""
    return f"{(c + MARGIN) * 1000:.1f} mm from the disc surface" if c >= 0 else R.depth_words(c)


def revalidation_words(v, what="configurations"):
    """The demo's wording for a validation result."""
    if v["valid"]:
        return (f"valid: the repeated edge tests evaluate all {v['scheduled']} scheduled {what}; the complete scan finds the smallest clearance "
                f"{clearance_words(v['deepest']['c'])}")
    f = v["fail"]
    return (f"INVALID: across all edges the repeated edge tests evaluate {v['evaluated']} of {v['scheduled']} scheduled {what}, each failing edge "
            f"stopping at its first invalid one; the first failure in path order is edge {f['edge']}->{f['edge'] + 1}, check {f['j']} of {f['n']} "
            f"({f['part']} {-f['c'] * 1000:.1f} mm inside the clearance ring); the complete scan finds the smallest clearance "
            f"{clearance_words(v['deepest']['c'])} at check {v['deepest']['j']} of {v['deepest']['n']} on edge {v['deepest']['edge']}->{v['deepest']['edge'] + 1}")


# ----------------------------------------------------------------------------- stage 1: PRM and shortest path
def prm(seed=7, n_samples=60, k=5, checker="links"):
    """Sample n_samples configurations uniformly in the box (the demo's generator restarted from the seed; two draws
    per sample, no goal draw), keep the valid ones, add the start and the goal, connect every node to its k nearest
    nodes with the demo's edge check, and run Dijkstra from the start to the goal with Euclidean edge lengths."""
    rng = R.mulberry32(seed)
    samples, valid = [], []
    for _ in range(n_samples):
        q = (-math.pi + 2 * math.pi * rng(), -math.pi + 2 * math.pi * rng())
        samples.append(q)
        if R.valid(q, checker)[0]:
            valid.append(q)
    nodes = [Q_START, Q_GOAL] + valid
    N = len(nodes)
    edges, tested, rejected, checked = {i: [] for i in range(N)}, 0, [], 0
    seen = set()
    for i in range(N):
        order = sorted(range(N), key=lambda m: R.dist(nodes[i], nodes[m]))
        for m in order[1:k + 1]:
            key = (min(i, m), max(i, m))
            if key in seen:
                continue
            seen.add(key)
            tested += 1
            ok, n, j, *_ = R.check_edge(nodes[key[0]], nodes[key[1]], checker)
            checked += n if ok else j
            if ok:
                d = R.dist(nodes[key[0]], nodes[key[1]])
                edges[key[0]].append((key[1], d)); edges[key[1]].append((key[0], d))
            else:
                rejected.append(key)
    # Dijkstra; `settled` counts the nodes settled before the goal is popped
    distv, prev = [math.inf] * N, [-1] * N
    distv[0] = 0.0
    heap, settled = [(0.0, 0)], 0
    while heap:
        d, i = heapq.heappop(heap)
        if d > distv[i]:
            continue
        if i == 1:
            break
        settled += 1
        for m, w in edges[i]:
            if d + w < distv[m]:
                distv[m], prev[m] = d + w, i
                heapq.heappush(heap, (d + w, m))
    path = None
    if distv[1] < math.inf:
        path, i = [], 1
        while i >= 0:
            path.append(i); i = prev[i]
        path.reverse()
    accepted = sum(len(v) for v in edges.values()) // 2
    return {"nodes": nodes, "samples": samples, "n_valid": len(valid), "edges": edges, "tested": tested, "accepted": accepted,
            "rejected": rejected, "checked_configurations": checked, "path": path, "settled_before_goal": settled, "k": k, "n_samples": n_samples, "seed": seed}


# ----------------------------------------------------------------------------- stage 2: RRT-Connect
def extend(tree_nodes, tree_parents, q_target, checker, counter):
    """One extension of a tree toward q_target by the demo's rule; counter[0] accumulates the configurations the edge
    check evaluated. Returns ('reached' | 'advanced' | 'trapped', index)."""
    near = min(range(len(tree_nodes)), key=lambda m: R.dist(q_target, tree_nodes[m]))
    q_near = tree_nodes[near]
    d = R.dist(q_target, q_near)
    q_new = q_target if d <= STEP else (q_near[0] + STEP * (q_target[0] - q_near[0]) / d, q_near[1] + STEP * (q_target[1] - q_near[1]) / d)
    ok, n, j, *_ = R.check_edge(q_near, q_new, checker)
    counter[0] += n if ok else j
    if not ok:
        return "trapped", None
    tree_nodes.append(q_new); tree_parents.append(near)
    return ("reached" if q_new == q_target else "advanced"), len(tree_nodes) - 1


def rrt_connect(seed=7, budget=3000, checker="links"):
    """Kuffner and LaValle's RRT-Connect with the demo's rules: tree A grows from the start, tree B from the goal; each
    iteration draws one uniform sample (two draws, no goal sampling), extends the current tree toward it by at most
    10 deg, then the other tree repeats extensions toward the new node until it reaches it or is trapped; the roles
    swap every iteration. The trace records every extension attempt."""
    rng = R.mulberry32(seed)
    A = {"nodes": [Q_START], "parents": [-1]}; B = {"nodes": [Q_GOAL], "parents": [-1]}
    trace, outcome, meet, counter = [], "budget", None, [0]
    for i in range(1, budget + 1):
        q_rand = (-math.pi + 2 * math.pi * rng(), -math.pi + 2 * math.pi * rng())
        a, b = (A, B) if i % 2 == 1 else (B, A)
        status, idx = extend(a["nodes"], a["parents"], q_rand, checker, counter)
        ev = {"i": i, "q_rand": q_rand, "grow": "A" if a is A else "B", "extend": status, "connect": []}
        if status != "trapped":
            q_new = a["nodes"][idx]
            while True:
                s2, idx2 = extend(b["nodes"], b["parents"], q_new, checker, counter)
                ev["connect"].append(s2)
                if s2 == "reached":
                    outcome, meet = "connected", {"A": idx if a is A else idx2, "B": idx2 if a is A else idx, "q": q_new}
                    break
                if s2 == "trapped":
                    break
        trace.append(ev)
        if outcome == "connected":
            break
    path = None
    if meet:
        pa, i = [], meet["A"]
        while i >= 0:
            pa.append(A["nodes"][i]); i = A["parents"][i]
        pa.reverse()
        pb, i = [], meet["B"]
        while i >= 0:
            pb.append(B["nodes"][i]); i = B["parents"][i]
        path = pa + pb[1:] if pa[-1] == pb[0] else pa + pb
    attempts = sum(1 + len(e["connect"]) for e in trace)
    added = sum((e["extend"] != "trapped") + sum(s != "trapped" for s in e["connect"]) for e in trace)
    return {"A": A, "B": B, "trace": trace, "outcome": outcome, "iterations": len(trace), "path": path, "seed": seed,
            "attempts": attempts, "added": added, "blocked": attempts - added, "checked_configurations": counter[0], "meet": meet}


def rrt_counts(seed, budget=3000, checker="links"):
    """The demo's one-tree RRT with the same counting: iterations, extension attempts (goal tries included), nodes added,
    configurations the edge checks evaluated."""
    trace = []
    r = R.rrt(seed, budget, checker, trace)
    attempts = sum(1 + ("goal_try" in e) for e in trace)
    checked = sum((e["n"] if e["result"] == "accepted" else e["j"]) + ((e["goal_try"]["n"] if e["goal_try"]["result"] == "accepted" else e["goal_try"]["j"]) if e.get("goal_try") else 0) for e in trace)
    pts = [r["nodes"][i] for i in r["path"]] if r["path"] else None
    return {"iterations": r["iterations"], "attempts": attempts, "added": len(r["nodes"]) - 1, "checked_configurations": checked,
            "nodes": len(r["nodes"]), "outcome": r["outcome"], "length_deg": path_length(pts) / DEG if pts else None, "edges": len(pts) - 1 if pts else None}


# ----------------------------------------------------------------------------- stage 3: shortcutting
def point_on_path(pts, cum, s):
    """The configuration at arc length s along the polyline pts (cum = cumulative lengths)."""
    for e in range(len(pts) - 1):
        if s <= cum[e + 1] or e == len(pts) - 2:
            L = cum[e + 1] - cum[e]
            f = 0.0 if L == 0 else min(max((s - cum[e]) / L, 0.0), 1.0)
            return (pts[e][0] + f * (pts[e + 1][0] - pts[e][0]), pts[e][1] + f * (pts[e + 1][1] - pts[e][1])), e, f
    raise AssertionError


def shortcut(pts, seed=7, attempts=100, checker="links"):
    """Random shortcutting with the complete-candidate rule: pick two random arc-length positions on the current path
    (two draws per attempt from the demo's generator, restarted from the seed), check the straight chord between them
    with the demo's edge check, build the candidate path (the chord replaces the part between the two positions, and
    the two edges that were split become shorter edges with their own check grids), validate the WHOLE candidate with
    the same 1 deg rule, and commit it only when that validation passes; otherwise keep the previous path. The log
    records, per attempt, the chord check and the candidate validation with their evaluation counts kept apart.
    Returns the path and every attempt."""
    rng = R.mulberry32(seed)
    pts = list(pts)
    log = []
    for t in range(1, attempts + 1):
        cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            cum.append(cum[-1] + R.dist(a, b))
        s1, s2 = sorted((rng() * cum[-1], rng() * cum[-1]))
        (q1, e1, f1), (q2, e2, f2) = point_on_path(pts, cum, s1), point_on_path(pts, cum, s2)
        if e1 == e2:
            log.append({"t": t, "result": "same edge", "s": (s1, s2)}); continue
        ok, n, j, q_fail, part, c_fail, _ = R.check_edge(q1, q2, checker)
        before = cum[-1]
        if not ok:
            log.append({"t": t, "result": "chord rejected", "s": (s1, s2), "n": n, "chord_evaluated": j, "j": j, "part": part, "c": c_fail, "q1": q1, "q2": q2})
            continue
        candidate = pts[:e1 + 1] + [q1, q2] + pts[e2 + 1:]
        candidate = [p for i, p in enumerate(candidate) if i == 0 or R.dist(p, candidate[i - 1]) > 1e-12]   # drop zero-length joints at f = 0 or 1
        vc = validate_path(candidate, checker)
        entry = {"t": t, "s": (s1, s2), "n": n, "chord_evaluated": n, "candidate_evaluated": vc["evaluated"], "candidate_scheduled": vc["scheduled"],
                 "length_before_deg": before / DEG, "length_candidate_deg": vc["length_deg"], "edges": len(candidate) - 1, "q1": q1, "q2": q2}
        if vc["valid"]:
            pts = candidate
            entry.update({"result": "accepted", "length_after_deg": vc["length_deg"]})
        else:
            f = vc["fail"]
            entry.update({"result": "candidate rejected", "fail_edge": f["edge"], "fail_j": f["j"], "fail_n": f["n"], "fail_part": f["part"], "fail_c": f["c"], "fail_q": f["q"]})
        log.append(entry)
    return pts, log


def resample(pts, T):
    """T + 1 configurations equally spaced in arc length along the polyline, the first and last being its endpoints."""
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + R.dist(a, b))
    return [point_on_path(pts, cum, cum[-1] * t / T)[0] for t in range(T + 1)]


# ----------------------------------------------------------------------------- stage 4: KOMO with whole-link clearance rows
T_K, K_ORDER = 20, 2
L1, L2, OBS_C, OBS_R = R.L1, R.L2, np.array(R.OBS_C), R.OBS_R


def komo_dt():
    return 1.0 / T_K


def knots(z):
    return np.vstack([np.array(Q_START), np.asarray(z).reshape(T_K, 2)])


def segment_point_and_jac(q, link):
    """The point of link `link` (1 or 2) closest to the disc center, its distance, and the Jacobian of that point with
    the closest-point parameter held fixed. The derivative of the distance to a segment equals the derivative at the
    fixed closest point wherever the distance is positive and the closest point is unique, including where the closest
    point is an endpoint (Danskin's theorem); only the second derivative jumps there."""
    c1, s1 = math.cos(q[0]), math.sin(q[0])
    c12, s12 = math.cos(q[0] + q[1]), math.sin(q[0] + q[1])
    e = np.array([L1 * c1, L1 * s1])
    if link == 1:
        a, b = np.zeros(2), e
    else:
        a, b = e, e + np.array([L2 * c12, L2 * s12])
    d = b - a
    dd = float(d @ d)
    s = float(np.clip((OBS_C - a) @ d / dd, 0.0, 1.0)) if dd > 0 else 0.0
    p = a + s * d
    if link == 1:
        J = np.array([[-s * L1 * s1, 0.0], [s * L1 * c1, 0.0]])
    else:
        J = np.array([[-L1 * s1 - s * L2 * s12, -s * L2 * s12], [L1 * c1 + s * L2 * c12, s * L2 * c12]])
    return p, float(np.linalg.norm(p - OBS_C)), J, s


def acc_residuals(z):
    Q = knots(z)
    c = math.sqrt(komo_dt()) / komo_dt() ** K_ORDER
    r, J = np.zeros(2 * T_K), np.zeros((2 * T_K, 2 * T_K))
    for t in range(1, T_K + 1):
        qm2 = Q[t - 2] if t >= 2 else np.array(Q_START)
        r[2 * (t - 1):2 * t] = qm2 - 2 * Q[t - 1] + Q[t]
        for idx, coef in ((t, 1.0), (t - 1, -2.0), (t - 2, 1.0)):
            if idx >= 1:
                J[2 * (t - 1):2 * t, 2 * (idx - 1):2 * idx] += coef * np.eye(2)
    return c * r, c * J


def eq_constraints(z):
    """The fixed goal configuration (two rows; the endpoint is identical across the comparison) and the rest condition
    (the final velocity, RAI's order-1 jointState)."""
    Q = knots(z)
    h = np.concatenate([Q[T_K] - np.array(Q_GOAL), (Q[T_K] - Q[T_K - 1]) / komo_dt()])
    J = np.zeros((4, 2 * T_K))
    J[0:2, 2 * (T_K - 1):2 * T_K] = np.eye(2)
    J[2:4, 2 * (T_K - 1):2 * T_K] = np.eye(2) / komo_dt()
    J[2:4, 2 * (T_K - 2):2 * (T_K - 1)] = -np.eye(2) / komo_dt()
    return h, J


def ineq_constraints(z):
    """g_{t,l} = r_o + 0.005 - dist(c, link_l(q_t)) <= 0 for every free knot and both links: the demo's checker as rows.
    There are no joint-box rows; the box enters only through the final validation."""
    Q = knots(z)
    g, J = np.zeros(2 * T_K), np.zeros((2 * T_K, 2 * T_K))
    for t in range(1, T_K + 1):
        for link in (1, 2):
            p, dist, Jp, _ = segment_point_and_jac(Q[t], link)
            row = 2 * (t - 1) + (link - 1)
            g[row] = OBS_R + MARGIN - dist
            n = (p - OBS_C) / dist
            J[row, 2 * (t - 1):2 * t] = -n @ Jp
    return g, J


def cost(z):
    r, _ = acc_residuals(z)
    return float(r @ r)


def grad_cost(z):
    r, J = acc_residuals(z)
    return 2.0 * J.T @ r


def linear_init():
    return np.array([np.array(Q_START) + (t / T_K) * (np.array(Q_GOAL) - np.array(Q_START)) for t in range(1, T_K + 1)]).ravel()


def path_init(pts):
    rs = resample(pts, T_K)
    return np.array(rs[1:]).ravel()


_shortcut_cache = {}


def shortcut_init(rrt_seed):
    """The start `shortcut` of a case: the demo's one-tree RRT at seed n (3000 extensions), shortened by 100 shortcut
    attempts with the same seed (stage 3's path for n = 7), resampled by arc length to the T_K free knots."""
    if rrt_seed not in _shortcut_cache:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            rr = R.rrt(rrt_seed, 3000, "links")
            sc, _ = shortcut([rr["nodes"][i] for i in rr["path"]], rrt_seed, 100)
        _shortcut_cache[rrt_seed] = sc
    return path_init(_shortcut_cache[rrt_seed])


def supported_case(case_id):
    """Parse a case ID and refuse every field this adapter does not implement, before any row is built. The whole-link
    program [W] implements: AugmentedLag and SquaredPenalty; the default and tight options; the starts line and
    shortcut (the RRT seed only with shortcut); any T; the acceleration cost only (k = 2); the disc always present; any
    positive margin; any positive constraint scale; -warm and -replay with a given z0. The historical fixed-mu options
    are archived and refused."""
    case = rai_nlp.Case(case_id)
    problems = []
    if case.model != "W":
        problems.append("staged_two_link solves only the whole-link model W")
    if case.opts == "hist":
        problems.append("the historical fixed-mu options are archived and not recomputed")
    if case.start not in ("line", "shortcut"):
        problems.append(f"start {case.start!r} is not a start of the whole-link model (line or shortcut)")
    if case.start != "shortcut" and "-rrt" in case.id:
        problems.append("an RRT seed applies only to the shortcut start")
    if case.k != 2:
        problems.append(f"k = {case.k}: the whole-link program implements the acceleration cost only (k = 2)")
    if not case.disc:
        problems.append("-nodisc: the whole-link program always has the disc rows")
    if not case.margin_mm > 0:
        problems.append(f"margin {case.margin_mm} mm: the whole-link program needs a positive margin")
    if not case.scale > 0:
        problems.append(f"scale {case.scale}: the constraint scale must be positive")
    if problems:
        raise ValueError(f"{case_id}: " + "; ".join(problems))
    return case


def solve_komo(case_id, z0=None):
    """Solve the worked example [W] for one case ID with RAI's loop (rai_nlp.solve_case), from an exact start.

    The case fixes the method and options (`default`: RAI's untouched options; `tight`: stopTolerance 1e-8,
    stopFTolerance 1e-12, stopEvals 5000), the constraint scale s (`-s10`), T, the margin (`-m5.1` is 5.1 mm) and the
    start: the straight joint-space line for `line`, the shortcut path of RRT seed n for `shortcut` (`-rrt<n>`,
    default 7). An explicit z0 replaces that start; a `-warm` or `-replay` case needs it (the parent's answer or the
    parent's recorded start). Seeded cases (`-seed<n>`) are RAI-only and raise. T_K and MARGIN are set from the case
    for the solve and restored afterwards.

    The rows are acc_residuals (sos), eq_constraints (eq) and ineq_constraints (ineq), with the eq and ineq rows
    multiplied by s, as KOMO multiplies a feature by its scale. RAI's dual therefore prices the scaled row s g, and the
    physical multiplier of g is s times RAI's dual: lam per metre for the clearance rows, kap per rad for the goal rows
    and per rad per normalized time for the rest rows. Returns (z, lam, kap, info) with the physical lam and kap;
    info is rai_nlp.solve_case's record plus the case fields, RAI's own duals (lam_rai, kap_rai) and the start kind."""
    global T_K, MARGIN
    case = supported_case(case_id)
    if (case.warm or case.replay) and z0 is None:
        raise ValueError(f"{case_id}: a -warm or -replay case starts from its parent's vector; pass it as z0")
    saved = (T_K, MARGIN)
    try:
        T_K, MARGIN = case.T, case.margin_mm / 1000.0
        if z0 is not None:
            x0 = np.array(z0, dtype=float)
        elif case.start == "line":
            x0 = linear_init()
        elif case.start == "shortcut":
            x0 = shortcut_init(case.rrt)
        else:
            raise ValueError(f"{case_id}: start {case.start} is not a start of the whole-link model")
        s = case.scale
        nlp = rai_nlp.NLP.from_blocks(2 * T_K, sos=acc_residuals,
                                      eq=lambda x: tuple(s * v for v in eq_constraints(x)),
                                      ineq=lambda x: tuple(s * v for v in ineq_constraints(x)))
        z, lam_rai, kap_rai, info = rai_nlp.solve_case(nlp, case, x0)
        info.update(case=case.id, scale=s, T=T_K, margin_m=MARGIN, start=case.start, rrt_seed=case.rrt,
                    start_kind="exact start" if z0 is None else ("warm start" if case.warm else "replay start" if case.replay else "exact start from z0"),
                    lam_rai=lam_rai, kap_rai=kap_rai)
        return z, s * lam_rai, s * kap_rai, info
    finally:
        T_K, MARGIN = saved


def knot_report(z):
    Q = knots(z)
    pts = [tuple(q) for q in Q]
    v = validate_path(pts)
    knot_c = [R.clearance(tuple(q), "links")[0] for q in Q]
    return {"validation": v, "knot_min_c": min(knot_c), "knot_argmin": int(np.argmin(knot_c)), "pts": pts}


class case_model:
    """`with case_model(case_id):` sets T_K and MARGIN from the case for the block (to evaluate that case's rows and
    knots after solve_komo has restored the defaults)."""

    def __init__(self, case_id):
        self.case = rai_nlp.Case(case_id)

    def __enter__(self):
        global T_K, MARGIN
        self.saved = (T_K, MARGIN)
        T_K, MARGIN = self.case.T, self.case.margin_mm / 1000.0
        return self.case

    def __exit__(self, *exc):
        global T_K, MARGIN
        T_K, MARGIN = self.saved
        return False


def physical_report(case_id, z, lam):
    """The residuals of a returned path in task units, computed here and never inferred from RAI's feasible flag: the
    knot clearance deficit per link in mm against the case's margin, the goal error in rad, the rest error in rad per
    normalized time, the clearance rows that carry a multiplier with their physical lam (per metre), and the demo's
    validation of the knot-to-knot motion (1 deg rule, every scheduled configuration scanned). The between-knot
    deficit is how far the deepest scanned configuration lies inside the 5 mm ring of rrt_two_link (0 when none does).
    The scan and the knot clearance use the demo's 5 mm ring; the knot rows use the case's margin."""
    with case_model(case_id):
        g, _ = ineq_constraints(z)
        h, _ = eq_constraints(z)
        rep = knot_report(z)
        worst = int(np.argmax(g))
        dc = rep["validation"]["deepest"]["c"]
        return {"case": case_id, "cost": cost(z), "T": T_K, "margin_mm": MARGIN * 1000,
                "knot_deficit_mm": float(max(0.0, g.max()) * 1000),
                "knot_deficit_mm_link1": float(max(0.0, g[0::2].max()) * 1000), "knot_deficit_mm_link2": float(max(0.0, g[1::2].max()) * 1000),
                "worst_row": (worst // 2 + 1, worst % 2 + 1),
                "goal_error_rad": float(np.abs(h[:2]).max()), "rest_error": float(np.abs(h[2:]).max()),
                "active": [(int(i) // 2 + 1, int(i) % 2 + 1, float(lam[i])) for i in np.nonzero(lam > 1e-6)[0]],
                "lam_sum": float(lam.sum()),
                "between_knot_deficit_mm": float(max(0.0, -dc) * 1000), "validation": rep["validation"], "pts": rep["pts"],
                "knot_min_c": rep["knot_min_c"], "knot_argmin": rep["knot_argmin"]}


def mm_words(v):
    """A length in mm: three decimals from 1 micrometre up, scientific notation below (a small nonzero residual is not 0)."""
    return "0 mm" if v == 0 else (f"{v:.3f} mm" if v >= 1e-3 else f"{v:.1e} mm")


def report_words(r):
    """One line of physical residuals for a physical_report, every number after its case ID."""
    act = ", ".join(f"knot {t} link {l}: {v:.3f} per m" for t, l, v in r["active"]) or "none"
    v = r["validation"]
    between = (f"the knot-to-knot motion is valid at 1 deg per joint, smallest clearance {clearance_words(v['deepest']['c'])}" if v["valid"] else
               f"the knot-to-knot motion fails the 1 deg check first on edge {v['fail']['edge']}->{v['fail']['edge'] + 1}, and the deepest scanned configuration "
               f"(edge {v['deepest']['edge']}->{v['deepest']['edge'] + 1}, check {v['deepest']['j']} of {v['deepest']['n']}) is {r['between_knot_deficit_mm']:.3f} mm inside the ring")
    return (f"{r['case']}: cost {r['cost']:.6f}; knot deficit {mm_words(r['knot_deficit_mm'])} (largest at knot {r['worst_row'][0]} link {r['worst_row'][1]}; "
            f"link 1 {mm_words(r['knot_deficit_mm_link1'])}, link 2 {mm_words(r['knot_deficit_mm_link2'])}, margin {r['margin_mm']:g} mm); goal error {r['goal_error_rad']:.2e} rad; "
            f"rest error {r['rest_error']:.2e} rad per normalized time; physical multipliers {act}; {between}")


# ----------------------------------------------------------------------------- figures
def figures(data, out, stages=(1, 2, 3, 4)):
    """Draw the staged figures of the given stages into the directory `out`; returns the file names written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke
    HALO = [withStroke(linewidth=5, foreground="white")]
    BLUE, GREEN, RED, GREY, ORANGE, PURPLE = "#1f5fbf", "#2a7d2a", "#b00020", "#9a9a9a", "#e08a00", "#7a3fb0"
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15, "axes.titlepad": 10})
    written = []

    def save(fig, name, **kw):
        fig.savefig(Path(out) / name, **kw)
        plt.close(fig)
        written.append(name)
    q1 = np.arange(-60, 120.5, 0.5); q2 = np.arange(-40, 150.5, 0.5)
    bad = np.array([[not R.valid((a * DEG, b * DEG), "links")[0] for a in q1] for b in q2])
    q1f = np.arange(-180, 180.5, 1.0); q2f = np.arange(-180, 180.5, 1.0)
    badf = np.array([[not R.valid((a * DEG, b * DEG), "links")[0] for a in q1f] for b in q2f])
    cmap = matplotlib.colors.ListedColormap(["#c9c9c9"])

    def cspace(ax, title, full=False, direct=True, window=None):
        if window:
            x0, x1, y0, y1 = window
            ax.imshow(np.where(bad, 1.0, np.nan), extent=(-60, 120, -40, 150), origin="lower", cmap=cmap, vmin=0, vmax=1, zorder=0, interpolation="nearest")
            ax.set(xlim=(x0, x1), ylim=(y0, y1))
        elif full:
            ax.imshow(np.where(badf, 1.0, np.nan), extent=(-180, 180, -180, 180), origin="lower", cmap=cmap, vmin=0, vmax=1, zorder=0, interpolation="nearest")
            ax.set(xlim=(-180, 180), ylim=(-180, 180), xticks=range(-180, 181, 90), yticks=range(-180, 181, 90))
        else:
            ax.imshow(np.where(bad, 1.0, np.nan), extent=(-60, 120, -40, 150), origin="lower", cmap=cmap, vmin=0, vmax=1, zorder=0, interpolation="nearest")
            ax.set(xlim=(-60, 120), ylim=(-40, 150))
        ax.set(xlabel="$q_1$, shoulder angle [deg]", ylabel="$q_2$, elbow angle [deg]", title=title)
        ax.set_aspect("equal"); ax.grid(color="#e6e6e6", linewidth=.6)
        if direct:
            ax.plot([Q_START[0] / DEG, Q_GOAL[0] / DEG], [Q_START[1] / DEG, Q_GOAL[1] / DEG], "--", color=RED, linewidth=1.2, alpha=.8, zorder=2)
        ax.plot(Q_START[0] / DEG, Q_START[1] / DEG, "o", color=BLUE, markersize=9, zorder=6)
        ax.plot(Q_GOAL[0] / DEG, Q_GOAL[1] / DEG, "D", color=GREEN, markersize=9, zorder=6)
        box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.85)
        ax.annotate("start", (Q_START[0] / DEG, Q_START[1] / DEG), xytext=(10, -18), textcoords="offset points", color=BLUE, bbox=box, zorder=9)
        ax.annotate("goal", (Q_GOAL[0] / DEG, Q_GOAL[1] / DEG), xytext=(-48, -22), textcoords="offset points", color=GREEN, bbox=box, zorder=9)

    def polyline(ax, pts, **kw):
        ax.plot([p[0] / DEG for p in pts], [p[1] / DEG for p in pts], **kw)

    def workspace(ax, title):
        ax.set(xlim=(-0.2, 0.7), ylim=(-0.25, 0.65), xlabel="x [m]", ylabel="y [m]", title=title); ax.set_aspect("equal"); ax.grid(color="#e6e6e6", linewidth=.6)
        ax.add_patch(plt.Circle(R.OBS_C, R.OBS_R, color="#bbbbbb", zorder=2)); ax.add_patch(plt.Circle(R.OBS_C, R.OBS_R + MARGIN, fill=False, color="#888888", linestyle="--", zorder=2))

    def arm(ax, q, color, lw=2.5, alpha=1.0, zorder=5, ls="-"):
        base, e, t, _ = R.points(q)
        ax.plot([0, e[0], t[0]], [0, e[1], t[1]], ls, color=color, linewidth=lw, alpha=alpha, zorder=zorder, solid_capstyle="round")

    forbidden_patch = matplotlib.patches.Patch(color="#c9c9c9", label="forbidden (link within 5 mm of disc)")
    ws_handles = [Line2D([], [], color=BLUE, linewidth=2.5, label="start"), Line2D([], [], color=GREEN, linewidth=2.5, label="goal"),
                  matplotlib.patches.Patch(color="#bbbbbb", label="disc, radius 0.05 m"), Line2D([], [], color="#888888", linestyle="--", label="5 mm clearance ring")]

    if 1 in stages:
        P = data["prm"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        cspace(axes[0], f"Roadmap: {P['n_valid']} samples, {P['accepted']} edges", full=True, direct=False)
        for i, nb in P["edges"].items():
            for m, _ in nb:
                if i < m:
                    polyline(axes[0], [P["nodes"][i], P["nodes"][m]], color=GREY, linewidth=.8, zorder=1)
        for q in P["nodes"][2:]:
            axes[0].plot(q[0] / DEG, q[1] / DEG, ".", color="#555555", markersize=6, zorder=3)
        for key in P["rejected"]:
            polyline(axes[0], [P["nodes"][key[0]], P["nodes"][key[1]]], color=RED, linewidth=1.4, linestyle=":", zorder=2)
        if P["path"]:
            polyline(axes[0], [P["nodes"][i] for i in P["path"]], color=BLUE, linewidth=3.2, zorder=5)
        cs1 = [forbidden_patch, Line2D([], [], color=GREY, linewidth=1, label="accepted roadmap edge"), Line2D([], [], color=RED, linestyle=":", linewidth=1.4, label="rejected candidate edge"),
               Line2D([], [], color=BLUE, linewidth=3, label="shortest route (Dijkstra)")]
        workspace(axes[1], "The route's postures")
        if P["path"]:
            for i in P["path"]:
                arm(axes[1], P["nodes"][i], BLUE, lw=1.4, alpha=.45)
        arm(axes[1], Q_START, BLUE); arm(axes[1], Q_GOAL, GREEN)
        fig.legend(handles=cs1 + ws_handles + [Line2D([], [], color=BLUE, linewidth=1.4, alpha=.45, label="route node, right panel")],
                   loc="lower center", ncol=3, frameon=False, fontsize=13, columnspacing=1.2, handlelength=2.2)
        fig.tight_layout(rect=[0, 0.17, 1, 1]); save(fig, "staged-1-prm.png", dpi=160)

    if 2 in stages:
        C = data["rrt_connect"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        cspace(axes[0], f"RRT-Connect: trees meet at iteration {C['iterations']}")
        for tree, color in ((C["A"], BLUE), (C["B"], GREEN)):
            for k in range(1, len(tree["nodes"])):
                polyline(axes[0], [tree["nodes"][tree["parents"][k]], tree["nodes"][k]], color=color, linewidth=1.6, alpha=.9, zorder=3)
        if C["path"]:
            polyline(axes[0], C["path"], color="#dbe6f7", linewidth=7, zorder=2)
            m = C["meet"]["q"]
            axes[0].plot(m[0] / DEG, m[1] / DEG, "o", markersize=13, markerfacecolor="none", markeredgecolor="black", markeredgewidth=2, zorder=7)
            axes[0].annotate("trees meet", (m[0] / DEG, m[1] / DEG), xytext=(12, -24), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.85), zorder=9)
        axes[0].legend(handles=[forbidden_patch, Line2D([], [], color=BLUE, linewidth=1.6, label="tree from the start"), Line2D([], [], color=GREEN, linewidth=1.6, label="tree from the goal"),
                                Line2D([], [], color="#dbe6f7", linewidth=7, label="connecting path (halo)"), Line2D([], [], color=RED, linestyle="--", linewidth=1.2, label="direct edge (invalid)")], loc="upper right")
        Rd = data["rrt"]
        cspace(axes[1], f"One-tree RRT: goal joined at iteration {Rd['iterations']}")
        for k in range(1, len(Rd["nodes"])):
            polyline(axes[1], [Rd["nodes"][Rd["parents"][k]], Rd["nodes"][k]], color=GREY, linewidth=1.2, zorder=2)
        polyline(axes[1], [Rd["nodes"][i] for i in Rd["path"]], color=BLUE, linewidth=3.4, alpha=.55, zorder=3)
        axes[1].legend(handles=[Line2D([], [], color=GREY, linewidth=1.2, label="tree edges"), Line2D([], [], color=BLUE, linewidth=3.4, alpha=.55, label="path to the goal"), Line2D([], [], color=RED, linestyle="--", linewidth=1.2, label="direct edge (invalid)")], loc="upper right")
        fig.tight_layout(); save(fig, "staged-2-rrt-connect.png", dpi=160)

    if 3 in stages:
        S = data["shortcut"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        cspace(axes[0], f"Shortcutting: path length {S['before']['length_deg']:.1f}° → {S['after']['length_deg']:.1f}°", window=(-30, 60, 30, 110))
        for a in S["log"]:
            if a["result"] in ("chord rejected", "candidate rejected"):
                polyline(axes[0], [a["q1"], a["q2"]], color=RED, linewidth=.9, alpha=.45, zorder=2)
        polyline(axes[0], S["before_pts"], color=GREY, linewidth=2.5, zorder=3)
        polyline(axes[0], S["after_pts"], color=BLUE, linewidth=3.2, zorder=5)
        dq = S["fine"]["deepest"]["q"]
        axes[0].plot(dq[0] / DEG, dq[1] / DEG, "x", color=RED, markersize=12, markeredgewidth=2.5, zorder=8, path_effects=HALO)
        axes[0].annotate("0.05° check: 0.1 mm\ninside the ring", (dq[0] / DEG, dq[1] / DEG), xytext=(50, 40), textcoords="offset points", ha="left", color=RED, fontsize=12, arrowprops=dict(arrowstyle="-", color=RED, linewidth=.8))
        cs3 = [forbidden_patch, Line2D([], [], color=GREY, linewidth=2.5, label="RRT path before"), Line2D([], [], color=BLUE, linewidth=3.2, label="after 100 attempts"),
               Line2D([], [], color=RED, linewidth=.9, alpha=.6, label="rejected chord or candidate"), Line2D([], [], color=RED, linestyle="--", linewidth=1.2, label="direct edge (invalid)"),
               Line2D([], [], color=RED, marker="x", linewidth=0, markeredgewidth=2.5, label="0.05° check: deepest point")]
        axes[0].text(.02, .97, "window of the ±180° box; the band continues below", transform=axes[0].transAxes, fontsize=12, color="#555555", va="top")
        workspace(axes[1], "Postures at 8 steps along each path")
        for q in resample(S["before_pts"], 8):
            arm(axes[1], q, GREY, lw=1.6, alpha=.7)
        for q in resample(S["after_pts"], 8):
            arm(axes[1], q, BLUE, lw=1.6, alpha=.7)
        arm(axes[1], Q_START, BLUE); arm(axes[1], Q_GOAL, GREEN)
        fig.legend(handles=cs3 + ws_handles[2:] + [Line2D([], [], color=GREY, linewidth=1.6, label="before, right panel"), Line2D([], [], color=BLUE, linewidth=1.6, alpha=.7, label="after, right panel")],
                   loc="lower center", ncol=3, frameon=False, fontsize=13, columnspacing=1.2, handlelength=2.2)
        fig.tight_layout(rect=[0, 0.2, 1, 1]); save(fig, "staged-3-shortcut.png", dpi=160)

    if 4 in stages:
        K = data["komo"]
        ids, runs = K["ids"], K["runs"]
        dl, ds, rl, rs = (runs[ids[k]] for k in ("default_line", "default_shortcut", "ref_line", "ref_shortcut"))
        fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.4))
        # (a) the default call from both starts, with the knot that misses the requirement
        cspace(axes[0], "Default options\nBoth initial guesses give the same result", direct=False, window=(-30, 60, 30, 110))
        axes[0].title.set_fontsize(13)
        polyline(axes[0], dl["init_pts"], color=RED, linewidth=1.6, linestyle="--", zorder=3)
        polyline(axes[0], ds["init_pts"], color="#444444", linewidth=1.2, linestyle=":", marker=".", markersize=4, zorder=6)
        polyline(axes[0], ds["pts"], color=BLUE, linewidth=3.6, marker="o", markersize=6, zorder=4)
        polyline(axes[0], dl["pts"], color=ORANGE, linewidth=0, marker="o", markersize=12, markerfacecolor="none", markeredgewidth=1.8, zorder=5)
        t_bad, l_bad = dl["report"]["worst_row"]
        q = dl["pts"][t_bad]
        axes[0].plot(q[0] / DEG, q[1] / DEG, "s", markersize=20, markerfacecolor="none", markeredgecolor=RED, markeredgewidth=2.0, zorder=7)
        axes[0].annotate(f"knot {t_bad}, link {l_bad}:\n{dl['report']['knot_deficit_mm']:.1f} mm inside the ring", (q[0] / DEG, q[1] / DEG), xytext=(.5, .92), textcoords="axes fraction", ha="center", va="top",
                         color=RED, fontsize=13, arrowprops=dict(arrowstyle="-", color=RED, linewidth=.8, shrinkB=11))
        axes[0].text(.02, .02, "window of the ±180° box", transform=axes[0].transAxes, fontsize=12, color="#555555")
        # (b) the converged reference: knots on the ring, the motion between them is not checked by the rows
        cspace(axes[1], "Accurate reference\nScaled constraints and tight stops", direct=False, window=(-30, 60, 30, 110))
        axes[1].title.set_fontsize(13)
        polyline(axes[1], rs["pts"], color=BLUE, linewidth=3.6, marker="o", markersize=6, zorder=4)
        polyline(axes[1], rl["pts"], color=ORANGE, linewidth=0, marker="o", markersize=12, markerfacecolor="none", markeredgewidth=1.8, zorder=5)
        active_knots = sorted({t for t, _, _ in rl["report"]["active"]})
        for t in active_knots:
            q = rl["pts"][t]
            axes[1].plot(q[0] / DEG, q[1] / DEG, "s", markersize=20, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.8, zorder=7)
            axes[1].annotate(f"knot {t}", (q[0] / DEG, q[1] / DEG), xytext=(34, 14) if t == active_knots[0] else (-30, 14), textcoords="offset points",
                             ha="left" if t == active_knots[0] else "center", fontsize=12,
                             arrowprops=dict(arrowstyle="-", color="black", linewidth=.6, shrinkB=11) if t == active_knots[0] else None)
        dpt = rl["report"]["validation"]["deepest"]
        dq = dpt["q"]
        axes[1].plot(dq[0] / DEG, dq[1] / DEG, "x", color=RED, markersize=12, markeredgewidth=2.5, zorder=8, path_effects=HALO)
        axes[1].annotate(f"between knots {dpt['edge']} and {dpt['edge'] + 1}:\n{rl['report']['between_knot_deficit_mm']:.1f} mm inside the ring", (dq[0] / DEG, dq[1] / DEG), xytext=(.5, .92),
                         textcoords="axes fraction", ha="center", va="top", color=RED, fontsize=13, arrowprops=dict(arrowstyle="-", color=RED, linewidth=.8))
        # (c) the reference's knots in the workspace and the deepest posture between them
        workspace(axes[2], "Accurate reference\nKnots and one intermediate posture")
        axes[2].title.set_fontsize(13)
        for q in rs["pts"]:
            arm(axes[2], q, BLUE, lw=1.1, alpha=.45)
        arm(axes[2], tuple(dq), RED, lw=2.2)
        arm(axes[2], Q_START, BLUE); arm(axes[2], Q_GOAL, GREEN)
        names = (f"{len(ds['init_pts'])} knots")
        handles = [forbidden_patch, Line2D([], [], color=RED, linestyle="--", linewidth=1.6, label="start 1: the straight line"),
                   Line2D([], [], color="#444444", linestyle=":", marker=".", linewidth=1.2, label=f"start 2: the shortcut path, {names}"),
                   Line2D([], [], color=BLUE, marker="o", linewidth=3.6, label="optimized from start 2"),
                   Line2D([], [], color=ORANGE, marker="o", markerfacecolor="none", linewidth=0, markersize=8, label="optimized from start 1 (the same knots)"),
                   Line2D([], [], color=RED, marker="s", markerfacecolor="none", linewidth=0, markersize=11, label="default call: knot that misses the margin"),
                   Line2D([], [], color="black", marker="s", markerfacecolor="none", linewidth=0, markersize=11, label="reference: knots with a multiplier"),
                   Line2D([], [], color=RED, marker="x", linewidth=0, markeredgewidth=2.5, label="deepest point between knots"),
                   Line2D([], [], color=BLUE, linewidth=1.1, alpha=.6, label="a knot, right panel"), Line2D([], [], color=RED, linewidth=2.2, label="deepest between knots, right panel")] + ws_handles[2:]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=13, columnspacing=1.2, handlelength=2.2)
        fig.tight_layout(rect=[0, 0.2, 1, 0.9])
        save(fig, "staged-4-komo.png", dpi=160)
        # Reuse the same computed artists for a readable two-panel lecture view.
        # The full workspace view remains in the notes figure above.
        axes[2].remove()
        for legend in list(fig.legends):
            legend.remove()
        fig.set_size_inches(12.5, 6.0)
        axes[1].set_title("Knot constraints met\nwithin numerical tolerance")
        for ax, x in zip(axes[:2], (.075, .57)):
            ax.set_position([x, .39, .39, .47])
            ax.set_xlim(-30, 60)
            ax.set_ylim(35, 105)
            ax.title.set_fontsize(17)
            for text in list(ax.texts):
                if text.get_text() == "window of the ±180° box":
                    text.remove()
                else:
                    text.set_fontsize(15)
        expected_labels = ["forbidden (link within 5 mm of disc)", "start 1: the straight line",
                           f"start 2: the shortcut path, {names}", "optimized from start 2",
                           "optimized from start 1 (the same knots)", "default call: knot that misses the margin",
                           "reference: knots with a multiplier", "deepest point between knots"]
        actual_labels = [handle.get_label() for handle in handles[:8]]
        if actual_labels != expected_labels:
            raise ValueError(f"Stage-4 slide legend order changed: expected {expected_labels!r}, got {actual_labels!r}")
        short_labels = ["less than 5 mm clearance", "direct initial guess", "sampled initial guess",
                        "result from sampled guess", "result from direct guess", "default: failing knot",
                        "reference: active knots", "reference: failing connection"]
        for handle, label in zip(handles[:8], short_labels):
            handle.set_label(label)
        fig.legend(handles=handles[:8], loc="lower center", ncol=2, frameon=False,
                   fontsize=15, columnspacing=2, handlelength=2.0)
        save(fig, "staged-4-komo-slide.png", dpi=180)
    return written


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true", help="draw the staged figures (written only with --out)")
    ap.add_argument("--out", help="the directory the figures are written to; without it nothing is written")
    ap.add_argument("--stage", choices=["all", "4"], default="all", help="'4' draws only the KOMO figure staged-4-komo.png")
    ap.add_argument("--json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    seed = args.seed
    data = {}
    print(f"=== the scene and the contract: rrt_two_link.py (whole links, {MARGIN * 1000:.0f} mm from the disc surface, box ±180° without wrapping, 10° extensions, edges at ≤ 1° per joint, mulberry32 restarted from seed {seed} at every stage) ===")
    d = validate_path([Q_START, Q_GOAL])
    print(f"  direct edge: {'valid' if d['valid'] else 'invalid'}; first invalid configuration {d['fail']['j']} of {d['fail']['n']}; deepest {R.depth_words(d['deepest']['c'])}")
    frac = R.forbidden_fraction("links", 2)
    print(f"  the forbidden region covers {100 * frac:.1f} % of the box on a 2° grid")

    # the demo's RRT, for comparison
    rr = R.rrt(seed, 3000, "links")
    rr_pts = [rr["nodes"][i] for i in rr["path"]]
    rc = rrt_counts(seed)
    data["rrt"] = {"nodes": rr["nodes"], "parents": rr["parents"], "path": rr["path"], "iterations": rr["iterations"], "validation": validate_path(rr_pts), "counts": rc}
    print(f"\n=== stage 0, the demo's one-tree RRT (10 % goal sampling): goal joined at iteration {rr['iterations']}, {rc['attempts']} extension attempts ({rc['added']} added a node), {rc['checked_configurations']} configurations evaluated, {len(rr['nodes'])} nodes, {len(rr['path']) - 1} edges, {data['rrt']['validation']['length_deg']:.2f}°; revalidation {revalidation_words(data['rrt']['validation'])} ===")

    # stage 1
    P = prm(seed)
    v = validate_path([P["nodes"][i] for i in P["path"]]) if P["path"] else None
    data["prm"] = {**P, "validation": v, "forbidden_fraction_2deg": frac}
    print(f"\n=== stage 1, PRM (seed {seed}, {P['n_samples']} uniform samples, each node joined to its {P['k']} nearest neighbors, Dijkstra) ===")
    print(f"  {P['n_valid']} of {P['n_samples']} samples valid; roadmap {len(P['nodes'])} nodes (start, goal, the valid samples); {P['tested']} candidate edges checked ({P['checked_configurations']} configurations evaluated), {P['accepted']} accepted, {len(P['rejected'])} rejected")
    if P["path"]:
        print(f"  Dijkstra settled {P['settled_before_goal']} nodes before the goal; shortest route {len(P['path']) - 1} edges, {v['length_deg']:.2f}° (the direct edge would be {R.dist(Q_START, Q_GOAL) / DEG:.2f}°); revalidation {revalidation_words(v)}")
        print("  route:", " -> ".join(f"#{i} {deg(P['nodes'][i])}" for i in P["path"]))
    else:
        print("  no route in the roadmap (the start and the goal are in different components); this does not show that no path exists")

    # stage 2
    C = rrt_connect(seed)
    data["rrt_connect"] = C
    print(f"\n=== stage 2, RRT-Connect (seed {seed}, two trees, no goal sampling, the same 10° step and edge check) ===")
    if C["outcome"] == "connected":
        vc = validate_path(C["path"]); data["rrt_connect"]["validation"] = vc
        print(f"  trees meet at iteration {C['iterations']}: {C['attempts']} extension attempts ({C['added']} added a node, {C['blocked']} were blocked), {C['checked_configurations']} configurations evaluated; tree A {len(C['A']['nodes'])} nodes, tree B {len(C['B']['nodes'])} nodes; path {len(C['path']) - 1} edges, {vc['length_deg']:.2f}°; revalidation {revalidation_words(vc)}")
        print(f"  against the one-tree RRT (10 % goal sampling): {rr['iterations']} iterations, {rc['attempts']} extension attempts, {rc['checked_configurations']} configurations evaluated, {len(rr['nodes'])} nodes, {data['rrt']['validation']['length_deg']:.2f}°")
        last = C["trace"][-1]
        other = "A" if last["grow"] == "B" else "B"
        print(f"  the meeting iteration: tree {last['grow']} {last['extend']} toward the sample {deg(last['q_rand'])}; tree {other} then extended toward that new node {len(last['connect'])} times ({', '.join(last['connect'])})")
    else:
        print("  budget exhausted; this does not show that no path exists")
    sweep2 = []
    for sd in range(1, 13):
        c2 = rrt_connect(sd); r2 = rrt_counts(sd)
        v2 = validate_path(c2["path"]) if c2["path"] else None
        sweep2.append({"seed": sd, "connect": {"iterations": c2["iterations"], "attempts": c2["attempts"], "checked": c2["checked_configurations"], "nodes": len(c2["A"]["nodes"]) + len(c2["B"]["nodes"]), "length_deg": v2["length_deg"] if v2 else None, "valid": v2["valid"] if v2 else None},
                       "rrt": {"iterations": r2["iterations"], "attempts": r2["attempts"], "checked": r2["checked_configurations"], "nodes": r2["nodes"], "length_deg": r2["length_deg"]}})
    data["rrt_connect"]["sweep"] = sweep2
    fewer_it = sum(s["connect"]["iterations"] < s["rrt"]["iterations"] for s in sweep2)
    fewer_chk = sum(s["connect"]["checked"] < s["rrt"]["checked"] for s in sweep2)
    med = statistics.median
    print(f"  seeds 1 to 12: RRT-Connect needs fewer iterations in {fewer_it} of 12 seeds (medians {med([s['connect']['iterations'] for s in sweep2])} against {med([s['rrt']['iterations'] for s in sweep2])}) and fewer evaluated configurations in {fewer_chk} of 12 (medians {med([s['connect']['checked'] for s in sweep2])} against {med([s['rrt']['checked'] for s in sweep2])}); every run connects and every path revalidates")
    print("  per seed (iterations / evaluated configurations, RRT-Connect against RRT): " + ", ".join(f"{s['seed']}: {s['connect']['iterations']}/{s['connect']['checked']} vs {s['rrt']['iterations']}/{s['rrt']['checked']}" for s in sweep2))

    # stage 3
    before_pts = rr_pts
    after_pts, log = shortcut(before_pts, seed, 100)
    vb, va = validate_path(before_pts), validate_path(after_pts)
    acc = [a for a in log if a["result"] == "accepted"]
    saved = R.SUB; R.SUB = 0.05 * DEG
    try:
        vfine = validate_path(after_pts)
    finally:
        R.SUB = saved
    chord_rej = [a for a in log if a["result"] == "chord rejected"]
    cand_rej = [a for a in log if a["result"] == "candidate rejected"]
    data["shortcut"] = {"before_pts": before_pts, "after_pts": after_pts, "log": log, "before": vb, "after": va, "fine": vfine, "accepted": len(acc), "attempts": len(log),
                        "chord_rejected": len(chord_rej), "candidate_rejected": len(cand_rej), "same_edge": sum(a["result"] == "same edge" for a in log),
                        "chord_evaluations": sum(a.get("chord_evaluated", 0) for a in log), "candidate_evaluations": sum(a.get("candidate_evaluated", 0) for a in log)}
    print(f"\n=== stage 3, shortcutting the demo's RRT path (seed {seed}, 100 attempts, two draws each, the same edge check on the chord, then the whole candidate path validated with the same 1° rule before it replaces the path) ===")
    print(f"  before: {len(before_pts) - 1} edges, {vb['length_deg']:.2f}°, smallest clearance {clearance_words(vb['deepest']['c'])}")
    print(f"  attempts: {len(acc)} accepted, {len(chord_rej)} chords rejected, {len(cand_rej)} candidates rejected by the whole-path validation, {data['shortcut']['same_edge']} with both positions on the same edge (nothing to cut)")
    print(f"  work: {data['shortcut']['chord_evaluations']} configurations evaluated by the chord checks, {data['shortcut']['candidate_evaluations']} by the whole-candidate validations")
    for a in acc:
        print(f"    attempt {a['t']:3d}: accepted, chord {a['n']} configuration{'s' if a['n'] != 1 else ''}, candidate {a['candidate_evaluated']} of {a['candidate_scheduled']} valid, {a['length_before_deg']:.2f}° -> {a['length_after_deg']:.2f}°, {a['edges']} edges")
    for a in cand_rej:
        print(f"    attempt {a['t']:3d}: candidate rejected: chord valid ({a['n']} configurations) but the whole path fails on edge {a['fail_edge']}->{a['fail_edge'] + 1}, check {a['fail_j']} of {a['fail_n']}, {a['fail_part']} {-a['fail_c'] * 1000:.3f} mm inside the clearance ring; the previous path is kept")
    print(f"  no shortcut ever spans the whole path; the direct edge ({R.dist(Q_START, Q_GOAL) / DEG:.2f}°) would be rejected")
    print(f"  after: {len(after_pts) - 1} edges, {va['length_deg']:.2f}°; revalidation at 1° per joint {revalidation_words(va)}")
    print(f"  finer diagnostic at 0.05° per joint: {revalidation_words(vfine)}")
    print("  path after:", " -> ".join(deg(p) for p in after_pts))

    # stage 4
    rrt_sfx = "" if seed == 7 else f"-rrt{seed}"
    ids = {"default_line": "W-auglag-default-line", "default_shortcut": "W-auglag-default-shortcut" + rrt_sfx,
           "ref_line": "W-auglag-tight-line-s10", "ref_shortcut": "W-auglag-tight-shortcut-s10" + rrt_sfx}
    print(f"\n=== stage 4, KOMO with whole-link clearance rows [W] (T = {T_K}, k = {K_ORDER}, dt = 1/T, fixed goal configuration, rest at the end; no joint-box rows), "
          f"solved with RAI's loop (rai_nlp.py); every run below starts exactly from its nominal vector ===")
    assert np.array_equal(shortcut_init(seed), path_init(after_pts)), "the shortcut start differs from stage 3's path"
    results = {}

    def run(cid, quiet=False):
        z, lam, kap, info = solve_komo(cid)
        rep = physical_report(cid, z, lam)
        results[cid] = {"z": z, "lam": lam, "kap": kap, "pts": rep["pts"], "report": rep, "evals": info["evals"], "outer_iters": info["outer_iters"],
                        "stop": info["stop"], "feasible": info["feasible"], "sos": info["errors_buffered"]["sos"], "ineq": info["errors_buffered"]["ineq"],
                        "eq": info["errors_buffered"]["eq"], "x_start": info["x_start"], "options_passed": info["options_passed"], "start_kind": info["start_kind"]}
        if not quiet:
            r = results[cid]
            print(f"    {cid} ({info['start_kind']}): {r['evals']} evaluations, {r['outer_iters']} multiplier updates, stop '{r['stop']}', RAI's feasible flag {r['feasible']}; "
                  f"RAI's summed errors sos {r['sos']:.6f}, ineq {r['ineq']:.3e}, eq {r['eq']:.3e}")
            print("      " + report_words(rep))
        return results[cid]

    print("\n  the default call (RAI's untouched options; the stop is a step test, so the residuals are checked in task units):")
    for key, label in (("default_line", "straight joint-space line"), ("default_shortcut", f"shortcut sampled path of seed {seed}, resampled by arc length to {T_K + 1} configurations (the fixed start and {T_K} free knots)")):
        cid = ids[key]
        z0 = linear_init() if key == "default_line" else shortcut_init(seed)
        init_pts = [tuple(q) for q in knots(z0)]
        vi = validate_path(init_pts)
        g0, _ = ineq_constraints(z0)
        h0, _ = eq_constraints(z0)
        print(f"\n    {cid}, initialization from the {label}:")
        print(f"      {cid} start: cost {cost(z0):.4f}; clearance rows: {'deepest violation ' + f'{g0.max() * 1000:.1f} mm beyond the ring (' + R.depth_words(-g0.max()) + ')' if g0.max() > 0 else 'all satisfied, the closest knot ' + f'{-g0.max() * 1000:.2f} mm outside the ring'}; "
              f"rest rows (final velocity): max |h| {np.abs(h0[2:]).max():.3g}; knot-to-knot edges {revalidation_words(vi)}")
        run(cid)
        results[cid].update(init_pts=init_pts, init_validation=vi, init_cost=cost(z0), init_max_g=float(g0.max()))
    same = max(R.dist(a, b) for a, b in zip(results[ids["default_line"]]["pts"], results[ids["default_shortcut"]]["pts"])) / DEG
    print(f"\n    {ids['default_line']} against {ids['default_shortcut']}: the final knot sequences differ by at most {same:.1e} deg at a knot "
          f"({'the same end point' if same < 1e-3 else 'different end points'})")

    print("\n  the converged reference (RAI's schedule, every constraint row times 10, tight stops; the physical multiplier is 10 times RAI's dual):")
    for key in ("ref_line", "ref_shortcut"):
        run(ids[key])
    same_ref = max(R.dist(a, b) for a, b in zip(results[ids["ref_line"]]["pts"], results[ids["ref_shortcut"]]["pts"])) / DEG
    print(f"    {ids['ref_line']} against {ids['ref_shortcut']}: the final knot sequences differ by at most {same_ref:.1e} deg at a knot")

    print("\n  more knots, T = 40, from the straight line:")
    for cid in ("W-auglag-tight-line-s10-T40", "W-auglag-default-line-T40"):
        run(cid)
    print("    (W-auglag-default-line-T40 is a stopping observation of the default call, not the knot-count lesson.)")

    print("\n  RRT seeds 1 to 12 (RRT, shortcut, then KOMO from the resampled path):")
    sweep = []
    for sd in range(1, 13):
        shortcut_init(sd)
        row = {"seed": sd, "shortcut_deg": path_length(_shortcut_cache[sd]) / DEG}
        for kind, cid in (("reference", f"W-auglag-tight-shortcut-s10-rrt{sd}"), ("default", f"W-auglag-default-shortcut-rrt{sd}")):
            r = run(cid, quiet=True)
            base = results[ids["ref_line"] if kind == "reference" else ids["default_line"]]["pts"]
            row[kind] = {"case": cid, "evals": r["evals"], "stop": r["stop"], "cost": r["report"]["cost"], "knot_deficit_mm": r["report"]["knot_deficit_mm"],
                         "max_knot_diff_deg": max(R.dist(tuple(a), tuple(b)) for a, b in zip(r["pts"], base)) / DEG}
        sweep.append(row)
        print(f"    seed {sd:2d}, shortcut {row['shortcut_deg']:.1f} deg: {row['reference']['case']}: {row['reference']['evals']} evaluations, cost {row['reference']['cost']:.6f}, "
              f"knot deficit {mm_words(row['reference']['knot_deficit_mm'])}, within {row['reference']['max_knot_diff_deg']:.1e} deg of {ids['ref_line']}; "
              f"{row['default']['case']}: {row['default']['evals']} evaluations, cost {row['default']['cost']:.6f}, knot deficit {mm_words(row['default']['knot_deficit_mm'])}, "
              f"within {row['default']['max_knot_diff_deg']:.1e} deg of {ids['default_line']}")
    data["komo"] = {"ids": ids, "runs": {cid: {k: v for k, v in r.items() if k not in ("z", "lam", "kap")} | {"z": r["z"].tolist(), "lam": r["lam"].tolist()} for cid, r in results.items()},
                    "sweep": sweep, "same_default_deg": same, "same_reference_deg": same_ref}

    if args.figures:
        if args.out is None:
            print("\n  --figures without --out: nothing written (new figures are staged outside modules/motion-planning/fig/; pass --out DIR)")
        else:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            written = figures(data, out, stages=(4,) if args.stage == "4" else (1, 2, 3, 4))
            print(f"\n  figures written to {out}: {', '.join(written)}")
    if args.json:
        def conv(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [conv(v) for v in o]
            return o
        with open(args.json, "w") as f:
            json.dump(conv(data), f)
        print(f"  data written to {args.json}")


if __name__ == "__main__":
    main()
