"""Reference implementation of the sampling demo in modules/motion-planning/index.html: a rapidly-exploring random
tree (RRT) in the joint space of the two-link arm (l1 = 0.4 m, l2 = 0.3 m) around the disc of Deck 05's trajectory
example, with the whole links checked as line segments. Every number the browser page shows comes from this exact
procedure, so the page and this script can disagree only if one of them is wrong; check_module.mjs compares them.

Scene (the KOMO worked example of Deck 06, code/komo_two_link.py): start q_start = (30 deg, 45 deg); goal = the positive-elbow inverse-
kinematics solution of the tip target y* = (0.5, 0.2) m, q_goal = (-11.517 deg, 80.406 deg); disc center c = (0.49, 0.35) m,
radius r_o = 0.05 m. A configuration is valid when both joint coordinates lie in [-pi, pi] (no wrapping) and every point
of both links (the segments base-elbow and elbow-tip) is at least r_o + 0.005 m from c: a 5 mm clearance margin.
The older three-point model (tip, elbow, midpoint of link 2, the constraint rows of komo_two_link.py) is kept as a
deliberate counterexample (--checker points): it accepts configurations in which link 2 crosses the disc. The points keep
the same 5 mm margin as the links here; the KOMO example itself used no margin, so only the geometry differs.

RRT, exactly as the page runs it (joint coordinates in radians, Euclidean distance, no wrapping):
  iteration i = 1, 2, ..., budget (3000):
    u = rng()                      if u < 0.1 the sample is the goal itself; otherwise
    q_rand = (-pi + 2 pi rng(), -pi + 2 pi rng())            (two more draws, in this order)
    q_near = the tree node nearest to q_rand (lowest index on a tie)
    q_new  = q_rand if |q_rand - q_near| <= step (10 deg), else q_near + step (q_rand - q_near)/|q_rand - q_near|
    edge check q_near -> q_new: n = max(1, ceil(max_i |q_new_i - q_near_i| / (1 deg))) sub-steps (at least one, so
             a zero-length edge still checks its endpoint); the configurations q_near + (j/n)(q_new - q_near), j = 1..n,
             are checked in order, the start being an already valid tree node; the first invalid one rejects the edge
    accepted: q_new joins the tree with parent q_near. Goal test: if |q_new - q_goal| <= step, the edge q_new -> q_goal
             is checked the same way; if valid the goal joins the tree and the search stops (q_new = q_goal counts at once).
  The random generator is mulberry32 (a 32-bit generator with one 32-bit state word), seeded with 7; it is written
  identically here and in the page, so the two trees are the same tree, not two runs of "the same" algorithm.
  The budget stops the search; running out of budget does not show that no path exists.

Run:  .venv/bin/python modules/motion-planning/rrt_two_link.py                 print the teaching states (about a second)
      .venv/bin/python modules/motion-planning/rrt_two_link.py --checker points   the three-point counterexample
      .venv/bin/python modules/motion-planning/rrt_two_link.py --seed 11 --budget 300   another seed, a smaller budget
      .venv/bin/python modules/motion-planning/rrt_two_link.py --json out.json  also write every iteration's event
"""
import argparse
import json
import math
import sys

L1, L2 = 0.4, 0.3
DEG = math.pi / 180
Q_START = (30 * DEG, 45 * DEG)
Y_STAR = (0.5, 0.2)
OBS_C, OBS_R = (0.49, 0.35), 0.05
MARGIN = 0.005                       # the extra clearance [m]
BOUND = math.pi                      # both joints in [-pi, pi], no wrapping
STEP = 10 * DEG                      # at most this far per extension [rad]
P_GOAL = 0.1                         # goal-sampling probability
SUB = 1 * DEG                        # at most this much change per joint between checked configurations [rad]
SEED, BUDGET = 7, 3000


# ----------------------------------------------------------------------------- the random generator (mulberry32)
def mulberry32(seed):
    """The same 32-bit generator as the page's JavaScript, bit for bit: returns u in [0, 1) with 2^32 possible values."""
    M = 0xFFFFFFFF
    a = seed & M

    def rng():
        nonlocal a
        a = (a + 0x6D2B79F5) & M
        t = ((a ^ (a >> 15)) * (1 | a)) & M
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M)) ^ t) & M
        return ((t ^ (t >> 14)) & M) / 4294967296
    return rng


# ----------------------------------------------------------------------------- kinematics and the two checkers
def ik_positive_elbow(y):
    c2 = (y[0] * y[0] + y[1] * y[1] - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    q2 = math.acos(c2)
    q1 = math.atan2(y[1], y[0]) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return (q1, q2)


Q_GOAL = ik_positive_elbow(Y_STAR)


def points(q):
    """base, elbow, tip and the midpoint of link 2."""
    e = (L1 * math.cos(q[0]), L1 * math.sin(q[0]))
    t = (e[0] + L2 * math.cos(q[0] + q[1]), e[1] + L2 * math.sin(q[0] + q[1]))
    m = (e[0] + 0.5 * L2 * math.cos(q[0] + q[1]), e[1] + 0.5 * L2 * math.sin(q[0] + q[1]))
    return (0.0, 0.0), e, t, m


def segment_distance(a, b, c):
    """Distance from the point c to the segment ab, and the closest point."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    dd = dx * dx + dy * dy
    s = ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / dd if dd > 0 else 0.0
    s = 0.0 if s < 0 else 1.0 if s > 1 else s
    p = (a[0] + s * dx, a[1] + s * dy)
    return math.sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2), p


def clearance(q, checker="links"):
    """The smallest distance from the modeled arm to the disc's surface minus the margin, in meters (negative = invalid),
    with the offending part and its closest point. 'links': both links as segments; 'points': tip, elbow, mid of link 2."""
    base, e, t, m = points(q)
    if checker == "links":
        cands = (("link 1", segment_distance(base, e, OBS_C)), ("link 2", segment_distance(e, t, OBS_C)))
    else:
        cands = tuple((name, (math.sqrt((p[0] - OBS_C[0]) ** 2 + (p[1] - OBS_C[1]) ** 2), p))
                      for name, p in (("tip", t), ("elbow", e), ("mid of link 2", m)))
    name, (d, p) = min(cands, key=lambda c: c[1][0])
    return d - OBS_R - MARGIN, name, p


def in_bounds(q):
    return -BOUND <= q[0] <= BOUND and -BOUND <= q[1] <= BOUND


def valid(q, checker="links"):
    if not in_bounds(q):
        return False, "outside the joint box", None, None
    c, part, p = clearance(q, checker)
    return c >= 0, part, c, p


# ----------------------------------------------------------------------------- the edge check and the tree
def dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def check_edge(qa, qb, checker="links"):
    """Sub-divide qa -> qb so that no joint changes by more than SUB between checked configurations; qa is assumed valid.
    Returns (ok, n, j, q_fail, part, clearance_at_fail, min_clearance_over_checked_configurations)."""
    n = max(1, math.ceil(max(abs(qb[0] - qa[0]), abs(qb[1] - qa[1])) / SUB))
    worst = math.inf
    for j in range(1, n + 1):
        q = (qa[0] + (j / n) * (qb[0] - qa[0]), qa[1] + (j / n) * (qb[1] - qa[1]))
        ok, part, c, p = valid(q, checker)
        if c is not None:
            worst = min(worst, c)
        if not ok:
            return False, n, j, q, part, c, worst
    return True, n, None, None, None, None, worst


def deepest_along(qa, qb, checker="links"):
    """Diagnostic: the smallest clearance over all checked configurations of the edge qa -> qb (the checker itself stops at
    the first invalid one). Returns (clearance, j, n, q, part)."""
    n = max(1, math.ceil(max(abs(qb[0] - qa[0]), abs(qb[1] - qa[1])) / SUB))
    best = None
    for j in range(0, n + 1):
        q = (qa[0] + (j / n) * (qb[0] - qa[0]), qa[1] + (j / n) * (qb[1] - qa[1]))
        c, part, _ = clearance(q, checker)
        if best is None or c < best[0]:
            best = (c, j, n, q, part)
    return best


def edge_table(qa, qb, checker="links"):
    """Every scheduled configuration of the edge qa -> qb, j = 0..n, with its clearance (the edge-check view of the page,
    ?mode=edge). Index 0 is qa, which the edge check assumes valid; the check itself evaluates j = 1, 2, ... and stops at
    the first invalid one. Returns (rows, first_failure_j or None, deepest_j); each row is (j, q, clearance, part)."""
    n = max(1, math.ceil(max(abs(qb[0] - qa[0]), abs(qb[1] - qa[1])) / SUB))
    rows = []
    for j in range(0, n + 1):
        q = (qa[0] + (j / n) * (qb[0] - qa[0]), qa[1] + (j / n) * (qb[1] - qa[1]))
        c, part, _ = clearance(q, checker)
        rows.append((j, q, c, part))
    first = next((j for j, _, c, _ in rows[1:] if c < 0), None)
    deepest = min(rows, key=lambda r: r[2])[0]
    return rows, first, deepest


def rrt(seed=SEED, budget=BUDGET, checker="links", trace=None):
    """Grow the tree; return a dict with the nodes, parents, the outcome and the path (goal first... start last reversed)."""
    rng = mulberry32(seed)
    nodes, parents = [Q_START], [-1]
    outcome, goal_index, iterations = "budget", None, 0
    for i in range(1, budget + 1):
        iterations = i
        u = rng()
        if u < P_GOAL:
            q_rand, kind = Q_GOAL, "goal"
        else:
            q_rand, kind = (-math.pi + 2 * math.pi * rng(), -math.pi + 2 * math.pi * rng()), "uniform"
        near = min(range(len(nodes)), key=lambda k: dist(q_rand, nodes[k]))     # min keeps the lowest index on a tie
        q_near = nodes[near]
        d = dist(q_rand, q_near)
        if d <= STEP:
            q_new = q_rand
        else:
            q_new = (q_near[0] + STEP * (q_rand[0] - q_near[0]) / d, q_near[1] + STEP * (q_rand[1] - q_near[1]) / d)
        ev = {"i": i, "kind": kind, "u": u, "q_rand": q_rand, "near": near, "q_new": q_new, "d": d}
        ok, n, j, q_fail, part, c_fail, worst = check_edge(q_near, q_new, checker)
        ev.update({"n": n, "result": "accepted" if ok else "rejected", "j": j, "q_fail": q_fail, "part": part,
                   "c_fail": c_fail, "min_clearance": worst, "goal": False})
        if ok:
            nodes.append(q_new)
            parents.append(near)
            k = len(nodes) - 1
            if q_new == Q_GOAL:
                outcome, goal_index = "connected", k
                ev["goal"] = True
            elif dist(q_new, Q_GOAL) <= STEP:
                g_ok, gn, gj, gq, gpart, gc, gworst = check_edge(q_new, Q_GOAL, checker)
                ev["goal_try"] = {"n": gn, "result": "accepted" if g_ok else "rejected", "j": gj, "q_fail": gq,
                                  "part": gpart, "c_fail": gc}
                if g_ok:
                    nodes.append(Q_GOAL)
                    parents.append(k)
                    outcome, goal_index = "connected", len(nodes) - 1
                    ev["goal"] = True
        if trace is not None:
            trace.append(ev)
        if outcome == "connected":
            break
    path = []
    if goal_index is not None:
        k = goal_index
        while k >= 0:
            path.append(k)
            k = parents[k]
        path.reverse()
    return {"nodes": nodes, "parents": parents, "outcome": outcome, "iterations": iterations, "path": path,
            "seed": seed, "budget": budget, "checker": checker}


def revalidate(nodes, path, checker="links", sub=SUB):
    """Two things on every edge of the path, with the same subdivision rule (optionally finer). First, the planner's
    own edge test is repeated: it stops at the first invalid configuration of an edge, and `evaluated` counts the
    configurations it actually evaluated. Second, a complete scan of every scheduled configuration of every edge
    (j = 0..n, no early stop) gives `worst`, the smallest clearance beyond the margin on the whole sampled path.
    Returns (worst, evaluated, first failure or None). scheduled_count() gives the number of scheduled configurations."""
    global SUB
    saved, SUB = SUB, sub
    try:
        worst, evaluated, fail = math.inf, 0, None
        for a, b in zip(path, path[1:]):
            ok, n, j, q_fail, part, c_fail, _ = check_edge(nodes[a], nodes[b], checker)
            evaluated += n if ok else j
            worst = min(worst, deepest_along(nodes[a], nodes[b], checker)[0])
            if not ok and fail is None:
                fail = {"edge": (a, b), "j": j, "n": n, "q": q_fail, "part": part, "clearance": c_fail}
        return worst, evaluated, fail
    finally:
        SUB = saved


def scheduled_count(nodes, path, sub=SUB):
    """The number of configurations the subdivision rule schedules on the path's edges (j = 1..n on each edge)."""
    return sum(max(1, math.ceil(max(abs(nodes[b][0] - nodes[a][0]), abs(nodes[b][1] - nodes[a][1])) / sub))
               for a, b in zip(path, path[1:]))


def path_deepest(nodes, path, checker="links"):
    """Diagnostic: the configuration of least clearance over EVERY checked configuration of every path edge (j = 0..n on
    each edge), without the early stop of check_edge. Returns (clearance, edge index e, j, n, q, part)."""
    best = None
    for e, (a, b) in enumerate(zip(path, path[1:])):
        c, j, n, q, part = deepest_along(nodes[a], nodes[b], checker)
        if best is None or c < best[0]:
            best = (c, e, j, n, q, part)
    return best


def depth_words(c):
    """How far inside: 'X mm inside the disc' when the surface is crossed, else 'X mm inside the 5 mm clearance ring'."""
    inside = -c - MARGIN
    return f"{inside * 1000:.1f} mm inside the disc" if inside > 0 else f"{-c * 1000:.1f} mm inside the 5 mm clearance ring, outside the disc"


def path_length(nodes, path):
    return sum(dist(nodes[a], nodes[b]) for a, b in zip(path, path[1:]))


def forbidden_fraction(checker, res_deg=2):
    """Fraction of the joint box that the checker forbids, on a grid of res_deg degrees."""
    n = int(round(360 / res_deg)) + 1
    bad = 0
    for i in range(n):
        for j in range(n):
            q = (-math.pi + i * res_deg * DEG, -math.pi + j * res_deg * DEG)
            if not valid(q, checker)[0]:
                bad += 1
    return bad / (n * n)


# ----------------------------------------------------------------------------- print mode
def deg(q):
    return f"({q[0] / DEG:.2f}°, {q[1] / DEG:.2f}°)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checker", choices=("links", "points"), default="links")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--json", help="write the full trace and results to this file")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sweep", type=int, metavar="N", help="run seeds 1..N under both checkers and print one line each")
    ap.add_argument("--edge", action="store_true", help="print every scheduled check of the direct edge start -> goal (the page's ?mode=edge)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    checker = args.checker
    out = print if not args.quiet else (lambda *a, **k: None)
    if args.edge:
        rows, first, deepest = edge_table(Q_START, Q_GOAL, checker)
        n = len(rows) - 1
        out(f"=== the direct edge start -> goal, checker {checker}: {n} subintervals of at most {SUB / DEG:.0f}° per joint, checks 0..{n} ===")
        out("     j      s     q1 (deg)  q2 (deg)   clearance from the disc surface (mm)   part      verdict")
        for j, q, c, part in rows:
            verdict = "start (assumed valid)" if j == 0 else ("valid" if c >= 0 else "INVALID")
            if first is not None and j > first:
                verdict += ", not evaluated by the edge check"
            out(f"  {j:4d}  {j / n:5.3f}  {q[0] / DEG:8.3f}  {q[1] / DEG:8.3f}   {(c + MARGIN) * 1000:9.2f}                              {part:8s}  {verdict}")
        if first is not None:
            c, part = rows[first][2], rows[first][3]
            out(f"  first failure: check {first} of {n}, {part} is {(c + MARGIN) * 1000:.2f} mm from the disc surface "
                f"({-c * 1000:.2f} mm inside the {MARGIN * 1000:.0f} mm clearance ring); the edge check stops and rejects the edge")
        else:
            out("  every check passes; this proves clearance only at the checked postures")
        out(f"  deepest check: {deepest} of {n}, {rows[deepest][3]} {depth_words(rows[deepest][2])}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"checker": checker, "n": n, "first_failure": first, "deepest": deepest,
                           "rows": [{"j": j, "q": q, "c": c, "part": part} for j, q, c, part in rows]}, f)
            out(f"  table written to {args.json}")
        return
    if args.sweep:
        for ch in ("links", "points"):
            print(f"=== seeds 1..{args.sweep}, checker {ch}, budget {args.budget} ===")
            for seed in range(1, args.sweep + 1):
                r = rrt(seed, args.budget, ch)
                line = f"  seed {seed:3d}: "
                if r["outcome"] != "connected":
                    print(line + f"budget of {args.budget} exhausted, {len(r['nodes'])} nodes"); continue
                line += f"connected at iteration {r['iterations']:4d}, {len(r['nodes']):3d} nodes, {len(r['path']) - 1:2d} edges, {path_length(r['nodes'], r['path']) / DEG:6.2f}°"
                if ch == "points":
                    _, _, fail_l = revalidate(r["nodes"], r["path"], "links")
                    dc, de, dj, dn, dq, dpart = path_deepest(r["nodes"], r["path"], "links")
                    line += "; whole-link check " + ("valid" if fail_l is None else f"INVALID, deepest {depth_words(dc)}")
                print(line)
        return

    out(f"=== the scene (checker: {'whole links as segments' if checker == 'links' else 'three points: tip, elbow, mid of link 2'}, margin {MARGIN * 1000:.0f} mm) ===")
    out(f"  start q_start = {deg(Q_START)}, goal q_goal = {deg(Q_GOAL)} (positive-elbow IK of y* = {Y_STAR}),  |q_goal - q_start| = {dist(Q_START, Q_GOAL) / DEG:.2f}°")
    for name, q in (("start", Q_START), ("goal", Q_GOAL)):
        c, part, _ = clearance(q, checker)
        out(f"  {name}: clearance {(c + MARGIN) * 1000:.1f} mm from the disc surface, {MARGIN * 1000:.0f} mm needed (closest part: {part}) -> {'valid' if c >= 0 else 'invalid'}")
    ok, n, j, qf, part, cf, worst = check_edge(Q_START, Q_GOAL, checker)
    out(f"  direct edge start -> goal: {n} sub-steps of at most 1° per joint; "
        + (f"invalid at sub-configuration {j} of {n}, q = {deg(qf)}: {part} is {-cf * 1000:.1f} mm inside the clearance ring"
           if not ok else f"valid (smallest clearance {(worst + MARGIN) * 1000:.1f} mm from the disc surface)"))
    deep_c, deep_j, deep_n, deep_q, deep_part = deepest_along(Q_START, Q_GOAL, "links")
    out(f"  deepest point of the direct edge under the whole-link checker: sub-configuration {deep_j} of {deep_n}, q = {deg(deep_q)}: "
        f"{deep_part} reaches {(-deep_c - MARGIN) * 1000:.1f} mm inside the disc")
    out(f"  forbidden fraction of the joint box on a 2° grid: whole links {forbidden_fraction('links'):.3f}, three points {forbidden_fraction('points'):.3f}")

    trace = []
    res = rrt(args.seed, args.budget, checker, trace)
    nodes, path = res["nodes"], res["path"]
    out(f"\n=== RRT: seed {args.seed}, budget {args.budget}, step {STEP / DEG:.0f}°, goal sampling {P_GOAL:.0%}, edge sub-steps of at most {SUB / DEG:.0f}° per joint ===")
    accepted = sum(e["result"] == "accepted" for e in trace)
    goal_samples = sum(e["kind"] == "goal" for e in trace)
    out(f"  {res['iterations']} iterations: {accepted} accepted, {res['iterations'] - accepted} rejected; {goal_samples} goal samples; {len(nodes)} nodes")
    out("  first iterations:")
    for e in trace[:12]:
        line = f"    i={e['i']:3d} {e['kind']:7s} q_rand={deg(e['q_rand'])} nearest #{e['near']} |q_rand-q_near|={e['d'] / DEG:6.2f}° q_new={deg(e['q_new'])} n={e['n']:2d} {e['result']}"
        if e["result"] == "rejected":
            line += f" at j={e['j']}: {e['part']} {-e['c_fail'] * 1000:.1f} mm too close"
        if e.get("goal_try"):
            g = e["goal_try"]
            line += f"; goal edge {g['result']}" + (f" at j={g['j']} of {g['n']}" if g["j"] else "")
        out(line)
    if res["outcome"] == "connected":
        out(f"\n  goal connected at iteration {res['iterations']} with {len(nodes)} nodes; path of {len(path) - 1} edges, joint-space length {path_length(nodes, path) / DEG:.2f}° (the direct edge would be {dist(Q_START, Q_GOAL) / DEG:.2f}°)")
        out("  path (node: q):")
        for k in path:
            out(f"    #{k:4d} {deg(nodes[k])}")
        worst, evaluated, fail = revalidate(nodes, path, checker)
        sched = scheduled_count(nodes, path)
        out(f"  revalidation with the same checker and resolution: across all edges the repeated edge tests evaluate {evaluated} of {sched} scheduled configurations"
            + (" without a failure" if fail is None else f", each failing edge stopping at its first invalid one; the first failure in path order is edge #{fail['edge'][0]}->#{fail['edge'][1]}, check {fail['j']} of {fail['n']}") + f"; the complete scan of all {sched} finds the smallest clearance {(worst + MARGIN) * 1000:.1f} mm from the disc surface ({MARGIN * 1000:.0f} mm needed) -> {'valid' if fail is None else 'INVALID'}")
        worst_f, evaluated_f, fail_f = revalidate(nodes, path, checker, sub=0.05 * DEG)
        sched_f = scheduled_count(nodes, path, sub=0.05 * DEG)
        out(f"  finer diagnostic at 0.05° per joint ({evaluated_f} of {sched_f} scheduled configurations evaluated): smallest clearance over all of them {(worst_f + MARGIN) * 1000:.1f} mm from the disc surface -> {'still valid' if fail_f is None else 'INVALID: the 1° check missed a collision'}")
        worst_l, evaluated_l, fail_l = revalidate(nodes, path, "links")
        res["deepest_links"] = path_deepest(nodes, path, "links")
        out(f"  the same path under the whole-link checker: across all edges the repeated edge tests evaluate {evaluated_l} of {sched} scheduled configurations"
            + ("; the complete scan of all of them finds the smallest clearance " + f"{(worst_l + MARGIN) * 1000:.1f} mm from the disc surface -> valid" if fail_l is None
               else f", each failing edge stopping at its first invalid one; the first failure in path order is edge #{fail_l['edge'][0]}->#{fail_l['edge'][1]}, check {fail_l['j']} of {fail_l['n']}, q = {deg(fail_l['q'])}, {fail_l['part']} {-fail_l['clearance'] * 1000:.1f} mm inside the clearance ring; the complete scan of all {sched} finds the smallest clearance {depth_words(worst_l)} -> INVALID"))
        if fail_l is not None:
            dc, de, dj, dn, dq, dpart = path_deepest(nodes, path, "links")
            where = f"node #{path[de + 1]}" if dj == dn else f"node #{path[de]}" if dj == 0 else f"configuration {dj} of {dn} on edge #{path[de]}->#{path[de + 1]}"
            out(f"  deepest point of that path (every checked configuration of every edge, no early stop): {where}, q = {deg(dq)}: {dpart} {depth_words(dc)}")
    else:
        out(f"\n  budget of {args.budget} iterations exhausted: {len(nodes)} nodes, no connection to the goal. This does not show that no path exists.")
    rejections = [e for e in trace if e["result"] == "rejected"]
    if rejections:
        e = rejections[0]
        out(f"  first rejection: iteration {e['i']}, sub-configuration {e['j']} of {e['n']}, {e['part']} {-e['c_fail'] * 1000:.1f} mm too close")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"constants": {"L1": L1, "L2": L2, "q_start": Q_START, "q_goal": Q_GOAL, "obs_c": OBS_C, "obs_r": OBS_R,
                                     "margin": MARGIN, "step": STEP, "p_goal": P_GOAL, "sub": SUB},
                       "result": res, "trace": trace}, f)
        out(f"  trace written to {args.json}")


if __name__ == "__main__":
    main()
