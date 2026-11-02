"""RAI-style bidirectional RRT on the sampling demo's scene: a NumPy-free port of rai::RRT_PathFinder (robotic 0.4.1,
rai @ 88b209f, src/PathAlgos/RRT_PathFinder.cpp and ConfigurationProblem.cpp; copies in
planning/rrt-bidir-shortcut-rrtstar-2026-10-31/upstream/), with the demo's random generator (mulberry32) in place of
RAI's. The browser view (index.html?mode=bidir) implements the same procedure and is checked against this file.

The procedure, as RAI implements it:
  - Two trees, rooted at the start and at the goal. One iteration is one call of RAI's stepConnect: the start tree
    makes one extension attempt toward the goal tree; only if that attempt does not join the trees does the goal tree
    make one attempt toward the start tree. Iterations stop at a join or when the iteration count exceeds maxIters.
  - An attempt by tree A toward tree B: with probability p_connect the target is a node of B chosen uniformly at
    random, otherwise a configuration drawn uniformly in the joint box. The proposal moves from A's nearest node toward
    the target by at most `stepsize` (Euclidean joint-space distance).
  - Checks (one query each): the proposal itself first; if it passes, the configurations at fractions 0.5, 0.25 and
    0.75 of the edge from the nearest node (van der Corput order, subsamples = 4). The first failing query rejects the
    attempt. The nearest node is not queried again.
  - A query sums the penetration of both links into the disc model; with RAI's geometry (capsules of radius EPS, a
    disc sphere of radius r_o + 5 mm - EPS) a link's penetration is its deficit from the 5 mm requirement. The query
    passes when the sum is below collisionTolerance = 1e-4 m (0.1 mm). (RAI raises the tolerance for one attempt when
    the nearest node itself exceeds it; the port keeps that rule, which cannot trigger here because both roots pass.)
  - After adding the proposal q, the trees are joined when B's node nearest to q lies closer than
    stepsize / subsamples = 0.025 rad. RAI runs no edge test from q to that node.
  - The returned path follows A's parents from q's PARENT (RAI's nearestID still points there) and B's parents from
    B's nearest node, ordered start to goal. The node q is therefore omitted from the returned vertex list, and RAI
    does not run its edge test on the joining segment parent(q) -> nearest_B(q), up to stepsize + 0.025 rad long.
    Three cases occur. If q coincides with nearest_B(q), the joining segment is the extension edge the attempt
    queried. If q lies on the joining segment, the attempt's queries cover its part from parent(q) to q, and the
    rest, shorter than 0.025 rad, lies outside that attempt. If q lies off the segment, the attempt's queries lie off
    it too. In the last two cases RAI did not check the complete joining connection (join_case() classifies a run).
  - Counts: RAI's ret.evals is the iteration count (maxIters + 1 when the budget is exhausted); its printed query
    count includes the two queries of the roots. The port reports the same numbers and its own extension attempts.

Native RAI evidence: rai_bidir_native.py (tolerance mapping, ordinary runs at rnd_seed 1..N, budget exhaustion).
RAI's random stream differs from mulberry32, so a seed here does not reproduce a native run; the two are compared by
their distributions (--compare). With stream="rai" the port draws RAI's own stream, and --replay reruns a native
record: counts must match exactly and path vertices within 1e-5 deg (the record stores them rounded to 1e-6 deg).
--replay exits with status 1 if any record differs; --replay-selftest also checks that two corrupted records fail.

Run:  .venv/bin/python modules/motion-planning/rai_bidir_rrt.py                 seed 7, the trace summary and the checks
      .venv/bin/python modules/motion-planning/rai_bidir_rrt.py --seed 3 --trace  print every attempt
      .venv/bin/python modules/motion-planning/rai_bidir_rrt.py --json out.json  dump the run (the browser check reads it)
      .venv/bin/python modules/motion-planning/rai_bidir_rrt.py --sweep 200 --compare native.json
      .venv/bin/python modules/motion-planning/rai_bidir_rrt.py --replay native.json --replay-selftest
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrt_two_link as R                                   # the demo's scene, generator and course edge check

DEG = R.DEG
Q_START, Q_GOAL = R.Q_START, R.Q_GOAL
OPTIONS = {"stepsize": 0.1, "subsamples": 4, "maxIters": 5000, "p_connect": 0.5, "collisionTolerance": 1e-4}
SEED = 7


class Mulberry:
    """The demo's generator (mulberry32): uni() in [0, 1); a node index as floor(uni() * n)."""
    def __init__(self, seed):
        self.r = R.mulberry32(seed)

    def uni(self):
        return self.r()

    def index(self, n):
        return math.floor(self.r() * n)


class RaiStream:
    """RAI's own random stream, for replaying native runs: rai::Rnd is std::minstd_rand (x <- 48271 x mod 2^31 - 1)
    seeded by ry.rnd_seed, read through libstdc++'s uniform_real_distribution (generate_canonical with two draws)
    and uniform_int_distribution (rejection downscaling). This reproduces native RAI only if the wheel was built with
    these library algorithms; the replay check (--replay) is what establishes it."""
    M = 2147483647

    def __init__(self, seed):
        self.x = seed % self.M or 1

    def _next(self):
        self.x = (48271 * self.x) % self.M
        return self.x

    def uni(self):
        r, total, tmp = 2147483646.0, 0.0, 1.0
        for _ in range(2):
            total += float(self._next() - 1) * tmp
            tmp *= r
        u = total / tmp
        return u if u < 1.0 else math.nextafter(1.0, 0.0)

    def index(self, n):
        urngrange, uerange = 2147483645, n
        scaling = urngrange // uerange
        past = uerange * scaling
        while True:
            v = self._next() - 1
            if v < past:
                return v // scaling


def corput(n, base=2):
    q, bk = 0.0, 1.0 / base
    while n > 0:
        q += (n % base) * bk
        n //= base
        bk /= base
    return q


def penetration(q):
    """RAI's query on this scene: the sum over both links of the deficit from the 5 mm requirement (m), and per link."""
    base, e, t, _ = R.points(q)
    per = [max(0.0, R.OBS_R + R.MARGIN - R.segment_distance(a, b, R.OBS_C)[0]) for a, b in ((base, e), (e, t))]
    return per[0] + per[1], per


class Tree:
    def __init__(self, name, root):
        self.name, self.nodes, self.parents, self.pen = name, [root], [0], [penetration(root)[0]]   # RAI: the root's parent is 0
        self.nearest_id = None

    def nearest(self, q):
        best, bd = 0, R.dist(q, self.nodes[0])
        for i in range(1, len(self.nodes)):
            d = R.dist(q, self.nodes[i])
            if d < bd:
                best, bd = i, d
        return best, bd

    def path_from(self, i):
        out = []
        while True:
            out.append(self.nodes[i])
            if i == 0:
                return out
            i = self.parents[i]


def attempt(A, B, rng, opt, counter, it):
    """One call of RAI's growTreeToTree(A, B). Returns (joined, event)."""
    ev = {"i": it, "tree": A.name}
    if rng.uni() < opt["p_connect"]:
        k = rng.index(len(B.nodes))
        target, ev["source"], ev["target_node"] = B.nodes[k], "node", k
    else:
        target, ev["source"] = (-math.pi + rng.uni() * (2 * math.pi), -math.pi + rng.uni() * (2 * math.pi)), "uniform"
    ev["target"] = target
    near, d = A.nearest(target)
    A.nearest_id = near
    q_near = A.nodes[near]
    s = opt["stepsize"] / d if d > opt["stepsize"] else 1.0
    q = (q_near[0] + s * (target[0] - q_near[0]), q_near[1] + s * (target[1] - q_near[1]))
    ev.update({"near": near, "proposal": q, "step": R.dist(q, q_near)})
    tol = opt["collisionTolerance"]
    if A.pen[near] > tol:                       # RAI's special rule for a nearest node that is itself in collision
        tol = A.pen[near] + 1e-6
    queries = []
    fracs = [1.0] + ([corput(i) for i in range(1, opt["subsamples"])] if opt["subsamples"] > 0 else [])
    ok = True
    for f in fracs:
        qf = q if f == 1.0 else (q_near[0] + f * (q[0] - q_near[0]), q_near[1] + f * (q[1] - q_near[1]))
        pen, per = penetration(qf)
        counter["queries"] += 1
        passed = pen < tol
        queries.append({"f": f, "q": qf, "pen": pen, "per_link": per, "pass": passed})
        if not passed:
            ok = False
            break
    ev["queries"] = queries
    if not ok:
        ev["decision"] = "endpoint rejected" if queries[-1]["f"] == 1.0 else "edge rejected"
        return False, ev
    A.nodes.append(q); A.parents.append(near); A.pen.append(queries[0]["pen"])
    ev["decision"], ev["new"] = "added", len(A.nodes) - 1
    nb, db = B.nearest(q)
    B.nearest_id = nb
    thr = opt["stepsize"] / opt["subsamples"] if opt["subsamples"] > 0 else opt["stepsize"]
    ev["join"] = {"node": nb, "dist": db, "threshold": thr, "joined": db < thr}
    return db < thr, ev


def plan(seed=SEED, stream="mulberry32", **kw):
    opt = dict(OPTIONS); opt.update(kw)
    rng = RaiStream(seed) if stream == "rai" else Mulberry(seed)
    S, G = Tree("start", Q_START), Tree("goal", Q_GOAL)
    counter = {"queries": 2}                    # RAI queries both roots in setStartGoal
    trace, iters, joined, last = [], 0, False, None
    while True:
        iters += 1
        if iters > opt["maxIters"]:
            break
        joined, ev = attempt(S, G, rng, opt, counter, iters)
        trace.append(ev); last = ev
        if not joined:
            joined, ev = attempt(G, S, rng, opt, counter, iters)
            trace.append(ev); last = ev
        if joined:
            break
    out = {"seed": seed, "options": opt, "start": S, "goal": G, "trace": trace, "joined": joined,
           "iterations": iters, "rai_evals": iters, "queries": counter["queries"], "attempts": len(trace)}
    if joined:
        A, B = (S, G) if last["tree"] == "start" else (G, S)
        # RAI: path = rrt0.getPathFromNode(rrt0.nearestID) reversed, then rrtT.getPathFromNode(rrtT.nearestID)
        p0 = list(reversed(S.path_from(S.nearest_id)))
        pT = G.path_from(G.nearest_id)
        out["path"] = p0 + pT
        par = A.parents[last["new"]]
        out["join"] = {"tree": A.name, "new": last["new"], "q_new": A.nodes[last["new"]], "parent": par,
                       "other": B.name, "other_node": last["join"]["node"], "dist_new_to_other": last["join"]["dist"],
                       "segment": (A.nodes[par], B.nodes[last["join"]["node"]]),
                       "segment_length": R.dist(A.nodes[par], B.nodes[last["join"]["node"]]),
                       "index": len(p0) - 1}       # the joining segment is path[index] -> path[index + 1]
    else:
        out["path"] = None
    return out


def join_case(r):
    """Where the joining attempt's new node q lies relative to the joining segment parent(q) -> nearest_B(q):
    ("coincide", 0, 0), ("on-segment", along, 0) or ("off-line", along, offset), with `along` the distance from the
    parent along the segment and `offset` the distance off it [rad]."""
    j = r["join"]
    (ax, ay), (bx, by) = j["segment"]
    qx, qy = j["q_new"]
    if j["dist_new_to_other"] < 1e-12:
        return "coincide", 0.0, 0.0
    ux, uy = (bx - ax) / j["segment_length"], (by - ay) / j["segment_length"]
    along = (qx - ax) * ux + (qy - ay) * uy
    offset = abs((qy - ay) * ux - (qx - ax) * uy)
    if offset < 1e-9 and 0 < along < j["segment_length"]:
        return "on-segment", along, 0.0
    return "off-line", along, offset


def check_path(path, sub_deg=1.0):
    """The course's independent finite check of a returned path: every edge, the joining one included, subdivided so
    that no joint changes by more than sub_deg between checked configurations, each configuration needing 5 mm of
    clearance from the disc. Early stopping per edge (the counts are the configurations actually evaluated); a
    separate complete scan gives the smallest clearance. Neither is a certificate of continuous clearance."""
    first, evaluated, scheduled, worst = None, 0, 0, (math.inf, None, None)
    for e, (a, b) in enumerate(zip(path, path[1:])):
        n = max(1, math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / (sub_deg * DEG)))
        scheduled += n
        stopped = False
        for j in range(1, n + 1):
            q = (a[0] + j / n * (b[0] - a[0]), a[1] + j / n * (b[1] - a[1]))
            c = R.clearance(q, "links")[0]
            if not stopped:
                evaluated += 1
                if c < 0:
                    stopped = True
                    if first is None:
                        first = {"edge": e, "j": j, "n": n, "q": q, "clearance_mm": (c + R.MARGIN) * 1e3}
            if c < worst[0]:
                worst = (c, e, q)
    return {"pass": first is None, "first": first, "evaluated": evaluated, "scheduled": scheduled,
            "min_clearance_mm": (worst[0] + R.MARGIN) * 1e3, "min_edge": worst[1], "min_q": worst[2]}


def summary(r):
    s = {k: r[k] for k in ("seed", "joined", "iterations", "rai_evals", "queries", "attempts")}
    s["tree_sizes"] = [len(r["start"].nodes), len(r["goal"].nodes)]
    if r["path"]:
        p = r["path"]
        steps = [R.dist(a, b) for a, b in zip(p, p[1:])]
        s.update({"vertices": len(p), "length_rad": sum(steps), "max_step": max(steps),
                  "long_steps": [i for i, x in enumerate(steps) if x > r["options"]["stepsize"] + 1e-9],
                  "join_index": r["join"]["index"], "join_length": r["join"]["segment_length"]})
        c1, c005 = check_path(p, 1.0), check_path(p, 0.05)
        s.update({"course_1deg_pass": c1["pass"], "course_1deg_first": c1["first"], "fine_min_clearance_mm": c005["min_clearance_mm"],
                  "fine_min_edge": c005["min_edge"]})
    else:
        s["rai_evals"] = r["iterations"]           # maxIters + 1, as RAI reports
    return s


def describe(ev):
    src = f"node #{ev['target_node']} of the other tree" if ev["source"] == "node" else "a uniform draw"
    q = "; ".join(f"f={x['f']:.2f} pen={x['pen'] * 1e3:.3f} mm {'pass' if x['pass'] else 'FAIL'}" for x in ev["queries"])
    j = ""
    if "join" in ev:
        j = f"; nearest other-tree node #{ev['join']['node']} at {ev['join']['dist']:.4f} rad{' -> JOINED' if ev['join']['joined'] else ''}"
    return (f"it {ev['i']:4} {ev['tree']:5} target {src}, nearest #{ev['near']}, step {ev['step']:.4f} rad -> {ev['decision']}"
            f" [{q}]{j}")


def to_json(r):
    def tree(T):
        return {"nodes": T.nodes, "parents": T.parents}
    return {"seed": r["seed"], "options": r["options"], "start": tree(r["start"]), "goal": tree(r["goal"]), "trace": r["trace"],
            "joined": r["joined"], "iterations": r["iterations"], "queries": r["queries"], "attempts": r["attempts"],
            "path": r["path"], "join": r.get("join"), "join_case": join_case(r) if r["path"] else None, "check_1deg": check_path(r["path"], 1.0) if r["path"] else None,
            "check_005deg": check_path(r["path"], 0.05) if r["path"] else None}


def compare(port, native):
    ok = [x for x in native["runs"] if x["feasible"]]
    pk = [x for x in port if x["joined"]]
    rows = []
    for name, kp, kn in (("iterations", "iterations", "iterations"), ("queries", "queries", "queries"), ("path vertices", "vertices", "vertices"),
                         ("path length [rad]", "length_rad", "length_rad"), ("longest step [rad]", "max_step", "max_step")):
        a, b = [x[kp] for x in pk], [x[kn] for x in ok]
        qa = statistics.quantiles(a, n=4); qb = statistics.quantiles(b, n=4)
        rows.append((name, qa, qb, ks(a, b)))
    print(f"\n== port (mulberry32, seeds 1..{len(port)}) against native RAI (rnd_seed 1..{len(native['runs'])}): quartiles, KS distance ==")
    for name, qa, qb, d in rows:
        print(f"  {name:20} port {fmt(qa)}   native {fmt(qb)}   KS {d:.3f}")
    fa = sum(bool(x["long_steps"]) for x in pk) / len(pk); fb = sum(bool(x["long_steps"]) for x in ok) / len(ok)
    ca = sum(x["course_1deg_pass"] for x in pk) / len(pk); cb = sum(x["course_1deg_pass"] for x in ok) / len(ok)
    print(f"  paths with a step longer than stepsize: port {fa:.2f}, native {fb:.2f}")
    print(f"  paths passing the course's 1 deg check: port {ca:.2f}, native {cb:.2f}")
    print(f"  found a path: port {len(pk)}/{len(port)}, native {len(ok)}/{len(native['runs'])}")
    return rows


def replay(native):
    """Rerun every native seed with RAI's random stream and compare the iterations, the query count, the tree sizes
    and the returned path (native paths are stored in degrees rounded to 1e-6, so they agree to 1e-5 deg)."""
    same, diff = 0, []
    for r in native["runs"] + [dict(b, budget=True) for b in native.get("budget", [])]:
        p = plan(r["seed"], "rai", maxIters=r.get("maxIters", OPTIONS["maxIters"]))
        if r.get("budget"):
            ok = p["joined"] == r["feasible"] and (not p["joined"]) and p["iterations"] == r["iterations"] and r["path_shape"] == [0]
        else:
            ok = p["joined"] == r["feasible"] and p["iterations"] == r["iterations"] and p["queries"] == r["queries"] \
                and [len(p["start"].nodes), len(p["goal"].nodes)] == r["tree_sizes"] and len(p["path"]) == len(r["path_deg"]) \
                and max(abs(x / DEG - y) for a, b in zip(p["path"], r["path_deg"]) for x, y in zip(a, b)) < 1e-5
        same += ok
        if not ok:
            diff.append(r["seed"])
    n = len(native["runs"]) + len(native.get("budget", []))
    print(f"\n== replay with RAI's random stream: {same} of {n} native runs reproduced "
          f"(counts exact: feasibility, iterations, queries, both tree sizes; every path vertex within 1e-5 deg of the "
          f"record, which stores degrees rounded to 1e-6; budget runs: feasible = false, no path, ret.evals)"
          + (f"; differing seeds {diff[:20]}" if diff else "") + " ==")
    return same, diff


def replay_selftest(native):
    """Negative regression cases for --replay: one ordinary record with its query count raised by one, and one budget
    record marked feasible, must each be reported as differing."""
    import copy
    bad = []
    a = copy.deepcopy(native); a["runs"] = a["runs"][:1]; a["budget"] = []; a["runs"][0]["queries"] += 1
    b = copy.deepcopy(native); b["runs"] = []; b["budget"] = b["budget"][:1]; b["budget"][0]["feasible"] = True
    for name, rec in (("query count + 1", a), ("budget record marked feasible", b)):
        same, diff = replay(rec)
        print(f"   selftest, {name}: {'detected' if same == 0 and diff else 'NOT DETECTED'}")
        if not (same == 0 and diff):
            bad.append(name)
    return bad


def ks(a, b):
    a, b = sorted(a), sorted(b)
    pts = sorted(set(a) | set(b))
    import bisect
    return max(abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)) for x in pts)


def fmt(q):
    return "(" + ", ".join(f"{v:.3g}" for v in q) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--sweep", type=int)
    ap.add_argument("--compare")
    ap.add_argument("--max-iters", type=int, default=OPTIONS["maxIters"])
    ap.add_argument("--tol", type=float, default=OPTIONS["collisionTolerance"], help="collisionTolerance [m]")
    ap.add_argument("--replay", help="native JSON from rai_bidir_native.py: rerun every seed with RAI's stream and compare; "
                    "exit status 1 if any record differs")
    ap.add_argument("--replay-selftest", action="store_true", help="with --replay: also check that two corrupted records fail")
    a = ap.parse_args()
    r = plan(a.seed, maxIters=a.max_iters, collisionTolerance=a.tol)
    s = summary(r)
    print(f"RAI-style bidirectional RRT, options {r['options']}, seed {a.seed} (mulberry32)")
    if a.trace:
        for ev in r["trace"]:
            print("  " + describe(ev))
    print(f"  joined: {s['joined']} after {s['iterations']} iterations ({s['attempts']} extension attempts, {s['queries']} queries incl. the two roots); "
          f"trees {s['tree_sizes']}")
    if r["path"]:
        j = r["join"]
        thr = r["options"]["stepsize"] / r["options"]["subsamples"]
        dj = j["dist_new_to_other"]
        dtxt = f"{dj:.7f}" if abs(dj - thr) < 5e-5 else f"{dj:.4f}"
        print(f"  the {j['tree']} tree added node #{j['new']}, {dtxt} rad from node #{j['other_node']} of the {j['other']} tree (< {thr:g}): joined")
        case, along, offset = join_case(r)
        nq = len(r["trace"][-1]["queries"])
        if case == "coincide":
            what = f"the new node coincides with that node, so the joining segment is the extension edge the attempt queried {nq} times"
        elif case == "on-segment":
            what = (f"the new node is omitted from the returned vertex list and lies on the joining segment; the attempt's {nq} queries "
                    f"cover its first {along:.4f} rad, and RAI did not check the complete connection")
        else:
            what = (f"the new node is omitted from the returned vertex list and lies {offset:.4f} rad off the joining segment; "
                    f"the attempt's queries lie off it, and RAI did not check this connection")
        print(f"  returned path: {s['vertices']} vertices, {s['length_rad']:.4f} rad; joining segment = edge {j['index'] + 1} of {s['vertices'] - 1} "
              f"(edges counted from 1 at the start), {j['segment_length']:.4f} rad; {what}")
        c = check_path(r["path"], 1.0)
        print(f"  course check (1 deg per joint, 5 mm, joining segment included): {'pass' if c['pass'] else 'FAIL'}; "
              f"{c['evaluated']} of {c['scheduled']} scheduled configurations evaluated" + (f"; first failure edge {c['first']['edge'] + 1}, check {c['first']['j']} of {c['first']['n']}, "
              f"clearance {c['first']['clearance_mm']:.3f} mm" if c["first"] else ""))
        f = check_path(r["path"], 0.05)
        print(f"  diagnostic scan (0.05 deg): smallest clearance {f['min_clearance_mm']:.3f} mm on edge {f['min_edge'] + 1}")
    else:
        print(f"  no path: RAI would report feasible = False and ret.evals = {r['iterations']} (maxIters + 1)")
    if a.json:
        json.dump(to_json(r), open(a.json, "w"), indent=1)
    if a.replay:
        native = json.load(open(a.replay))
        same, diff = replay(native)
        bad = replay_selftest(native) if a.replay_selftest else []
        if diff or bad:
            raise SystemExit(1)
    if a.sweep:
        port = [summary(plan(sd)) for sd in range(1, a.sweep + 1)]
        if a.compare:
            compare(port, json.load(open(a.compare)))


if __name__ == "__main__":
    main()
