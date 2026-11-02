"""Reference numbers for the RRT* rewiring illustration of the sampling demo (index.html?mode=rrtstar).

The illustration inserts ONE new node into a fixed tree the way RRT* does (Karaman and Frazzoli 2011, Algorithm 6),
and shows each candidate in turn. It is not a full RRT* planner. The tree is the demo's one-tree RRT at the moment it
reached the goal (rrt_two_link.rrt, 10 deg extensions, 10 % goal sampling, the course's edge check). The sample is a
later draw of the same random stream: the generator continues past the connection exactly as rrt() would have drawn,
and `sample` names the iteration whose draw is used. The neighborhood is a fixed joint-space radius; RRT* shrinks its
radius as the tree grows, which this illustration does not show.

The cost of a node is the joint-space length of its tree route from the start (the root), the sum of the Euclidean
joint-space lengths of its edges. One insertion:
  1. Sample and steer as RRT does: the nearest node (lowest index on a tie), then a step of at most 10 deg. The edge
     from the nearest node is checked; if it fails, the sample is skipped.
  2. The neighborhood is every node within the radius of the new node, in order of distance (index on a tie).
  3. Parent choice: the nearest node is the default parent. Each other neighbor in turn: the cost through it is its
     cost plus its distance to the new node. A connection that is not cheaper than the best so far is not checked
     ("costlier"). A cheaper one is checked from the neighbor to the new node: it becomes the best parent if the edge
     passes ("better"), and is rejected if it fails ("invalid").
  4. The new node joins the tree under the best parent.
  5. Rewiring: each neighbor in turn. The new node's ancestors are skipped, because making one of them a child of the
     new node would close a cycle ("ancestor"). Otherwise the cost through the new node is the new node's cost plus the
     distance. A connection that is not cheaper than the neighbor's present cost is not checked ("not cheaper"). A
     cheaper one is checked from the new node to the neighbor; if it passes the neighbor's parent becomes the new node
     ("rewired") and the costs of its whole subtree are recomputed from their parents, top down; if it fails the
     neighbor keeps its parent ("invalid").
The edge check is rrt_two_link.check_edge: configurations at most 1 deg apart per joint, each needing 5 mm between the
whole links and the disc, the first failure rejecting the edge.

Run:  .venv/bin/python modules/motion-planning/rrtstar_reference.py                   the default example, every step
      .venv/bin/python modules/motion-planning/rrtstar_reference.py --json out.json   (the browser check reads it)
      .venv/bin/python modules/motion-planning/rrtstar_reference.py --sweep 7 17 --samples 2000   invariants on many insertions
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrt_two_link as R                                                   # the demo's scene and contract

DEG = R.DEG
SEED, SAMPLE, RADIUS_DEG = 17, 141, 20.0


def tree_and_stream(seed):
    """The tree that rrt_two_link.rrt(seed) returns at the connection, and the generator positioned after its last
    draw. The draws are counted from rrt()'s trace (a goal sample uses one draw, a uniform sample three)."""
    trace = []
    r = R.rrt(seed, R.BUDGET, "links", trace)
    if r["outcome"] != "connected":
        raise ValueError(f"seed {seed}: the RRT did not connect")
    rng = R.mulberry32(seed)
    for ev in trace:
        rng()
        if ev["kind"] == "uniform":
            rng(); rng()
    return r["nodes"], r["parents"], r["iterations"], rng


def draw(rng):
    u = rng()
    return R.Q_GOAL if u < R.P_GOAL else (-math.pi + 2 * math.pi * rng(), -math.pi + 2 * math.pi * rng())


def sample_at(seed, sample):
    """The tree at the connection and the sample drawn at iteration `sample` of the continued stream."""
    nodes, parents, it, rng = tree_and_stream(seed)
    if sample <= it:
        raise ValueError(f"seed {seed}: the sample must come after the connection at iteration {it}")
    for _ in range(it + 1, sample):
        draw(rng)
    return nodes, parents, it, draw(rng)


def route_costs(nodes, parents):
    """cost[k] = cost[parent] + dist(parent, k); in rrt()'s trees every parent has a lower index than its child."""
    c = [0.0] * len(nodes)
    for k in range(1, len(nodes)):
        c[k] = c[parents[k]] + R.dist(nodes[parents[k]], nodes[k])
    return c


def edge(a, b):
    ok, n, j, q, part, c, _ = R.check_edge(a, b, "links")
    return {"ok": ok, "n": n, "evaluated": n if ok else j, "j": j, "q": q, "part": part, "c": c}


def children_of(parents):
    ch = {}
    for k, p in enumerate(parents):
        if p >= 0:
            ch.setdefault(p, []).append(k)
    return ch


def insert(nodes, parents, qr, radius):
    """One RRT* insertion of the sample qr into a copy of the tree. Returns the events and the final tree."""
    nodes, parents = list(nodes), list(parents)
    cost = route_costs(nodes, parents)
    goal = len(nodes) - 1                                                  # rrt() appends the goal last
    goal_before = cost[goal]
    near = min(range(len(nodes)), key=lambda k: R.dist(qr, nodes[k]))     # lowest index on a tie
    qn = nodes[near]
    d = R.dist(qr, qn)
    x = qr if d <= R.STEP else (qn[0] + R.STEP * (qr[0] - qn[0]) / d, qn[1] + R.STEP * (qr[1] - qn[1]) / d)
    e0 = edge(qn, x)
    # the nearest node lies within the step of the new node, so it belongs to any neighborhood at least that wide; it is
    # included explicitly because a steered node can lie a rounding error beyond exactly one step from it
    X = sorted((k for k in range(len(nodes)) if R.dist(nodes[k], x) <= radius or (k == near and radius >= R.STEP)),
               key=lambda k: R.dist(nodes[k], x))
    events = [{"kind": "sample", "sample": qr, "nearest": near, "x": x, "step": R.dist(qn, x), "edge": e0, "near": X,
               "radius": radius}]
    if not e0["ok"]:
        events.append({"kind": "skipped"})
        return {"events": events, "nodes": nodes, "parents": parents, "cost": cost, "goal": goal, "goal_before": goal_before}
    best, cbest = near, cost[near] + R.dist(qn, x)
    for k in X:
        if k == near:
            continue
        via = cost[k] + R.dist(nodes[k], x)
        ev = {"kind": "parent", "k": k, "cost": cost[k], "dist": R.dist(nodes[k], x), "via": via, "best": best, "best_cost": cbest}
        if not via < cbest:
            ev["result"] = "costlier"
        else:
            ev["edge"] = edge(nodes[k], x)
            if ev["edge"]["ok"]:
                ev["result"] = "better"
                best, cbest = k, via
            else:
                ev["result"] = "invalid"
        events.append(ev)
    nodes.append(x); parents.append(best); cost.append(cbest)
    new = len(nodes) - 1
    events.append({"kind": "connect", "new": new, "parent": best, "cost": cbest, "saving": cost[near] + R.dist(qn, x) - cbest})
    anc, a = [], best
    while a != -1:
        anc.append(a)
        a = parents[a]
    for k in X:
        ev = {"kind": "rewire", "k": k, "cost": cost[k], "dist": R.dist(x, nodes[k]), "via": cost[new] + R.dist(x, nodes[k])}
        if k in anc:
            ev["result"] = "ancestor"
        elif not ev["via"] < cost[k]:
            ev["result"] = "not cheaper"
        else:
            ev["edge"] = edge(x, nodes[k])
            if not ev["edge"]["ok"]:
                ev["result"] = "invalid"
            else:
                ev["result"] = "rewired"
                ev["old_parent"] = parents[k]
                parents[k] = new
                before = list(cost)
                ch, order = children_of(parents), [k]
                for i in order:                                            # breadth first: parents before children
                    if i != k:
                        cost[i] = cost[parents[i]] + R.dist(nodes[parents[i]], nodes[i])
                    else:
                        cost[i] = ev["via"]
                    order.extend(ch.get(i, []))
                ev["descendants"] = [{"k": i, "before": before[i], "after": cost[i]} for i in order[1:]]
                ev["after"] = cost[k]
        events.append(ev)
    route, a = [], goal
    while a != -1:
        route.append(a)
        a = parents[a]
    events.append({"kind": "done", "goal": goal, "goal_before": goal_before, "goal_after": cost[goal], "route": route[::-1]})
    return {"events": events, "nodes": nodes, "parents": parents, "cost": cost, "goal": goal, "goal_before": goal_before}


def invariants(res, nodes0, parents0):
    """Checks on the result of insert(). Safety: a tree (every node reaches the root without a cycle), stored costs
    equal to costs recomputed from scratch, every new or changed edge passing the edge check, no cost rising. Decisions,
    each recomputed here independently of insert()'s order of tests: the new node's cost is the cheapest over the
    nearest node and every neighbor whose edge to the new node passes; no neighbor other than an ancestor of the new node
    is left with a cheaper route through the new node over a passing edge; and every node that kept its parent saved
    exactly what its nearest re-parented ancestor saved. Returns a list of problems (empty when all hold)."""
    N, P, c = res["nodes"], res["parents"], res["cost"]
    bad = []
    for k in range(1, len(N)):
        seen, a = set(), k
        while a != 0:
            if a < 0 or a in seen:
                bad.append(f"node {k}: no route to the root")
                break
            seen.add(a)
            a = P[a]
    if bad:
        return bad                                                         # costs along a cycle are undefined
    scratch = [None] * len(N)
    def cost_of(k):
        chain = []
        while scratch[k] is None and k != 0:
            chain.append(k)
            k = P[k]
        base = 0.0 if scratch[k] is None else scratch[k]
        scratch[k] = base
        for i in reversed(chain):
            base = base + R.dist(N[P[i]], N[i])
            scratch[i] = base
        return scratch
    for k in range(len(N)):
        cost_of(k)
    for k in range(len(N)):
        if abs(scratch[k] - c[k]) > 1e-12:
            bad.append(f"node {k}: stored cost {c[k]} differs from the recomputed {scratch[k]}")
    c0 = route_costs(nodes0, parents0)
    for k in range(len(nodes0)):
        if c[k] > c0[k] + 1e-12:
            bad.append(f"node {k}: cost rose from {c0[k]} to {c[k]}")
        if P[k] != parents0[k] and not R.check_edge(N[P[k]], N[k])[0]:
            bad.append(f"node {k}: its new edge fails the check")
    s0 = res["events"][0]
    if len(N) == len(nodes0):
        if s0["edge"]["ok"]:
            bad.append("the sample's first edge passes, but no node was added")
        return bad
    new, x = len(N) - 1, N[-1]
    if not R.check_edge(N[P[new]], x)[0]:
        bad.append("the new node's edge fails the check")
    cands = [s0["nearest"]] + [k for k in s0["near"] if k != s0["nearest"] and R.check_edge(nodes0[k], x)[0]]
    best = min(c0[k] + R.dist(nodes0[k], x) for k in cands)
    if abs(c[new] - best) > 1e-12:
        bad.append(f"the new node's cost {c[new]} is not the cheapest over its valid candidates, {best}")
    anc, a = set(), P[new]
    while a != -1:
        anc.add(a)
        a = P[a]
    for k in s0["near"]:
        if k not in anc and c[new] + R.dist(x, N[k]) < c[k] - 1e-12 and R.check_edge(x, N[k])[0]:
            bad.append(f"neighbor {k} is left at {c[k]} although a passing edge through the new node costs {c[new] + R.dist(x, N[k])}")
    moved = {k for k in range(len(nodes0)) if P[k] != parents0[k]}
    for k in range(1, len(nodes0)):
        if k in moved:
            continue
        a = P[k]
        while a not in moved and a > 0:
            a = P[a]
        expected = c0[a] - c[a] if a in moved else 0.0
        if abs((c0[k] - c[k]) - expected) > 1e-12:
            bad.append(f"node {k} saved {c0[k] - c[k]}, but its nearest re-parented ancestor saved {expected}")
    return bad


def margins(events):
    """The cost comparisons of an insertion, as |cost through the candidate - the cost it is compared with| [rad]."""
    out = []
    for e in events:
        if e["kind"] == "parent":
            out.append(abs(e["via"] - e["best_cost"]))
        elif e["kind"] == "rewire" and e["result"] != "ancestor":
            out.append(abs(e["via"] - e["cost"]))
    return out


def search(seeds, samples, radius_deg, max_near=5, min_margin_deg=0.3):
    """The later samples whose insertion shows every operation: a cheaper parent than the nearest node, a rewiring with at
    least one descendant, a cheaper connection whose edge fails, and a skipped ancestor; with at most max_near neighbors,
    every cost comparison at least min_margin_deg apart, and a cheaper route to the goal. Prints and returns them."""
    found = []
    for seed in seeds:
        nodes, parents, it, rng = tree_and_stream(seed)
        for m in range(it + 1, it + 1 + samples):
            res = insert(nodes, parents, draw(rng), radius_deg * DEG)
            ev = res["events"]
            kinds = {(e["kind"], e.get("result")) for e in ev}
            if ("skipped", None) in kinds or len(ev[0]["near"]) > max_near:
                continue
            conn = next(e for e in ev if e["kind"] == "connect")
            done = ev[-1]
            if conn["parent"] == ev[0]["nearest"] or ("rewire", "ancestor") not in kinds:
                continue
            if not (("parent", "invalid") in kinds or ("rewire", "invalid") in kinds):
                continue
            if not any(e.get("result") == "rewired" and e["descendants"] for e in ev):
                continue
            mg = margins(ev)
            if not mg or min(mg) < min_margin_deg * DEG or not done["goal_after"] < done["goal_before"]:
                continue
            found.append({"seed": seed, "sample": m, "neighbors": len(ev[0]["near"]), "smallest_margin_deg": min(mg) / DEG,
                          "goal_before_deg": done["goal_before"] / DEG, "goal_after_deg": done["goal_after"] / DEG})
    print(f"search: seeds {seeds}, {samples} later samples each, radius {radius_deg:g} deg, at most {max_near} neighbors, "
          f"margins at least {min_margin_deg:g} deg: {len(found)} found")
    for f in found[:40]:
        print(f"  seed {f['seed']}, sample {f['sample']}: {f['neighbors']} neighbors, smallest margin {f['smallest_margin_deg']:.2f} deg, "
              f"goal {f['goal_before_deg']:.2f} -> {f['goal_after_deg']:.2f} deg")
    return found


def run(seed=SEED, sample=SAMPLE, radius_deg=RADIUS_DEG):
    nodes, parents, it, qr = sample_at(seed, sample)
    res = insert(nodes, parents, qr, radius_deg * DEG)
    return {"seed": seed, "sample": sample, "radius_deg": radius_deg, "connected_at": it, "tree": {"nodes": nodes, "parents": parents},
            "cost0": route_costs(nodes, parents), "events": res["events"], "final": {"nodes": res["nodes"], "parents": res["parents"], "cost": res["cost"]},
            "invariants": invariants(res, nodes, parents)}


def words(ev):
    d = lambda v: f"{v / DEG:.2f} deg"
    k = ev["kind"]
    if k == "sample":
        e = ev["edge"]
        return (f"sample ({ev['sample'][0] / DEG:.2f}, {ev['sample'][1] / DEG:.2f}) deg; nearest #{ev['nearest']}; new node {d(ev['step'])} from it; "
                f"edge {'passes' if e['ok'] else 'FAILS'} ({e['evaluated']} of {e['n']}); neighbors within {d(ev['radius'])}: {['#%d' % i for i in ev['near']]}")
    if k == "parent":
        s = f"parent candidate #{ev['k']}: cost {d(ev['cost'])} + {d(ev['dist'])} = {d(ev['via'])} against best #{ev['best']} {d(ev['best_cost'])}: {ev['result']}"
        return s + (f" (edge {ev['edge']['evaluated']} of {ev['edge']['n']} checks)" if "edge" in ev else "")
    if k == "connect":
        return f"connect: new node #{ev['new']} under #{ev['parent']}, cost {d(ev['cost'])} ({d(ev['saving'])} below the nearest node's)"
    if k == "rewire":
        s = f"rewire candidate #{ev['k']}: cost {d(ev['cost'])}, through the new node {d(ev['via'])}: {ev['result']}"
        if "edge" in ev:
            s += f" (edge {ev['edge']['evaluated']} of {ev['edge']['n']} checks)"
        if ev["result"] == "rewired":
            s += "; descendants " + ", ".join(f"#{x['k']} {x['before'] / DEG:.2f} -> {x['after'] / DEG:.2f}" for x in ev["descendants"])
        return s
    if k == "done":
        return f"goal #{ev['goal']}: {d(ev['goal_before'])} -> {d(ev['goal_after'])}; route {ev['route']}"
    return "the sample's edge from the nearest node fails, so RRT* skips it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument("--radius", type=float, default=RADIUS_DEG, help="neighborhood radius [deg]")
    ap.add_argument("--json")
    ap.add_argument("--sweep", type=int, nargs="*", help="seeds: insert every continued sample and check the invariants")
    ap.add_argument("--samples", type=int, default=1000, help="with --sweep or --search: continued samples per seed")
    ap.add_argument("--search", type=int, nargs="+", help="seeds: list the later samples whose insertion shows every operation")
    a = ap.parse_args()
    if a.search:
        search(a.search, a.samples, a.radius)
        return
    if a.sweep is not None:
        if not a.sweep:
            ap.error("--sweep needs at least one seed")
        total, rewires, problems = 0, 0, []
        for seed in a.sweep:
            nodes, parents, it, rng = tree_and_stream(seed)
            for m in range(it + 1, it + 1 + a.samples):
                res = insert(nodes, parents, draw(rng), a.radius * DEG)
                total += 1
                rewires += sum(e.get("result") == "rewired" for e in res["events"])
                problems += [(seed, m, p) for p in invariants(res, nodes, parents)]
        print(f"sweep: {total} insertions (seeds {a.sweep}, {a.samples} continued samples each, radius {a.radius:g} deg), {rewires} rewirings; "
              f"invariant violations: {len(problems)}" + (f"; first {problems[:3]}" if problems else ""))
        if not rewires:
            print("  no insertion rewired a node, so the sweep does not exercise rewiring")
        raise SystemExit(1 if problems or not rewires else 0)
    r = run(a.seed, a.sample, a.radius)
    print(f"RRT* rewiring illustration: the one-tree RRT at seed {a.seed} (connected at iteration {r['connected_at']}, {len(r['tree']['nodes'])} nodes), "
          f"the sample of iteration {a.sample}, radius {a.radius:g} deg")
    for ev in r["events"]:
        print("  " + words(ev))
    print("  invariants: " + ("all hold" if not r["invariants"] else "; ".join(r["invariants"])))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    raise SystemExit(1 if r["invariants"] else 0)


if __name__ == "__main__":
    main()
