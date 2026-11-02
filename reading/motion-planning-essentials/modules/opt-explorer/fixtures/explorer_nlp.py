"""The opt-explorer's KOMO program in NumPy (angles in radians, clearance rows in metres), the row check against
the explorer's exported rows, and the parser of RAI's ==nlp== log lines. NumPy only, so that both the container-side
fixture writer (komo_al_rai.py) and the host-side replay check (lectures/04-D2-motion-planning/code/check_rai_nlp.py)
use the same rows. Call `load_problem(path)` first: it reads komo_al_problem.json (written by
`node scripts/check_komo_al.mjs --export-problem`) and sets the arm lengths and the disc.
"""
import json
import pathlib
import re

import numpy as np

OT = {"sos": 1, "ineq": 2, "eq": 3}      # ry.OT's integer values on robotic 0.4.1 (rai-probe.txt)
P = None
L1 = L2 = None
DISC = None
problem_text = None


def load_problem(path):
    """Read komo_al_problem.json; returns the parsed dict and keeps the arm and disc as module globals."""
    global P, L1, L2, DISC, problem_text
    problem_text = pathlib.Path(path).read_text()
    P = json.loads(problem_text)
    L1, L2 = P["arm"]["l1"], P["arm"]["l2"]
    DISC = P["disc"]
    return P


def fd_coeffs(k):
    C, binom = [0] * (k + 1), 1
    for j in range(k + 1):
        C[j] = (1 if (k - j) % 2 == 0 else -1) * binom
        if j < k:
            binom = binom * (k - j) // (j + 1)
    return C


class ExplorerNLP:
    """The explorer's KOMO program in NumPy; angles in radians, clearance rows in metres (the base at the origin)."""

    def __init__(self, case):
        self.T = len(case["x0"]) // 2
        self.k, self.disc = case["k"], case["disc"]
        self.start = np.array(case["start"])
        self.goal = np.array(case["goal"])
        self.dt = 1.0 / self.T
        self.types_str = case["types"]
        self.types = [OT[t] for t in self.types_str]
        self.dimension = 2 * self.T
        self.bounds = np.zeros((2, 0))
        self.x0 = np.array(case["x0"])
        self.calls = 0

    def theta(self, x, t):
        """Knot t's angles: knot 0 is the fixed start (the prefix)."""
        return self.start if t <= 0 else x[2 * (t - 1):2 * (t - 1) + 2]

    def evaluate(self, x):
        self.calls += 1
        T, k, dt, n = self.T, self.k, self.dt, self.dimension
        rows, J = [], []
        C = fd_coeffs(k)
        scale = np.sqrt(dt) / dt ** k
        for t in range(1, T + 1):                     # smoothness: sqrt(dt) D^k theta_t / dt^k with the prefix
            for j in range(2):
                v, row = 0.0, np.zeros(n)
                for i, c in enumerate(C):
                    knot = max(0, t - k + i)
                    v += c * self.theta(x, knot)[j]
                    if knot >= 1:
                        row[2 * (knot - 1) + j] += c
                rows.append(scale * v); J.append(scale * row)
        if "goal" in self.types_str_kinds():
            for j in range(2):
                row = np.zeros(n); row[2 * (T - 1) + j] = 1.0
                rows.append(x[2 * (T - 1) + j] - self.goal[j]); J.append(row)
        if "rest" in self.types_str_kinds():
            for j in range(2):
                row = np.zeros(n); row[2 * (T - 1) + j] = 1.0 / dt
                prev = self.theta(x, T - 1)[j]
                if T >= 2:
                    row[2 * (T - 2) + j] = -1.0 / dt
                rows.append((x[2 * (T - 1) + j] - prev) / dt); J.append(row)
        if self.disc:
            c = np.array([DISC["x"], DISC["y"]]); r = DISC["r"]
            for t in range(1, T + 1):
                th1, th2 = self.theta(x, t)
                base = np.zeros(2)
                elbow = np.array([L1 * np.cos(th1), L1 * np.sin(th1)])
                ee = elbow + np.array([L2 * np.cos(th1 + th2), L2 * np.sin(th1 + th2)])
                for link, (a, b) in enumerate(((base, elbow), (elbow, ee)), start=1):
                    d = b - a; len2 = d @ d
                    tau = 0.0 if len2 < 1e-14 else min(1.0, max(0.0, ((c - a) @ d) / len2))
                    p = a + tau * d
                    dist = np.linalg.norm(c - p)
                    rows.append(r - dist)
                    row = np.zeros(n)
                    if dist > 1e-12:
                        nvec = (c - p) / dist
                        dp1 = np.array([-p[1], p[0]])
                        row[2 * (t - 1)] = nvec @ dp1
                        if link == 2:
                            dp2 = np.array([-(p[1] - elbow[1]), p[0] - elbow[0]])
                            row[2 * (t - 1) + 1] = nvec @ dp2
                    J.append(row)
        return np.array(rows), np.array(J)

    def types_str_kinds(self):
        return set(r["kind"] for r in self._rows)

    def getInitializationSample(self):
        return self.x0.copy()


def check_rows(case, nlp):
    """The NumPy rows and Jacobians against the explorer's at x0 and the probe points."""
    worst_phi = worst_J = 0.0
    for pr in case["probes"]:
        phi, J = nlp.evaluate(np.array(pr["x"]))
        worst_phi = max(worst_phi, float(np.max(np.abs(phi - np.array(pr["phi"])))))
        worst_J = max(worst_J, float(np.max(np.abs(J.reshape(-1) - np.array(pr["J"])))))
    nlp.calls = 0
    return worst_phi, worst_J


NLP_LINE = re.compile(r"==nlp== it:\s*(\d+)\s+evals:\s*(\d+)\s+A\(x\):\s*([-+0-9.e]+)\s+(.*)")


def parse_log(text):
    """RAI's ==nlp== lines: the end of each inner loop (f, g, h, |x-x'|, stop) and each start (mu)."""
    ends, starts, stop, newton = [], [], None, []
    for line in text.splitlines():
        m = NLP_LINE.match(line.strip())
        if m:
            rest = m.group(4)
            if "|x-x'|" in rest:
                g = re.search(r"f:\s*([-+0-9.e]+)\s+g:\s*([-+0-9.e]+)\s+h:\s*([-+0-9.e]+)\s+\|x-x'\|:\s*([-+0-9.e]+)\s+stop:(\w+)(?:\s+\(bad:(\d+)\))?", rest)
                ends.append({"it": int(m.group(1)), "evals": int(m.group(2)), "A": float(m.group(3)), "f": float(g.group(1)), "g": float(g.group(2)),
                             "h": float(g.group(3)), "step": float(g.group(4)), "stop": g.group(5), "bad": int(g.group(6) or 0)})
            elif "mu:" in rest:
                starts.append({"it": int(m.group(1)), "evals": int(m.group(2)), "A": float(m.group(3)), "mu": float(re.search(r"mu:\s*([-+0-9.e]+)", rest).group(1))})
        elif "StoppingCriterion" in line or "numBadSteps" in line or "inner aborted" in line or "squaredPenaltyFixed" in line:
            stop = (stop + " | " if stop else "") + line.strip().replace("==nlp== ", "")
        elif line.strip().startswith("--newton--") and "it:" in line:
            newton.append(line.strip())
    return {"inner_loops": ends, "updates": starts, "stop": stop, "newton_lines": newton}
