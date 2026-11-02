"""A NumPy port of RAI's NLP_Solver for the course's KOMO examples (rai commit ccc36b55 = robotic 0.4.1).

Every KOMO example in Deck 06, Chapter 03 and the motion-planning module is solved with this loop, so that the course's
numbers are those of `ry.NLP_Solver` with the same options. The port follows `src/Optim/constrained.cpp`
(ConstrainedSolver), `m_Newton.cpp` (OptNewton) and `lagrangian.cpp` (LagrangianProblem) line by line, through the
TypeScript twin `modules/opt-explorer/src/komoAugLag.ts` (2026-10-30), which real RAI runs replay to 1e-8 rad:

  * the augmented Lagrangian  L = sum phi_sos^2 + sum_{active} mu g^2 + sum_{lambda > 0} lambda g + sum (mu h^2 + kappa h)
    with the gate active = (g > 0 or lambda > 0);
  * the Gauss-Newton Newton step (H + damping I) Delta = -grad L with the step cap stepMax, the lazy stop
    |Delta| < 0.1 stopTolerance, the Wolfe backtracking with stepInc / stepDec, the tiny-step counters and RAI's stop order;
  * the outer loop: tripled tolerances in the first outer iteration, the stop test |x - x'| < stopTolerance from the
    second loop on, the update lambda <- max(0, lambda + 2 mu g), kappa <- kappa + 2 mu h with the current mu, then
    mu <- min(mu muInc, muMax);
  * RAI's buffer of the last evaluated point (accepted or rejected), from which the update recomputes L, its gradient
    and Hessian without a new evaluation (including the quirk that an inner loop ending on a rejected trial leaves the
    gradient and Hessian of the rejected point beside the old x);
  * `SquaredPenalty`: the same loop with the multiplier lines of aulaUpdate skipped (lambdaStepsize = -1), so lambda
    stays zero and the gate is g > 0; `AugmentedLag` is RAI's default method.

The supported subset, stated plainly. The port implements AugmentedLag and SquaredPenalty on problems without bounds,
with RAI's default values of `stepMin`, `interiorPadding`, `finiteDifference`, `muLBInit` and `muLBDec` (the log-barrier
and bounded-Newton paths are not ported): `setOptions` raises NotImplementedError on any other value, and `setProblem`
raises on a problem with bounds. RAI's Newton hard-codes the line-search limit 10 and the tiny-step limit 4 at this
commit (`stopLineSteps` and `stopTinySteps` are options it does not read), and so does the port. The port never adds
initialization noise: RAI's `solve()` and `solve(0)` both draw N(0, 0.01 rad) noise through KOMO's
`getInitializationSample` unless `setInitialization` was called, so here `solve()` raises unless `setInitialization`
was called, and `solve(1)` (resample) raises always. A run that should start from a recorded noisy sample starts from
that recorded vector through `setInitialization`.

Arithmetic: the decision scalars (L, the Wolfe slope, max|Delta|, the descent test) are accumulated in plain
sequential loops in the TypeScript row order; the gradient and the Gauss-Newton Hessian are NumPy products; the Newton
system is solved with `numpy.linalg.cholesky` and two triangular solves (RAI calls LAPACK's dposv), with the gradient
direction of length stepMax as the fallback on a failed factorization or an ascent direction, as in RAI. Differences
to RAI are at rounding level; `check_rai_nlp.py` classifies a decision that flips within a relative margin of 1e-10
as a qualified warning and compares the end states.

API (the names of `ry.NLP_Solver` where they exist):
  NLP(dimension, types, evaluate, x0)          the problem: `evaluate(x) -> (phi, J)`, types in {'f','sos','ineq','eq'} or ry.OT ints
  NLP.from_blocks(n, sos=, eq=, ineq=, x0=)    stack row functions `f(x) -> (values, Jacobian)` in the order sos, eq, ineq
  NLP_Solver().setProblem(nlp).setOptions(**o).setInitialization(x0).solve(0) -> SolverReturn
  solver.getOptions().dict(), solver.getTrace_x(), solver.getTrace_errs(), solver.dual
  port-only: solver.ministep(), solver.run_inner_loop(), solver.log (RAI's ==nlp== rows), solver.state (the solver's
  state: x, buffered point, trials, steps), solver.buffered_x
  parse_case(case_id), options_for(case), solve_case(nlp, case, x_start) -> (z, lam, kap, info)   the case-ID interface

`python rai_nlp.py` replays the tiny problem of modules/opt-explorer/fixtures/probe_pynlp.py against RAI's recorded
output (planning/opt-explorer-komo-al-2026-10-30/rai-probe.txt: 12 evaluations, x = [0.74998506, 0.24990452],
dual [0, 0, 2.02865286, -1.52089998], the ==nlp== rows) and exits 1 on a mismatch.
"""
import math
import re
import sys
import time

import numpy as np

UPSTREAM = {
    "rai_commit": "ccc36b55",
    "robotic": "0.4.1",
    "compiled": "compile time: Jul 16 2026 07:44:51",   # ry.compiled() of the rai-mac image the fixtures were written on
    "files": ["src/Optim/constrained.cpp", "src/Optim/m_Newton.cpp", "src/Optim/lagrangian.cpp", "src/Optim/NLP.cpp", "src/Optim/options.h"],
    "note": "byte-identical on master on 2026-10-30 (GitHub compare API); ported through modules/opt-explorer/src/komoAugLag.ts",
}

OT = {"f": 0, "sos": 1, "ineq": 2, "eq": 3}
OT_NAME = {v: k for k, v in OT.items()}

# every option of rai::OptOptions (options.h) with its default, in the order of the struct
RAI_DEFAULTS = dict(
    verbose=1, stopTolerance=1e-2, stopFTolerance=1e-4, stopGTolerance=-1.0, stopEvals=1000, stopInners=1000, stopOuters=1000,
    stopLineSteps=10, stopTinySteps=4, stepInit=1.0, stepMin=-1.0, stepMax=0.2, stepInc=1.5, stepDec=0.5, damping=1.0, wolfe=0.01,
    muInit=1.0, muInc=5.0, muMax=1e4, muLBInit=0.1, muLBDec=0.2, lambdaMax=-1.0, interiorPadding=1e-2, boundedNewton=True,
    method="AugmentedLag", finiteDifference=-1.0,
)
METHODS = ("AugmentedLag", "SquaredPenalty")
# options whose non-default value selects a path the port does not implement
UNSUPPORTED_NONDEFAULT = ("stepMin", "interiorPadding", "finiteDifference", "muLBInit", "muLBDec")

STOP_RANK = {"None": 0, "DeltaConverge": 1, "TinyFSteps": 2, "TinyXSteps": 3, "CritEvals": 4, "LineSearchSteps": 5}


class Options:
    """RAI's option table; `dict()` as `ry.NLP_Solver.getOptions().dict()`."""

    def __init__(self, values):
        self._v = dict(values)

    def dict(self):
        return dict(self._v)

    def __getattr__(self, name):
        try:
            return self.__dict__["_v"][name]
        except KeyError:
            raise AttributeError(name) from None


def _types_to_ints(types):
    out = []
    for t in types:
        if isinstance(t, str):
            out.append(OT[t])
        else:
            out.append(int(t))
    return out


class NLP:
    """A problem in RAI's NLP form: `evaluate(x) -> (phi, J)` with J dense (m x n), one feature type per row."""

    def __init__(self, dimension, types, evaluate, x0=None, bounds=None, rows=None):
        self.dimension = int(dimension)
        self.types = _types_to_ints(types)
        self.types_str = [OT_NAME[t] for t in self.types]
        self._evaluate = evaluate
        self.x0 = None if x0 is None else np.array(x0, dtype=float)
        self.bounds = np.zeros((2, 0)) if bounds is None else np.asarray(bounds, dtype=float)
        self.rows = rows
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        phi, J = self._evaluate(np.asarray(x, dtype=float))
        phi = np.asarray(phi, dtype=float)
        J = np.asarray(J, dtype=float)
        if phi.shape != (len(self.types),) or J.shape != (len(self.types), self.dimension):
            raise ValueError(f"evaluate returned phi {phi.shape} and J {J.shape}; expected ({len(self.types)},) and ({len(self.types)}, {self.dimension})")
        return phi, J

    def getInitializationSample(self):
        if self.x0 is None:
            raise NotImplementedError("this NLP has no x0; the port never draws a random sample")
        return self.x0.copy()

    @classmethod
    def from_blocks(cls, n, sos=None, eq=None, ineq=None, x0=None):
        """Stack row functions in the order sos, eq, ineq. Each block is `f(x) -> (values, Jacobian)`; its row count is
        read from one evaluation at x0 (or zeros)."""
        probe = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
        blocks = [(kind, fn) for kind, fn in (("sos", sos), ("eq", eq), ("ineq", ineq)) if fn is not None]
        counts = [len(fn(probe)[0]) for _, fn in blocks]
        types = []
        for (kind, _), c in zip(blocks, counts):
            types += [kind] * c

        def evaluate(x):
            parts = [fn(x) for _, fn in blocks]
            return np.concatenate([p[0] for p in parts]), np.vstack([p[1] for p in parts])

        nlp = cls(n, types, evaluate, x0=x0)
        nlp.block_slices = {}
        start = 0
        for (kind, _), c in zip(blocks, counts):
            nlp.block_slices[kind] = slice(start, start + c)
            start += c
        return nlp

    def split_dual(self, dual):
        """RAI's per-row dual as (lam on the ineq rows, kap on the eq rows), in block order."""
        dual = np.asarray(dual, dtype=float)
        lam = dual[[i for i, t in enumerate(self.types) if t == OT["ineq"]]]
        kap = dual[[i for i, t in enumerate(self.types) if t == OT["eq"]]]
        return lam, kap

    def join_dual(self, lam, kap):
        dual = np.zeros(len(self.types))
        dual[[i for i, t in enumerate(self.types) if t == OT["ineq"]]] = lam
        dual[[i for i, t in enumerate(self.types) if t == OT["eq"]]] = kap
        return dual


def summarize_errors(types, phi):
    """NLP::summarizeErrors: [f, sos, ineq, eq] = (sum f, sum phi^2, sum max(g, 0), sum |h|) in RAI's column order."""
    e = [0.0, 0.0, 0.0, 0.0]
    for i, v in enumerate(phi):
        t = types[i]
        if t == 0:
            e[0] += v
        elif t == 1:
            e[1] += v * v
        elif t == 2:
            if v > 0:
                e[2] += v
        else:
            e[3] += abs(v)
    return e


def rai_feasible(errs):
    """SolverReturn's flag: ineq < 0.1 and eq < 0.1 (never a tolerance test)."""
    return errs[2] < 0.1 and errs[3] < 0.1


class SolverReturn:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return ("{ time: %g, evals: %d, done: %d, feasible: %d, eq: %g, ineq: %g, sos: %g, f: %g, x-dim: [%d], stop: %s, outer_iters: %d }"
                % (self.time, self.evals, self.done, self.feasible, self.eq, self.ineq, self.sos, self.f, len(self.x), self.stop, self.outer_iters))


def _lagrangian_at(types, phi, J, mu, lam):
    """LagrangianProblem::eval_scalar on the buffered rows: L, grad L and the Gauss-Newton Hessian."""
    m = len(phi)
    L = 0.0
    coeff = np.zeros(m)
    w = np.zeros(m)
    for i in range(m):
        v = phi[i]
        t = types[i]
        if t == 1:
            L += v * v
            coeff[i] = 2.0 * v
            w[i] = 2.0
        elif t == 2:
            lm = lam[i]
            if v > 0.0 or lm > 0.0:
                L += mu * v * v
                coeff[i] += 2.0 * mu * v
                w[i] += 2.0 * mu
            if lm > 0.0:
                L += lm * v
                coeff[i] += lm
        elif t == 3:
            lm = lam[i]
            L += mu * v * v + lm * v
            coeff[i] = 2.0 * mu * v + lm
            w[i] = 2.0 * mu
        else:
            L += v
            coeff[i] = 1.0
    g = J.T @ coeff
    H = (J * w[:, None]).T @ J
    return L, g, H


def _abs_max(v):
    m = 0.0
    for a in v:
        a = abs(a)
        if a > m:
            m = a
    return m


class _State:
    """ConstrainedSolver + OptNewton + LagrangianProblem, as komoAugLag.ts's SolverState."""

    def __init__(self, nlp, x0, opts, dual=None):
        n = nlp.dimension
        self.nlp = nlp
        self.opts = dict(opts)
        self.unconstrained = all(t == 1 for t in nlp.types)
        self.method = "Newton" if self.unconstrained else opts["method"]
        self.tol = dict(stopTolerance=opts["stopTolerance"], stopFTolerance=opts["stopFTolerance"], stopGTolerance=opts["stopGTolerance"])
        if self.method != "Newton":
            self.tol["stopTolerance"] = 3.0 * opts["stopTolerance"]
            self.tol["stopFTolerance"] = 3.0 * opts["stopFTolerance"]
            self.tol["stopGTolerance"] = 3.0 * opts["stopGTolerance"]
            if self.tol["stopFTolerance"] <= 0.0:
                self.tol["stopFTolerance"] = 1e-3
        self.mu = float(opts["muInit"])
        self.lam = np.zeros(len(nlp.types)) if dual is None else np.array(dual, dtype=float)
        self.buf = None                      # (x, phi, J) of the last evaluated point
        self.x = np.array(x0, dtype=float)
        self.fx = float("nan")
        self.gx = np.zeros(n)
        self.Hx = np.zeros((n, n))
        self.alpha = float(opts["stepInit"])
        self.beta = float(opts["damping"])
        self.evals = 0
        self.inner_iters = 0
        self.num_tiny_f = 0
        self.num_tiny_x = 0
        self.stop_criterion = "None"
        self.last_line_search_steps = 0
        self.last_accepted = None
        self.last_delta_max = float("nan")
        self.trials = []
        self.steps = []
        self.outer_iters = 0
        self.num_bad_steps = 0
        self.x_before = self.x.copy()
        self.done = False
        self.reason = ""
        self.log = []
        self.trace_x = []
        self.trace_errs = []
        self.verbose = int(opts.get("verbose", 0))

    # LagrangianProblem through RAI's buffer: a new NLP evaluation only when x differs from the buffered point
    def eval_L(self, x):
        if self.buf is None or self.buf[0].shape != x.shape or not np.array_equal(self.buf[0], x):
            phi, J = self.nlp.evaluate(x)
            self.buf = (x.copy(), phi, J)
            self.trace_x.append(x.copy())
            self.trace_errs.append(summarize_errors(self.nlp.types, phi))
        return _lagrangian_at(self.nlp.types, self.buf[1], self.buf[2], self.mu, self.lam)

    def newton_step(self):
        """OptNewton::step (m_Newton.cpp) without bounds; returns the stop criterion."""
        n = self.nlp.dimension
        o, tol = self.opts, self.tol
        if not self.evals:
            L, g, H = self.eval_L(self.x)
            self.fx, self.gx, self.Hx = L, g, H
            self.evals += 1
            if self.fx != self.fx:
                raise FloatingPointError("rai_nlp: NaN at the initial point")
            if self.verbose > 1:
                print(f"--newton-- initial point f(x):{self.fx:g} alpha:{self.alpha:g} beta:{self.beta:g}")
        self.inner_iters += 1
        if self.fx != self.fx:
            raise FloatingPointError("rai_nlp: Newton step from a NaN value")
        R = self.Hx.copy()
        if self.beta:
            R[np.diag_indices(n)] += self.beta
        rhs = -self.gx
        inversion_failed = False
        try:
            Lc = np.linalg.cholesky(R)
            y = np.linalg.solve(Lc, rhs)
            Delta = np.linalg.solve(Lc.T, y)
            if not np.all(np.isfinite(Delta)):
                inversion_failed = True
        except np.linalg.LinAlgError:
            inversion_failed = True
            Delta = np.zeros(n)
        if not inversion_failed:
            dg = 0.0
            for i in range(n):
                dg += Delta[i] * self.gx[i]
            if dg > 0.0:
                inversion_failed = True
        if inversion_failed:
            length = math.sqrt(float(self.gx @ self.gx))
            Delta = self.gx * (-o["stepMax"] / length)
        max_delta = _abs_max(Delta)
        if o["stepMax"] > 0.0 and max_delta > o["stepMax"]:
            Delta = Delta * (o["stepMax"] / max_delta)
            max_delta = o["stepMax"]
        self.last_delta_max = max_delta
        self.last_line_search_steps = 0
        self.last_accepted = None
        evals_before = self.evals
        if _abs_max(Delta) < 1e-1 * tol["stopTolerance"]:
            self.steps.append(dict(evalsBefore=evals_before, deltaMax=max_delta, lazy=True, stop="DeltaConverge", tol=tol["stopTolerance"]))
            if self.verbose > 1:
                print(f"--newton-- it:{self.inner_iters:4d}  |Delta|:{_abs_max(Delta):11g} \t -- absMax(Delta)<1e-1*o.stopTolerance -- NO UPDATE")
            self.stop_criterion = "DeltaConverge"
            return self.stop_criterion
        line_search_steps = 0
        while True:
            if self.alpha > 1.0:
                self.alpha = 1.0
            y = self.x + self.alpha * Delta
            Ly, gy, Hy = self.eval_L(y)
            self.evals += 1
            fy = Ly
            slope = 0.0
            for i in range(n):
                slope += (y[i] - self.x[i]) * self.gx[i]
            bound = self.fx + o["wolfe"] * slope
            wolfe = fy <= bound
            accepted = (fy == fy) and wolfe
            self.trials.append(dict(evalIndex=self.evals - 1, fx=self.fx, fy=fy, bound=bound, accepted=accepted))
            if self.verbose > 1:
                print(f"--newton-- it:{self.inner_iters:4d}  |Delta|:{max_delta:11g}  alpha:{self.alpha:11g}  evals:{self.evals:4d}  f(y):{fy:11g}  {'ACCEPT' if accepted else 'reject'}")
            if accepted:
                if tol["stopFTolerance"] > 0.0 and self.fx - fy < tol["stopFTolerance"]:
                    self.num_tiny_f += 1
                else:
                    self.num_tiny_f = 0
                dmax = 0.0
                for i in range(n):
                    dmax = max(dmax, abs(y[i] - self.x[i]))
                if dmax < 1e-2 * tol["stopTolerance"]:
                    self.num_tiny_x += 1
                else:
                    self.num_tiny_x = 0
                self.x = y.copy()
                self.fx, self.gx, self.Hx = fy, gy, Hy
                self.alpha *= o["stepInc"]
                self.last_accepted = True
                break
            self.last_accepted = False
            if self.evals > o["stopEvals"]:
                self.num_tiny_x += 1
                break
            if line_search_steps > 10:
                self.num_tiny_x += 1
                break
            self.alpha *= o["stepDec"]
            line_search_steps += 1
        self.last_line_search_steps = line_search_steps
        crit = "None"
        if _abs_max(Delta) < tol["stopTolerance"]:
            crit = "DeltaConverge"
        elif self.num_tiny_f > 4:
            self.num_tiny_f = 0
            crit = "TinyFSteps"
        elif self.num_tiny_x > 4:
            self.num_tiny_x = 0
            crit = "TinyXSteps"
        elif self.evals >= o["stopEvals"]:
            crit = "CritEvals"
        elif self.inner_iters >= o["stopInners"]:
            crit = "CritEvals"
        elif line_search_steps > 10:
            crit = "LineSearchSteps"
        self.steps.append(dict(evalsBefore=evals_before, deltaMax=max_delta, lazy=False, stop=crit, tol=tol["stopTolerance"]))
        if self.verbose > 1 and crit != "None":
            print(f"--newton-- stopping: '{crit}'")
        self.stop_criterion = crit
        return crit

    def aula_update(self):
        """LagrangianProblem::aulaUpdate at the buffered point: the multipliers with the current mu (AugmentedLag only),
        then mu, the clip, and L with its derivatives at the buffered point without a new evaluation."""
        xb, phi, _ = self.buf
        types = self.nlp.types
        if self.method == "AugmentedLag":
            for i in range(len(self.lam)):
                if types[i] == 3:
                    self.lam[i] += 2.0 * self.mu * phi[i]
                elif types[i] == 2:
                    self.lam[i] += 2.0 * self.mu * phi[i]
                    if self.lam[i] < 0.0:
                        self.lam[i] = 0.0
        if self.opts["muInc"] > 0.0:
            self.mu *= self.opts["muInc"]
            if self.mu > self.opts["muMax"]:
                self.mu = self.opts["muMax"]
        if self.opts["lambdaMax"] > 0.0:
            self.lam = np.clip(self.lam, -self.opts["lambdaMax"], self.opts["lambdaMax"])
        self.fx, self.gx, self.Hx = self.eval_L(xb)

    def multiplier_summary(self):
        lam_max, kap_abs_max = 0.0, 0.0
        for i, t in enumerate(self.nlp.types):
            if t == 2:
                lam_max = max(lam_max, self.lam[i])
            elif t == 3:
                kap_abs_max = max(kap_abs_max, abs(self.lam[i]))
        return lam_max, kap_abs_max

    def count_active(self, phi):
        return sum(1 for i, t in enumerate(self.nlp.types) if t == 2 and (phi[i] > 0.0 or self.lam[i] > 0.0))

    def ministep(self):
        """ConstrainedSolver::ministep: one Newton step; when the inner loop has finished, the stop tests and the update."""
        if self.done:
            return True
        self.newton_step()
        if self.stop_criterion == "None":
            return False
        xb, phi, _ = self.buf
        err = summarize_errors(self.nlp.types, phi)
        step = 0.0
        for i in range(len(self.x)):
            step = max(step, abs(self.x_before[i] - self.x[i]))
        if STOP_RANK[self.stop_criterion] > STOP_RANK["DeltaConverge"]:
            self.num_bad_steps += 1
        else:
            self.num_bad_steps = 0
        o = self.opts
        lam_max, kap_abs_max = self.multiplier_summary()
        row = dict(it=self.outer_iters, evals=self.evals, A=self.fx, f=err[1], g=err[2], h=err[3], step=step, stop=self.stop_criterion,
                   badSteps=self.num_bad_steps, muUsed=self.mu, muNext=self.mu, lambdaMax=lam_max, kappaAbsMax=kap_abs_max,
                   active=self.count_active(phi), updated=False)
        if self.verbose > 0:
            print(f"==nlp== it:{self.outer_iters:4d}  evals:{self.evals:4d}  A(x):{self.fx:11g}  f:{err[1]:11g}  g:{err[2]:11g}  h:{err[3]:11g}  |x-x'|:{step:11g} \tstop:{self.stop_criterion}")

        def stop(reason):
            self.done = True
            self.reason = reason
            row["reason"] = reason
            self.log.append(row)
            if self.verbose > 0:
                print(f"==nlp== StoppingCriterion {reason}")
            return True

        if self.method == "Newton":
            return stop("no constraints: one Newton loop (RAI runs plain Newton once)")
        if self.outer_iters >= 1 and step < o["stopTolerance"]:
            if o["stopGTolerance"] < 0.0 or err[2] + err[3] < o["stopGTolerance"]:
                return stop(f"Delta<{o['stopTolerance']:g}")
            if self.num_bad_steps > 6:
                return stop(f"Delta<{o['stopTolerance']:g} with the violation {err[2] + err[3]:.2e} above stopGTolerance, but numBadSteps > 6")
        if o["stopEvals"] > 0 and self.evals >= o["stopEvals"]:
            return stop("MAX EVALS")
        if o["stopInners"] > 0 and self.inner_iters >= o["stopInners"]:
            self.inner_iters = 0
        if o["stopOuters"] > 0 and self.outer_iters >= o["stopOuters"]:
            return stop("MAX OUTERS")
        self.outer_iters += 1
        self.aula_update()
        row["muNext"] = self.mu
        row["lambdaMax"], row["kappaAbsMax"] = self.multiplier_summary()
        row["active"] = self.count_active(phi)
        row["updated"] = True
        self.log.append(row)
        if self.verbose > 0:
            print(f"==nlp== it:{self.outer_iters:4d}  evals:{self.evals:4d}  A(x):{self.fx:11g}  mu:{self.mu:g}")
        self.x_before = self.x.copy()
        self.alpha = max(1e-2, min(1.0, self.alpha))
        self.tol = dict(stopTolerance=o["stopTolerance"], stopFTolerance=o["stopFTolerance"], stopGTolerance=o["stopGTolerance"])
        return False

    def run_inner_loop(self):
        steps, before = 0, self.outer_iters
        while not self.done:
            self.ministep()
            steps += 1
            if self.outer_iters != before:
                break
        return steps

    def run_to_end(self):
        while not self.ministep():
            pass
        return self

    def buffered_errors(self):
        return None if self.buf is None else summarize_errors(self.nlp.types, self.buf[1])


class NLP_Solver:
    """`ry.NLP_Solver`'s surface on the ported loop (the supported subset; see the module docstring)."""

    def __init__(self, nlp=None, verbose=None):
        self._opts = dict(RAI_DEFAULTS)
        if verbose is not None:
            self._opts["verbose"] = int(verbose)
        self.nlp = None
        self._x_init = None
        self._dual_init = None
        self.state = None
        self.dual = None
        self.log = []
        if nlp is not None:
            self.setProblem(nlp)

    def setProblem(self, nlp):
        if nlp.bounds is not None and np.asarray(nlp.bounds).size:
            raise NotImplementedError("rai_nlp: bounded problems are not ported (RAI's boundedNewton path)")
        self.nlp = nlp
        return self

    def setPyProblem(self, nlp):
        return self.setProblem(nlp)

    def setOptions(self, **kw):
        """RAI's setOptions resets every unnamed option to its default."""
        opts = dict(RAI_DEFAULTS)
        for k, v in kw.items():
            if k not in opts:
                raise KeyError(f"rai_nlp: unknown option {k!r}")
            opts[k] = v
        if opts["method"] not in METHODS:
            raise NotImplementedError(f"rai_nlp: method {opts['method']!r} is not ported (supported: {METHODS})")
        for k in UNSUPPORTED_NONDEFAULT:
            if opts[k] != RAI_DEFAULTS[k]:
                raise NotImplementedError(f"rai_nlp: option {k} = {opts[k]!r} selects a path the port does not implement (RAI's default {RAI_DEFAULTS[k]!r})")
        if opts["boundedNewton"] is not True:
            raise NotImplementedError("rai_nlp: boundedNewton = False is not a ported path (the port has no bounds anyway)")
        self._opts = opts
        return self

    def getOptions(self):
        return Options(self._opts)

    def setInitialization(self, x0):
        self._x_init = np.array(x0, dtype=float)
        return self

    def setDual(self, dual):
        self._dual_init = np.array(dual, dtype=float)
        return self

    def setTracing(self, *args):
        return self

    def solve(self, resampleInitialization=-1, verbose=-1):
        if self.nlp is None:
            raise RuntimeError("rai_nlp: setProblem first")
        if resampleInitialization == 1:
            raise NotImplementedError("rai_nlp: solve(1) resamples the initialization in RAI; the port never draws a sample")
        if self._x_init is None:
            raise NotImplementedError("rai_nlp: call setInitialization(x0) first; RAI's solve() and solve(0) would draw N(0, 0.01 rad) noise here, which the port never does")
        if len(self._x_init) != self.nlp.dimension:
            raise ValueError(f"rai_nlp: x0 has {len(self._x_init)} entries; the problem has {self.nlp.dimension}")
        opts = dict(self._opts)
        if verbose >= 0:
            opts["verbose"] = int(verbose)
        t0 = time.perf_counter()
        s = _State(self.nlp, self._x_init, opts, self._dual_init)
        if opts["verbose"] > 0:
            print(f"==nlp== problem-dim: {self.nlp.dimension}/{len(self.nlp.types)} method:{s.method} bounded: no")
            print(f"==nlp== it:0 evals:0 mu:{s.mu:g}")
        s.run_to_end()
        self.state = s
        self.log = s.log
        self.dual = s.lam.copy()
        errs = s.buffered_errors()
        ret = SolverReturn(evals=s.evals, sos=errs[1], ineq=errs[2], eq=errs[3], f=errs[0], feasible=rai_feasible(errs), done=True,
                           x=s.x.copy(), dual=s.lam.copy(), time=time.perf_counter() - t0, stop=s.reason, outer_iters=s.outer_iters)
        return ret

    # ---- port-only stepping (the explorer's "Newton step" / "Inner loop + update" buttons)
    def start(self):
        if self._x_init is None:
            raise NotImplementedError("rai_nlp: call setInitialization(x0) first")
        self.state = _State(self.nlp, self._x_init, self._opts, self._dual_init)
        self.log = self.state.log
        return self.state

    def ministep(self):
        if self.state is None:
            self.start()
        done = self.state.ministep()
        self.dual = self.state.lam.copy()
        return done

    def run_inner_loop(self):
        if self.state is None:
            self.start()
        n = self.state.run_inner_loop()
        self.dual = self.state.lam.copy()
        return n

    def getTrace_x(self):
        return np.array(self.state.trace_x) if self.state else np.zeros((0, self.nlp.dimension))

    def getTrace_errs(self):
        return np.array(self.state.trace_errs) if self.state else np.zeros((0, 4))

    @property
    def buffered_x(self):
        return None if self.state is None or self.state.buf is None else self.state.buf[0].copy()


# ----------------------------------------------------------------------------- the case-ID interface
CASE_RE = re.compile(
    r"^(?P<model>W|3P|X)-(?P<method>auglag|sqpen)-(?P<opts>default|tight|hist)-"
    r"(?P<start>line|shortcut|guessA|guessB|const|explorerA|explorerB)"
    r"(?:-s(?P<scale>\d+(?:\.\d+)?))?(?:-T(?P<T>\d+))?(?:-k(?P<k>\d+))?(?:-m(?P<margin>\d+(?:\.\d+)?))?(?P<nodisc>-nodisc)?"
    r"(?:-rrt(?P<rrt>\d+))?(?P<warm>-warm)?(?P<replay>-replay)?(?:-seed(?P<seed>\d+))?$")
TIGHT = dict(stopTolerance=1e-8, stopFTolerance=1e-12, stopEvals=5000)


class Case:
    """A parsed case ID (the grammar of the 2026-10-31 plan): model, method, opts, start and the suffix fields with their
    defaults filled in (scale 1, T 20, k 2, margin 5 mm for W, disc on, RRT seed 7 for shortcut starts)."""

    def __init__(self, case_id):
        m = CASE_RE.match(case_id)
        if not m:
            raise ValueError(f"rai_nlp: {case_id!r} is not a case ID of the grammar <model>-<method>-<opts>-<start>[-s][-T][-k][-m][-nodisc][-rrt][-warm][-replay][-seed]")
        d = m.groupdict()
        self.id = case_id
        self.model, self.start, self.opts = d["model"], d["start"], d["opts"]
        self.method = {"auglag": "AugmentedLag", "sqpen": "SquaredPenalty"}[d["method"]]
        self.scale = float(d["scale"]) if d["scale"] else 1.0
        self.T = int(d["T"]) if d["T"] else 20
        self.k = int(d["k"]) if d["k"] else 2
        self.margin_mm = float(d["margin"]) if d["margin"] else (5.0 if self.model == "W" else 0.0)
        self.disc = d["nodisc"] is None
        self.rrt = int(d["rrt"]) if d["rrt"] else (7 if self.start == "shortcut" else None)
        self.warm = d["warm"] is not None
        self.replay = d["replay"] is not None
        self.seed = int(d["seed"]) if d["seed"] else None
        if self.seed is not None and (self.warm or self.replay):
            raise ValueError(f"rai_nlp: {case_id!r}: -warm and -replay cases never carry -seed (the parent field names the seed)")
        if self.model == "X" and self.seed is not None:
            raise ValueError(f"rai_nlp: {case_id!r}: the explorer program has no KOMO program, so no seeded case exists")

    def options(self):
        """The RAI options of this case: `default` passes nothing (RAI's defaults), `tight` the three tight stops, `hist`
        the archived course settings (fixed mu = 10 x 2 dt^(1-2k))."""
        o = {"method": self.method}
        if self.opts == "tight":
            o.update(TIGHT)
        elif self.opts == "hist":
            mu = 10.0 * 2.0 * float(self.T) ** (2 * self.k - 1)
            o.update(TIGHT, muInit=mu, muInc=1.0, muMax=mu)
        return o

    def __repr__(self):
        return f"Case({self.id})"


def parse_case(case_id):
    return Case(case_id)


def options_for(case):
    return (case if isinstance(case, Case) else Case(case)).options()


def solve_case(nlp, case, x_start, dual=None, verbose=0):
    """Solve `nlp` (its rows already built for the case's model, scale, T, k, margin and disc) with the case's method and
    options from the exact vector `x_start`. Seeded cases are RAI-only (the port cannot add RAI's noise): replay them
    through the parent's recorded x_start under the `-replay` ID. Returns (z, lam, kap, info)."""
    case = case if isinstance(case, Case) else Case(case)
    if case.seed is not None:
        raise NotImplementedError(f"rai_nlp: {case.id} is a seeded RAI call; the port replays it as {case.id.replace('-seed' + str(case.seed), '')}-replay from the recorded x_start")
    solver = NLP_Solver(nlp, verbose=verbose)
    solver.setOptions(verbose=verbose, **case.options())
    solver.setInitialization(x_start)
    if dual is not None:
        solver.setDual(dual)
    ret = solver.solve(0)
    s = solver.state
    lam, kap = nlp.split_dual(s.lam)
    phi_x, _ = nlp.evaluate(ret.x)
    nlp.calls -= 1
    at_x = summarize_errors(nlp.types, phi_x)
    info = dict(case=case.id, method=case.method, options_passed=case.options(), options_effective=solver.getOptions().dict(),
                x_start=np.array(x_start, dtype=float).tolist(), evals=ret.evals, outer_iters=ret.outer_iters, stop=ret.stop,
                feasible=bool(ret.feasible), mu=s.mu,
                errors_buffered=dict(f=ret.f, sos=ret.sos, ineq=ret.ineq, eq=ret.eq),
                errors_at_x=dict(f=at_x[0], sos=at_x[1], ineq=at_x[2], eq=at_x[3]),
                buffered_x=s.buf[0].tolist(), x=ret.x.tolist(), dual=s.lam.tolist(), log=s.log,
                trace_x=np.array(s.trace_x), trace_errs=np.array(s.trace_errs), trials=s.trials, steps=s.steps,
                upstream=UPSTREAM)
    return ret.x.copy(), lam, kap, info


# ----------------------------------------------------------------------------- self-test: the tiny probe against RAI's recorded run
def _tiny_nlp():
    """probe_pynlp.py's TinyNLP: min (x-1)^2 + (y-2)^2  s.t.  x + y <= 1,  x - y = 0.5."""
    def evaluate(x):
        phi = np.array([x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 1.0, x[0] - x[1] - 0.5])
        J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
        return phi, J
    return NLP(2, ["sos", "sos", "ineq", "eq"], evaluate, x0=np.zeros(2))


def _self_test():
    # planning/opt-explorer-komo-al-2026-10-30/rai-probe.txt (robotic 0.4.1, bounds empty, verbose 2)
    expected_rows = [  # (it, evals, A, f, g, h, step) of the ==nlp== end-of-loop lines, six significant digits
        (0, 6, 2.08347, 1.40163, 0.657143, 0.5, 0.828571),
        (1, 10, 3.09226, 2.9343, 0.0626048, 0.0456587, 0.52444),
        (2, 11, 3.12496, 3.11954, 0.00176637, 0.00128625, 0.0526055),
        (3, 12, 3.125, 3.12534, 0.0, 8.05441e-05, 0.00162179),
    ]
    expected_mu = [5.0, 25.0, 125.0]
    expected_x = np.array([0.74998506, 0.24990452])
    expected_dual = np.array([0.0, 0.0, 2.02865286, -1.52089998])
    failures = []

    def check(cond, msg):
        print(("  OK    " if cond else "  FAIL  ") + msg)
        if not cond:
            failures.append(msg)

    solver = NLP_Solver(_tiny_nlp(), verbose=0)
    solver.setInitialization(np.zeros(2))
    ret = solver.solve(0)
    check(ret.evals == 12 and ret.outer_iters == 3 and ret.stop == "Delta<0.01",
          f"12 evaluations, 3 outer iterations, stop Delta<0.01: got {ret.evals}, {ret.outer_iters}, {ret.stop!r}")
    check(np.allclose(ret.x, expected_x, atol=1e-8), f"x = {ret.x} (RAI {expected_x})")
    check(np.allclose(ret.dual, expected_dual, atol=1e-8), f"dual = {ret.dual} (RAI {expected_dual})")
    check(abs(ret.sos - 3.12534165796545) < 1e-9 and abs(ret.eq - 8.05441470519419e-05) < 1e-12 and ret.ineq == 0.0 and ret.feasible,
          f"ret: sos {ret.sos:.14g}, ineq {ret.ineq}, eq {ret.eq:.14g}, feasible {ret.feasible} (RAI 3.12534165796545, 0, 8.05441470519419e-05, True)")
    rows = [r for r in solver.log]
    check(len(rows) == 4, f"four ==nlp== end-of-loop rows (got {len(rows)})")
    for r, e in zip(rows, expected_rows):
        got = (r["it"], r["evals"], r["A"], r["f"], r["g"], r["h"], r["step"])
        close = got[0] == e[0] and got[1] == e[1] and all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(got[2:], e[2:]))
        check(close, f"row it {e[0]}: evals {got[1]}, A {got[2]:.6g}, f {got[3]:.6g}, g {got[4]:.6g}, h {got[5]:.6g}, |x-x'| {got[6]:.6g} (RAI {e[1:]})")
    mus = [r["muNext"] for r in rows if r["updated"]]
    check(mus == expected_mu, f"mu after the updates {mus} (RAI {expected_mu})")
    tx = solver.getTrace_x()
    check(tx.shape == (12, 2) and solver.getTrace_errs().shape == (12, 4), f"trace shapes {tx.shape}, {solver.getTrace_errs().shape} (RAI (12, 2), (12, 4))")
    check(np.allclose(tx[5], [0.828571, 0.828571], atol=2e-6) and np.allclose(tx[9], [0.758473, 0.304132], atol=2e-6),
          "the traced points after the first two inner loops equal RAI's printed x")
    # the guard rails
    for label, fn in (("solve() without setInitialization raises", lambda: NLP_Solver(_tiny_nlp()).solve()),
                      ("solve(1) raises", lambda: NLP_Solver(_tiny_nlp()).setInitialization(np.zeros(2)).solve(1)),
                      ("a non-default stepMin raises", lambda: NLP_Solver(_tiny_nlp()).setOptions(stepMin=0.1)),
                      ("an unknown method raises", lambda: NLP_Solver(_tiny_nlp()).setOptions(method="LogBarrier")),
                      ("a seeded case ID is refused by solve_case", lambda: solve_case(_tiny_nlp(), "W-auglag-default-line-seed1", np.zeros(2)))):
        try:
            fn()
            check(False, label + " (no exception)")
        except NotImplementedError:
            check(True, label)
    # setOptions resets unnamed options
    s2 = NLP_Solver(_tiny_nlp()).setOptions(stopTolerance=1e-8)
    check(s2.getOptions().dict()["muInc"] == 5.0 and s2.getOptions().dict()["stopTolerance"] == 1e-8 and len(s2.getOptions().dict()) == 26,
          "setOptions names one option and resets the others to RAI's defaults (26 options)")
    # SquaredPenalty keeps lambda at zero and follows the same mu schedule
    s3 = NLP_Solver(_tiny_nlp()).setOptions(method="SquaredPenalty", verbose=0).setInitialization(np.zeros(2))
    r3 = s3.solve(0)
    check(np.all(r3.dual == 0.0) and s3.state.mu > 1.0 and r3.evals > 0, f"SquaredPenalty: dual {r3.dual}, mu {s3.state.mu:g}, evals {r3.evals}, x {r3.x}")
    # the case grammar
    c = Case("W-auglag-tight-shortcut-s10-rrt3-seed1")
    check(c.scale == 10.0 and c.rrt == 3 and c.seed == 1 and c.opts == "tight" and c.options()["stopEvals"] == 5000 and c.margin_mm == 5.0,
          f"parsed {c.id}: scale {c.scale}, rrt {c.rrt}, seed {c.seed}, options {c.options()}")
    c2 = Case("3P-sqpen-default-guessB-k1")
    check(c2.k == 1 and c2.method == "SquaredPenalty" and c2.margin_mm == 0.0 and c2.rrt is None and c2.options() == {"method": "SquaredPenalty"}, f"parsed {c2.id}")
    for bad in ("W-auglag-default-line-seed1-T40", "W-auglag-tight-line-s10-m5.1-warm-seed1", "X-auglag-default-explorerA-seed1", "W-auglag-default-foo"):
        try:
            Case(bad)
            check(False, f"{bad!r} is rejected")
        except ValueError:
            check(True, f"{bad!r} is rejected")
    print(f"rai_nlp self-test: {len(failures)} failure(s)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _self_test() else 0)
