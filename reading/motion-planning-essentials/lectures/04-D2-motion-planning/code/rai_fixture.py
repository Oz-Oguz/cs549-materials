"""Shared helper of the real-RAI fixture writers (container side; robotic 0.4.1 is imported lazily).

A fixture case is one recorded run of `ry.NLP_Solver` under a case ID of rai_nlp.Case's grammar. The record carries
what the 2026-10-31 plan requires so that a figure or a number can be traced to the run that produced it: the model,
method, options passed and effective, constraint scale, T, k, margin, the nominal start, the first evaluated vector
(x_start = trace_x[0]), the noise seed, the RRT seed, the parent case (warm starts and replays), the call, the program
(`twin`: the NumPy rows through setPyProblem; `komo`: KOMO's own NLP), the binary and the upstream revision, every
evaluated point and its errors, RAI's return with ret.x, the buffered last-evaluated point (trace_x[-1]), KOMO's
getPath() where the program is `komo`, the dual, the parsed ==nlp== log, and the physical residuals recomputed outside
RAI at ret.x by the writer's own `residuals(x)`.

The start is asserted by kind: a seeded case starts away from the nominal vector (RAI's noise), a `-warm` case at its
parent's ret.x, a `-replay` case at its parent's x_start, and every other case exactly at the nominal vector.
`-seed` cases are accepted only on program `komo`: a setPyProblem twin's getInitializationSample returns x0 itself.
"""
import datetime
import hashlib
import importlib.metadata
import importlib.util
import os
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import rai_nlp  # noqa: E402
from rai_nlp import Case, summarize_errors  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location("explorer_nlp", ROOT / "modules" / "opt-explorer" / "fixtures" / "explorer_nlp.py")
_E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_E)
parse_log = _E.parse_log

_ry = None
TRACE_ROWS = 1000      # traces longer than this (the 5000-evaluation cycles) keep their first rows and a flag


def ry():
    global _ry
    if _ry is None:
        import robotic
        _ry = robotic
    return _ry


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def header(writer_path, note, problem_text=None, extra=None):
    h = {"robotic": importlib.metadata.version("robotic"), "compiled": ry().compiled(), "date": datetime.date.today().isoformat(),
         "upstream": rai_nlp.UPSTREAM["rai_commit"], "upstream_files": rai_nlp.UPSTREAM["files"], "image": os.environ.get("RAI_IMAGE"),
         "problem_sha256": sha(problem_text) if problem_text is not None else None,
         "writer_sha256": sha(pathlib.Path(writer_path).read_bytes()), "note": note}
    if extra:
        h.update(extra)
    return h


def method_enum(name):
    return getattr(ry().OptMethod, name)


class TwinNLP:
    """A setPyProblem problem from NumPy row blocks (the same stacking as rai_nlp.NLP.from_blocks: sos, eq, ineq),
    with the eq and ineq rows multiplied by `scale` (KOMO's objective scale on those rows)."""

    def __init__(self, n, sos, eq, ineq, x0, scale=1.0):
        R = ry()
        self.dimension = n
        self.sos, self.eq, self.ineq, self.scale = sos, eq, ineq, float(scale)
        probe = np.asarray(x0, dtype=float)
        ns, ne, ni = len(sos(probe)[0]), len(eq(probe)[0]), len(ineq(probe)[0])
        self.types = [int(R.OT.sos)] * ns + [int(R.OT.eq)] * ne + [int(R.OT.ineq)] * ni
        self.types_str = ["sos"] * ns + ["eq"] * ne + ["ineq"] * ni
        self.bounds = np.zeros((2, 0))
        self.x0 = probe.copy()
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        r, Jr = self.sos(x)
        h, Jh = self.eq(x)
        g, Jg = self.ineq(x)
        s = self.scale
        return np.concatenate([r, s * h, s * g]), np.vstack([Jr, s * Jh, s * Jg])

    def getInitializationSample(self):
        return self.x0.copy()


def _capture(fn, verbose):
    """Run fn() with the C++ stdout (fd 1) captured; returns (result, text)."""
    log = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".log")
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(log.fileno(), 1)
    try:
        out = fn()
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
    text = pathlib.Path(log.name).read_text()
    os.unlink(log.name)
    return out, text


def run_case(case, *, program, solver, nlp_eval, nominal_x, x_start_override=None, parent=None, residuals=None,
             komo=None, model_fields=None, verbose=3):
    """Run one case on `solver` (a ry.NLP_Solver that already holds its problem) and return its record.

    nlp_eval(x) -> (phi, J) recomputes the rows in NumPy at ret.x (the twin's or the model's rows, unscaled or scaled
    as the writer decides; the record says which through model_fields['rows_at_x']). `residuals(x) -> dict` gives
    the physical residuals (knot deficit in mm, goal error in rad, ...). `parent` is the parent record for -warm and
    -replay cases; `x_start_override` is the exact start for those (the parent's ret.x or x_start)."""
    case = case if isinstance(case, Case) else Case(case)
    if case.seed is not None and program != "komo":
        raise ValueError(f"{case.id}: a seeded case must run on program komo (a twin's sample is x0 itself)")
    if (case.warm or case.replay) and parent is None:
        raise ValueError(f"{case.id}: -warm and -replay cases need their parent record")
    opts = case.options()
    kw = dict(opts)
    kw["method"] = method_enum(kw["method"])
    solver.setOptions(verbose=verbose, **kw)                 # RAI resets every unnamed option to its default
    effective = solver.getOptions().dict()
    effective["method"] = opts["method"]                      # RAI's dict does not echo the method (probe 2026-10-31)
    solver.setTracing(True, True, False, False)
    nominal_x = np.asarray(nominal_x, dtype=float)
    if case.seed is not None:
        ry().rnd_seed(case.seed)
        call = f"ry.rnd_seed({case.seed}); solve()"
        ret, text = _capture(lambda: solver.solve(-1, verbose), verbose)
    else:
        x0 = nominal_x if x_start_override is None else np.asarray(x_start_override, dtype=float)
        solver.setInitialization(x0.copy())
        call = "setInitialization + solve(0)"
        ret, text = _capture(lambda: solver.solve(0, verbose), verbose)
    tx = np.array(solver.getTrace_x())
    te = np.array(solver.getTrace_errs())
    x_start = tx[0]
    x_end = np.array(ret.x)
    # the start, asserted by kind
    if case.seed is not None:
        assert float(np.max(np.abs(x_start - nominal_x))) > 0.0, f"{case.id}: the seeded call started exactly at the nominal vector"
    elif case.warm:
        assert np.array_equal(x_start, np.array(parent["x"])), f"{case.id}: a warm start must begin at the parent's ret.x"
    elif case.replay:
        assert np.array_equal(x_start, np.array(parent["x_start"])), f"{case.id}: a replay must begin at the parent's x_start"
    else:
        assert np.array_equal(x_start, nominal_x), f"{case.id}: an exact case must begin at the nominal vector"
    parsed = parse_log(text)
    phi_end, _ = nlp_eval(x_end)
    rec = {
        "case": case.id, "model": case.model, "method": case.method, "opts": case.opts, "start": case.start,
        "options_passed": opts, "options_effective": effective, "constraint_scale": case.scale, "T": case.T, "k": case.k,
        "margin_mm": case.margin_mm, "disc": case.disc, "rrt_seed": case.rrt, "seed": case.seed,
        "parent": None if parent is None else parent["case"], "call": call, "program": program,
        "nominal_x": nominal_x.tolist(), "x_start": x_start.tolist(),
        "ret": {"evals": int(ret.evals), "sos": float(ret.sos), "ineq": float(ret.ineq), "eq": float(ret.eq), "f": float(ret.f),
                "feasible": bool(ret.feasible), "done": bool(ret.done), "time": float(ret.time)},
        "x": x_end.tolist(), "x_end": x_end.tolist(), "buffered_x": tx[-1].tolist(),
        "komo_path": None if komo is None else np.asarray(komo.getPath()).flatten().tolist(),
        "dual": np.array(solver.dual).tolist(),
        "trace_x": tx[:TRACE_ROWS].tolist(), "trace_errs": te[:TRACE_ROWS].tolist(),
        "trace_rows": int(tx.shape[0]), "trace_truncated": bool(tx.shape[0] > TRACE_ROWS),
        "outer_iters": len(parsed["updates"]), "mu_sequence": [u["mu"] for u in parsed["updates"]],
        "stop": parsed["stop"], "inner_loops": parsed["inner_loops"], "updates": parsed["updates"], "newton_lines": parsed["newton_lines"],
        "phi_at_x": phi_end.tolist(),
        "residuals_at_x": None if residuals is None else residuals(x_end),
        "residuals_at_buffered": None if residuals is None or np.array_equal(tx[-1], x_end) else residuals(tx[-1]),
        "upstream": rai_nlp.UPSTREAM["rai_commit"],
    }
    if model_fields:
        rec.update(model_fields)
    print(f"{case.id} [{program}]: evals {ret.evals} (trace {tx.shape[0]}), outer {rec['outer_iters']}, sos {ret.sos:.6f}, ineq {ret.ineq:.3e}, eq {ret.eq:.3e}, "
          f"feasible {bool(ret.feasible)}, stop {parsed['stop']}; |x_start - nominal| {float(np.max(np.abs(x_start - nominal_x))):.2e}"
          + (f"; residuals {rec['residuals_at_x']}" if residuals else ""))
    return rec



def write_fixture(path, header_dict, cases, merge=True):
    """Write {header, cases}; with merge=True the cases of an existing file under the same problem sha256 are kept."""
    import json
    path = pathlib.Path(path)
    out = {"header": header_dict, "cases": {}}
    if merge and path.exists():
        old = json.loads(path.read_text())
        if old.get("header", {}).get("problem_sha256") == header_dict.get("problem_sha256"):
            out["cases"].update(old.get("cases", {}))
        else:
            print(f"note: {path.name} was written against another problem file; its cases are dropped")
    out["cases"].update(cases)
    path.write_text(json.dumps(out))
    print("wrote", path, path.stat().st_size, "bytes,", len(out["cases"]), "cases")
    return out
