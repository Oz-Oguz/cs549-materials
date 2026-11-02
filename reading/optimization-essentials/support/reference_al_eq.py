"""Reference for increment 3's events: the course's augmented-Lagrangian loop with equality and inequality rows written
from komo_two_link.py's rules (kappa <- kappa + 2 mu h, lambda_i <- max(lambda_i + 2 mu g_i, 0), rows included while
g_i > 0 or lambda_i > 0) and constrained_ik.py's newton_armijo, with the demo's stated parameters, without importing
the instrumented loop. It is a re-transcription of the same rules in the same operation order (only Drake's FK and the
Jacobian are shared), so it agrees with the demo bit for bit; it checks the demo's event hook and bookkeeping, not the
arithmetic. Writes reference-iterates.json (q, kappa and lambda per timeline event, both runs); the demo's --selftest
compares the events to 1e-10.

Run:  .venv/bin/python planning/optimization-step-visuals-2026-10-14/increment3/reference_al_eq.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drake/02-gauss-newton-ik"))
import gn_ik_panda as gn

MU_H, MU_G, RHO = 1000.0, 100.0, 1e-3
INNER_TOL, INNER_MAX, C, FLOOR = 3e-7, 100, 0.01, 1e-12
FEAS_H, FEAS_G, STAT, COMP, OUTER_MAX = 1e-6, 1e-6, 1e-6, 1e-8, 30
Q_REF = gn.Q_READY
RUNS = {"runA": ([0.45, 0.10, 0.45], gn.Q_READY), "runB": ([-0.20, 0.0, 0.60], gn.Q_READY + np.array([0.01, 0, 0, 0, 0, 0, 0]))}


def run(phi, J, lo, hi, y, q_init):
    E = np.vstack([-np.eye(7), np.eye(7)])
    g = lambda z: np.concatenate([lo - z, z - hi])
    h = lambda z: phi(z) - y
    f = lambda z: 0.5 * float(np.sum((z - Q_REF) ** 2))
    z, kap, lam = q_init.copy(), np.zeros(3), np.zeros(14)
    qs, ks, ls = [z.tolist()], [kap.tolist()], [lam.tolist()]
    for _ in range(OUTER_MAX):
        act = lambda x: ((g(x) > 0) | (lam > 0)).astype(float)                      # komo_two_link.py: active = (g > 0) | (lam > 0)
        F = lambda x: f(x) + float(kap @ h(x)) + MU_H * float(h(x) @ h(x)) + float(np.sum(act(x) * (lam * g(x) + MU_G * g(x) ** 2)))
        G = lambda x: (x - Q_REF) + J(x).T @ (kap + 2 * MU_H * h(x)) + E.T @ (act(x) * (lam + 2 * MU_G * g(x)))
        H = lambda x: np.eye(7) + 2 * MU_H * J(x).T @ J(x) + 2 * MU_G * E.T @ np.diag(act(x)) @ E + RHO * np.eye(7)
        st = "limit"
        for _ in range(INNER_MAX):                                                   # newton_armijo with the Gauss-Newton model
            gr = G(z)
            if np.linalg.norm(gr) < INNER_TOL:
                st = "stationary"; break
            d = -np.linalg.solve(H(z), gr)
            a, F0, slope = 1.0, F(z), float(gr @ d)
            if -slope < 1e-13 * max(1.0, abs(F0)):                                      # the demo's rounding floor of the Armijo test
                st = "stationary"; break
            ok = lambda aa: np.isfinite(F(z + aa * d)) and F(z + aa * d) <= F0 + C * aa * slope
            while not ok(a) and a > FLOOR:
                a *= 0.5
            if not ok(a):
                st = "stalled"; break
            z = z + a * d
            qs.append(z.tolist()); ks.append(kap.tolist()); ls.append(lam.tolist())
        hz, gz = h(z), g(z)
        kap_new = kap + 2 * MU_H * hz; lam_new = np.maximum(lam + 2 * MU_G * gz, 0.0)
        feas_h, feas_g = np.abs(hz).max(), max(gz.max(), 0.0)
        stat = np.linalg.norm((z - Q_REF) + J(z).T @ kap_new + E.T @ lam_new); comp = np.abs(lam_new * gz).max()
        qs.append(z.tolist()); ks.append(kap_new.tolist()); ls.append(lam_new.tolist())
        kap, lam = kap_new, lam_new
        if st == "stalled" or (feas_h < FEAS_H and feas_g < FEAS_G and stat < STAT and comp < COMP):
            break
    return qs, ks, ls, kap, lam


if __name__ == "__main__":
    d, plant, pc, ctx, Panda, ee = gn.build()
    phi, J = gn.make_fk(plant, pc, Panda, ee)
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    out = {}
    for key, (y, q_init) in RUNS.items():
        qs, ks, ls, kap, lam = run(phi, J, lo, hi, np.array(y), np.array(q_init))
        out[key] = {"q": qs, "kappa": ks, "lam": ls}
        print(f"{key}: {len(qs) - 1} events, kappa* {np.round(kap, 5)}, lambda_2 {lam[1]:.5f}, f* {0.5 * np.sum((np.array(qs[-1]) - Q_REF) ** 2):.8f}")
    Path(__file__).with_name("reference-iterates.json").write_text(json.dumps(out))
    print("wrote reference-iterates.json")
