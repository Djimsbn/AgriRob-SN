"""
Statistics of the damped-least-squares inverse kinematics (Section 7.3).

  1. Warm-started solve at every control instant of the picking trajectory.
  2. Cold start from a fixed initial guess on 500 random reachable targets,
     full pose and position only.
  3. Convergence rate by quartile of joint-space distance to the initial guess.

Outputs (results/): ik_statistics.json, ik_records.json, ik_distance.json
Run time: about 30 s.
"""
import json
import numpy as np
from agrirob_model import forward_kinematics, jacobian, inverse_kinematics, JOINT_LIMITS
from paths import RESULTS

# ----------------------------------------------------------------- 1. warm start
T_TOT, TC = 4.0, 0.005
q0 = np.array([0.0, -0.3, 0.5, 0.0, 0.5, 0.0])
T0 = forward_kinematics(q0)
Rd, p0 = T0[:3, :3], T0[:3, 3].copy()
pf = p0 + np.array([0.15, 0.20, 0.10])
t = np.linspace(0, T_TOT, int(T_TOT / TC) + 1)
qt, ok = q0.copy(), 0
for tk in t:
    s = np.clip(tk / T_TOT, 0, 1)
    Td = np.eye(4); Td[:3, :3] = Rd
    Td[:3, 3] = p0 + (pf - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)
    qt, conv, _ = inverse_kinematics(Td, qt, max_iter=100, tol=1e-6, position_only=True)
    ok += int(conv)
warm = {'steps': len(t), 'converged': ok}
print(f"warm start along the trajectory: {ok}/{len(t)}")

# ----------------------------------------------------------------- 2. cold start
SEED, NT = 2026, 500
Q_INIT = np.array([0.1, -0.3, 0.5, 0.2, 0.3, 0.1])
rng = np.random.default_rng(SEED)
targets, recs = [], {'full': [], 'pos': []}
for _ in range(NT):
    q_t = rng.uniform(JOINT_LIMITS[:, 0] * 0.7, JOINT_LIMITS[:, 1] * 0.7)
    targets.append(q_t)
    Tt = forward_kinematics(q_t)
    J = jacobian(q_t)
    w = float(np.sqrt(abs(np.linalg.det(J @ J.T))))
    for mode in ('full', 'pos'):
        _, conv, hist = inverse_kinematics(Tt, Q_INIT, max_iter=200, tol=1e-4,
                                           position_only=(mode == 'pos'))
        recs[mode].append({'w': w, 'conv': bool(conv), 'it': len(hist)})

def summarise(rs):
    c = np.array([r['conv'] for r in rs]); w = np.array([r['w'] for r in rs])
    its = np.array([r['it'] for r in rs if r['conv']])
    by = []
    for lo, hi in zip([0, 1e-4, 1e-3, 1e-2], [1e-4, 1e-3, 1e-2, 1]):
        m = (w >= lo) & (w < hi)
        by.append({'w_lo': lo, 'w_hi': hi, 'n': int(m.sum()),
                   'rate': float(c[m].mean()) if m.sum() else None})
    return {'n': len(rs), 'converged': int(c.sum()), 'rate': float(c.mean()),
            'it_mean': float(its.mean()), 'it_median': float(np.median(its)),
            'it_p95': float(np.percentile(its, 95)), 'by_manipulability': by}

E = {'n_targets': NT, 'q_init': Q_INIT.tolist(), 'max_iter': 200, 'tol': 1e-4,
     'lambda': 0.01, 'seed': SEED,
     'full_pose': summarise(recs['full']), 'position_only': summarise(recs['pos'])}
for k in ('full_pose', 'position_only'):
    s = E[k]
    print(f"{k:14}: {s['converged']}/{s['n']} ({100*s['rate']:.1f} %), "
          f"mean {s['it_mean']:.1f} it., median {s['it_median']:.0f}")

# ----------------------------------------------------------------- 3. distance
d = np.array([np.linalg.norm(q - Q_INIT) for q in targets])
edges = np.quantile(d, [0, .25, .5, .75, 1.0])
dist = {}
for mode in ('full', 'pos'):
    c = np.array([r['conv'] for r in recs[mode]])
    dist[mode] = [[float(lo), float(hi), int(((d >= lo) & (d <= hi)).sum()),
                   float(c[(d >= lo) & (d <= hi)].mean())]
                  for lo, hi in zip(edges[:-1], edges[1:])]

json.dump({'E': E, 'ik_warmstart_tracking': warm},
          open(RESULTS / 'ik_statistics.json', 'w'), indent=2)
json.dump(recs, open(RESULTS / 'ik_records.json', 'w'))
json.dump(dist, open(RESULTS / 'ik_distance.json', 'w'))
print("written: results/ik_statistics.json, ik_records.json, ik_distance.json")
