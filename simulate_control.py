"""
Simulation dynamique finale, corrigée.

  - commande échantillonnée à Tc (bloqueur d'ordre zéro), 200 Hz par défaut ;
  - intégration RK4 à sous-pas h = Tc / n_sub, découplée de la commande ;
  - gains PD mis à l'échelle de l'inertie nominale (bande passante égale à CT).
Sauvegarde toutes les séries nécessaires aux figures et au bilan énergétique.
  usage : python3 simulate_control.py <ct|pdg|pd> <Tc_ms> <n_sub> [payload_kg]
"""
import sys, json, time
from paths import RESULTS
import numpy as np
from agrirob_model import (forward_kinematics, inverse_kinematics, mass_matrix,
                           coriolis_matrix, gravity_vector)

LAW, Tc, NSUB = sys.argv[1], float(sys.argv[2]) / 1e3, int(sys.argv[3])
PAYLOAD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
import agrirob_model as am
NOM = am.LINK_MASSES.copy()
PLANT = NOM.copy(); PLANT[5] += PAYLOAD      # charge non modélisée au segment 6
T_TOT = 4.0
q0 = np.array([0.0, -0.3, 0.5, 0.0, 0.5, 0.0])
T0 = forward_kinematics(q0); Rd, p0 = T0[:3, :3], T0[:3, 3].copy()
pf = p0 + np.array([0.15, 0.20, 0.10])
KP = np.array([200, 250, 200, 150, 100, 80.]); KD = np.array([30, 40, 30, 20, 15, 12.])
MD0 = np.diag(mass_matrix(q0))
fric = lambda v: 0.5 * v + 0.3 * np.tanh(10 * v)
ETA = 0.70

def traj(tt):
    s = np.clip(tt / T_TOT, 0, 1)
    return p0 + (pf - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)

Nc = int(round(T_TOT / Tc)) + 1
tc = np.linspace(0, T_TOT, Nc)
qd = np.zeros((Nc, 6)); qt = q0.copy()
for k, tk in enumerate(tc):
    Td = np.eye(4); Td[:3, :3] = Rd; Td[:3, 3] = traj(tk)
    qt, _, _ = inverse_kinematics(Td, qt, max_iter=100, tol=1e-9, position_only=True)
    qd[k] = qt
vd = np.gradient(qd, Tc, axis=0); ad = np.gradient(vd, Tc, axis=0)

def acc(q, v, tau):
    # le système réel porte la charge ; le contrôleur ne la connaît pas
    am.LINK_MASSES[:] = PLANT
    try:
        return np.linalg.solve(mass_matrix(q), tau - coriolis_matrix(q, v) @ v
                               - gravity_vector(q) - fric(v))
    finally:
        am.LINK_MASSES[:] = NOM

def control(k, q, v):
    e, ed = qd[k] - q, vd[k] - v
    if LAW == 'ct':
        return (mass_matrix(q) @ (ad[k] + KP * e + KD * ed)
                + coriolis_matrix(q, v) @ v + gravity_vector(q) + fric(v))
    b = MD0 * KP * e + MD0 * KD * ed
    return b + gravity_vector(q) if LAW == 'pdg' else b

h = Tc / NSUB
q, v = q0.copy(), np.zeros(6)
err = np.zeros(Nc); pa = np.zeros((Nc, 3)); pdes = np.zeros((Nc, 3))
tau_l = np.zeros((Nc, 6)); v_l = np.zeros((Nc, 6)); pm_sub = []
t_start = time.time()
for k in range(Nc):
    pa[k] = forward_kinematics(q)[:3, 3]; pdes[k] = traj(tc[k])
    err[k] = np.linalg.norm(pdes[k] - pa[k]) * 1e3
    tau = np.clip(control(k, q, v), -150, 150)
    tau_l[k], v_l[k] = tau, v
    if k == Nc - 1:
        break
    for _ in range(NSUB):
        pm_sub.append(np.sum(np.abs(tau * v)))
        k1q, k1v = v, acc(q, v, tau)
        k2q, k2v = v + .5*h*k1v, acc(q + .5*h*k1q, v + .5*h*k1v, tau)
        k3q, k3v = v + .5*h*k2v, acc(q + .5*h*k2q, v + .5*h*k2v, tau)
        k4q, k4v = v + h*k3v, acc(q + h*k3q, v + h*k3v, tau)
        q = q + h/6 * (k1q + 2*k2q + 2*k3q + k4q)
        v = v + h/6 * (k1v + 2*k2v + 2*k3v + k4v)

pm_sub = np.array(pm_sub)
tag = f"{LAW}_Tc{sys.argv[2]}_sub{NSUB}" + (f"_m{PAYLOAD:g}" if PAYLOAD else "")
summ = {'tag': tag, 'law': LAW, 'payload_kg': PAYLOAD, 'Tc_ms': Tc * 1e3, 'h_ms': h * 1e3,
        'rms_mm': float(np.sqrt(np.mean(err**2))), 'max_mm': float(err.max()),
        'final_mm': float(err[-1]),
        'E_mech_J': float(pm_sub.sum() * h), 'P_mech_mean_W': float(pm_sub.mean()),
        'P_elec_mean_W': float(pm_sub.mean() / ETA),
        'peak_tau_per_joint': np.abs(tau_l).max(axis=0).round(3).tolist(),
        'signed_peak_tau': [float(tau_l[np.argmax(np.abs(tau_l[:, j])), j])
                            for j in range(6)],
        'wall_s': time.time() - t_start}
np.savez(RESULTS / f"sim_{tag}.npz", t=tc, err=err, pa=pa, pd=pdes, tau=tau_l, v=v_l)
json.dump(summ, open(RESULTS / f"sim_{tag}.json", 'w'), indent=1)
print(f"{tag:22} RMS {summ['rms_mm']:7.3f}  max {summ['max_mm']:7.3f} mm  "
      f"E {summ['E_mech_J']:6.2f} J  pic {max(summ['peak_tau_per_joint']):6.2f} N·m  "
      f"({summ['wall_s']:.0f} s)", flush=True)
