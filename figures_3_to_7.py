"""
Figures révisées, entièrement dérivées des simulations corrigées
(commande à 200 Hz, intégration RK4 à 1 ms) et des statistiques du MGI.
Chaque figure est tracée depuis des fichiers de résultats ; aucune valeur
n'est saisie à la main.
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from agrirob_model import (forward_kinematics, inverse_kinematics, jacobian,
                           MobileBase, JOINT_LIMITS)

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 15, 'axes.titlesize': 16,
    'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 13, 'legend.framealpha': 0.92, 'lines.linewidth': 2.2,
    'axes.grid': True, 'grid.alpha': 0.28, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
})
from paths import RESULTS, FIGURES
OUT = str(FIGURES)
TAG = 'Tc5_sub5'

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png', dpi=200)
    plt.close(fig); print('  ', name)

LAWS = ('ct', 'pdg', 'pd')
PAYS = ['0', '0.5', '1', '2']
def tag(law, m, sub=5):
    return f"{law}_Tc5_sub{sub}" + ("" if m == '0' else f"_m{m}")
nom = np.load(f'{RESULTS}/sim_{tag("ct","0")}.npz')          # couples nominaux
one = {l: np.load(f'{RESULTS}/sim_{tag(l,"1")}.npz') for l in LAWS}
sone = {l: json.load(open(f'{RESULTS}/sim_{tag(l,"1")}.json')) for l in LAWS}
sw = {l: [json.load(open(f'{RESULTS}/sim_{tag(l,m)}.json')) for m in PAYS] for l in LAWS}
ct10 = json.load(open(f'{RESULTS}/sim_{tag("ct","0",10)}.json'))
t = nom['t']; ct = one['ct']

# ---------------------------------------------------------------- Fig. 3
# The reference is a straight segment, so a 3D line view is nearly featureless
# and crowds three axes into half a figure. Panel (a) therefore resolves the
# deviation from that segment along each Cartesian axis, on one clear scale.
dev = (ct['pa'] - ct['pd']) * 1e3
norm = np.linalg.norm(ct['pa'] - ct['pd'], axis=1) * 1e3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6))
for k, (lab, c) in enumerate(zip(('$X$', '$Y$', '$Z$'),
                                 ['#1b7837', '#762a83', '#b35806'])):
    ax1.plot(t, dev[:, k], color=c, lw=2.4, label=f'{lab} deviation')
ax1.plot(t, norm, color='0.25', lw=1.6, ls=':', label='magnitude')
ax1.axhline(0, color='0.55', lw=1.0)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Deviation from reference path (mm)')
ax1.set_title('(a) Per-axis tracking deviation, computed torque', pad=8)
ax1.set_xlim(0, t[-1])
ax1.legend(loc='lower right', fontsize=12.5, ncol=2, columnspacing=1.0)

st3 = {'ct': ('#d62728', '-', 'computed torque'),
       'pdg': ('#1f77b4', '--', 'PD + gravity'),
       'pd': ('#7f7f7f', '-.', 'PD alone')}
for l in ('pd', 'pdg', 'ct'):
    c, ls, lab = st3[l]
    ax2.semilogy(t, np.maximum(one[l]['err'], 1e-2), ls, color=c, lw=2.4,
                 label=f"{lab} — RMS {sone[l]['rms_mm']:.1f} mm")
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position error magnitude (mm)')
ax2.set_title('(b) Tracking error, three control laws', pad=8)
ax2.set_xlim(0, t[-1]); ax2.set_ylim(1e-1, 2e2)
ax2.legend(loc='lower right', fontsize=12.5)
fig.suptitle('Tomato-picking motion with a 1 kg unmodelled payload',
             fontsize=15, y=1.01)
fig.subplots_adjust(wspace=0.26); save(fig, 'fig3_tracking')

# ---------------------------------------------------------------- Fig. 4
names = ['1 — base rotation', '2 — shoulder', '3 — elbow',
         '4 — wrist roll', '5 — wrist pitch', '6 — wrist yaw']
col = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
tau = nom['tau']
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), sharex=True)
for j in range(6):
    ax = axes[j // 3, j % 3]
    ax.plot(t, tau[:, j], color=col[j], lw=2.4)
    ax.fill_between(t, 0, tau[:, j], alpha=0.16, color=col[j])
    ax.axhline(0, color='0.5', lw=0.9)
    pk = tau[np.argmax(np.abs(tau[:, j])), j]
    ax.set_title(f'Joint {names[j]}\npeak $={pk:+.2f}$ N$\\cdot$m', fontsize=14.5, pad=6)
    ax.set_ylabel(r'$\tau_{%d}$ (N$\cdot$m)' % (j + 1)); ax.margins(y=0.14)
    if j >= 3: ax.set_xlabel('Time (s)')
axes[0, 0].set_xlim(0, t[-1]); fig.subplots_adjust(hspace=0.46, wspace=0.32)
save(fig, 'fig4_joint_torques')

# ---------------------------------------------------------------- Fig. 5 (new)
pv = np.array([float(m) for m in PAYS])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.0, 5.6))
st5 = {'ct': ('#d62728', 'o', 'computed torque'), 'pdg': ('#1f77b4', 's', 'PD + gravity'),
       'pd': ('#7f7f7f', '^', 'PD alone')}
for l in ('pd', 'pdg', 'ct'):
    c, mk, lab = st5[l]
    rms = np.array([r['rms_mm'] for r in sw[l]])
    if l == 'ct':
        rms[0] = ct10['rms_mm']          # borne supérieure : résolution d'intégration
        a1.semilogy(pv[1:], rms[1:], '-', marker=mk, color=c, ms=9, lw=2.4, label=lab)
        a1.semilogy([pv[0], pv[1]], [rms[0], rms[1]], ':', color=c, lw=1.8)
        a1.plot(pv[0], rms[0], marker='v', color=c, ms=11, ls='none',
                label='CT, exact model: upper bound')
    else:
        a1.semilogy(pv, rms, '-', marker=mk, color=c, ms=9, lw=2.4, label=lab)
    a2.plot(pv, [r['E_mech_J'] for r in sw[l]], '-', marker=mk, color=c, ms=9, lw=2.4, label=lab)
a1.set_xlabel('Unmodelled payload (kg)'); a1.set_ylabel('RMS tracking error (mm)')
a1.set_title('(a) Tracking accuracy', pad=6); a1.set_xticks(pv)
a1.legend(loc='lower right', fontsize=12)
a2.set_xlabel('Unmodelled payload (kg)'); a2.set_ylabel('Joint mechanical energy (J)')
a2.set_title('(b) Energy over the 4 s picking motion', pad=6); a2.set_xticks(pv)
a2.legend(loc='upper left', fontsize=12)
fig.subplots_adjust(wspace=0.24); save(fig, 'fig5_controller_comparison')

# ---------------------------------------------------------------- Fig. 6 (IK)
res = json.load(open(f'{RESULTS}/ik_statistics.json'))['E']
dist = json.load(open(f'{RESULTS}/ik_distance.json'))
rec = json.load(open(f'{RESULTS}/ik_records.json'))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.0, 5.6))
lbl = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(4); wdt = 0.38
for off, mode, c, name in ((-wdt/2, 'full', '#1f77b4', 'full pose'),
                           (wdt/2, 'pos', '#ff7f0e', 'position only')):
    rates = [100 * r[3] for r in dist[mode]]
    b = a1.bar(x + off, rates, wdt, color=c, edgecolor='0.2', label=name)
    for rr, v in zip(b, rates):
        a1.text(rr.get_x() + rr.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=12)
ticks = [f'{l}\n[{r[0]:.2f}, {r[1]:.2f}]' for l, r in zip(lbl, dist['full'])]
a1.set_xticks(x); a1.set_xticklabels(ticks, fontsize=12)
a1.set_xlabel(r'quartile of $\|\mathbf{q}_\mathrm{target}-\mathbf{q}_\mathrm{init}\|$ (rad)')
a1.set_ylabel('Convergence rate (%)'); a1.set_ylim(0, 108); a1.grid(axis='x', visible=False)
a1.set_title('(a) Cold start: rate vs. distance to initial guess', pad=6)
a1.legend(loc='upper right', fontsize=12.5)
for mode, c, name in (('full', '#1f77b4', 'full pose'), ('pos', '#ff7f0e', 'position only')):
    its = [r['it'] for r in rec[mode] if r['conv']]
    a2.hist(its, bins=np.arange(0, 205, 5), color=c, alpha=0.6, edgecolor='0.2',
            label=f"{name} — median {np.median(its):.0f}")
a2.set_xlabel('Iterations to convergence'); a2.set_ylabel('Number of targets')
a2.set_title('(b) Iterations, converged targets', pad=6); a2.set_xlim(0, 100)
a2.legend(loc='upper right', fontsize=12.5)
fig.subplots_adjust(wspace=0.26); save(fig, 'fig6_ik_statistics')

# ---------------------------------------------------------------- Fig. 7 energy
base = MobileBase(); v = np.linspace(0.1, 1.2, 111)
soils = [('Asphalt (reference)', 0.02, 0.02, '#7f7f7f'),
         ('Deck soil — Niayes', 0.12, 0.06, '#2ca02c'),
         ('Dior soil — groundnut basin', 0.18, 0.10, '#ff7f0e'),
         ('Wet sandy soil', 0.25, 0.15, '#d62728')]
en = json.load(open(f'{RESULTS}/energy_rev.json'))
P_LO, P_HI, P_AUX = en['P_arm_low_W'], en['P_arm_high_W'], en['P_aux_W']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.0, 5.8))
for n, fr, sl, c in soils:
    base.f_r, base.slip_coeff = fr, sl
    a1.plot(v, [base.power_consumption(x_, 10) for x_ in v], color=c, lw=2.6, label=n)
a1.set_xlabel('Travel speed (m/s)'); a1.set_ylabel('Locomotion power (W)')
a1.set_title('(a) Locomotion power, 10 kg payload', pad=6)
a1.set_xlim(0.1, 1.2); a1.set_ylim(0, None); a1.legend(loc='upper left', fontsize=12.5)
base.f_r, base.slip_coeff = 0.18, 0.10
lo = np.array([1500 / (base.power_consumption(x_, 10) + P_AUX + P_HI) for x_ in v])
hi = np.array([1500 / (base.power_consumption(x_, 10) + P_AUX + P_LO) for x_ in v])
a2.fill_between(v, lo, hi, color='#ff7f0e', alpha=0.30,
                label=f'10 kg payload, arm {P_LO:.1f}–{P_HI:.0f} W')
a2.plot(v, hi, color='#ff7f0e', lw=2.2); a2.plot(v, lo, color='#ff7f0e', lw=2.2, ls='--')
a2.axhline(4, ls='--', color='0.25', lw=2.0)
a2.axvspan(en['v_cross_high_W'], en['v_cross_low_W'], color='0.5', alpha=0.15)
a2.annotate('4 h target', xy=(1.17, 4.12), ha='right', fontsize=13.5, color='0.2')
a2.annotate(f"{en['v_cross_high_W']:.2f}–{en['v_cross_low_W']:.2f} m/s",
            xy=(en['v_cross_low_W'] + 0.03, 8.0), fontsize=13, color='0.3')
a2.set_xlabel('Travel speed (m/s)'); a2.set_ylabel('Endurance (h)')
a2.set_title('(b) Endurance on Dior soil, 1.5 kWh battery', pad=6)
a2.set_xlim(0.1, 1.2); a2.legend(loc='upper right', fontsize=12.5)
fig.subplots_adjust(wspace=0.24); save(fig, 'fig7_energy')
print('terminé')
