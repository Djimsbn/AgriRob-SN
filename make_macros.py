"""
Transforme les fichiers de résultats en macros LaTeX et tableaux, pour que
les manuscrits anglais et français citent exactement les mêmes chiffres.
Les nombres sont écrits avec un point décimal et doivent être utilisés dans
\\num{...} (siunitx convertit selon la langue du document).
"""
import json, glob, os
import numpy as np

from paths import RESULTS, LATEX
D = str(RESULTS)
PAY = [('0', 'Zero'), ('0.5', 'Half'), ('1', 'One'), ('2', 'Two')]
LAWS = [('ct', 'CT'), ('pdg', 'PDG'), ('pd', 'PD')]

def sim(law, m, sub=5):
    tag = f"{law}_Tc5_sub{sub}" + ("" if m == '0' else f"_m{m}")
    f = f"{D}/sim_{tag}.json"
    return json.load(open(f)) if os.path.exists(f) else None

M = {}
def put(name, val, fmt):
    M[name] = fmt.format(val)

# ------------------------------------------------------------- commande
for m, mk in PAY:
    for law, lk in LAWS:
        s = sim(law, m)
        if s is None:
            raise SystemExit(f"résultat manquant : {law} {m} kg")
        put(f"RMS{lk}{mk}", s['rms_mm'], "{:.2f}")
        put(f"MAX{lk}{mk}", s['max_mm'], "{:.2f}")
        put(f"E{lk}{mk}", s['E_mech_J'], "{:.2f}")
        put(f"TAU{lk}{mk}", max(s['peak_tau_per_joint']), "{:.2f}")

ct10 = sim('ct', '0', 10); ct5 = sim('ct', '0', 5)
put("RMSCTZeroSubFive", ct5['rms_mm'], "{:.3f}")
put("RMSCTZeroSubTen", ct10['rms_mm'], "{:.3f}")
put("RMSCTZeroOrig", 4.32, "{:.2f}")
c1_10 = sim('ct', '1', 10)
if c1_10:
    put("RMSCTOneSubTen", c1_10['rms_mm'], "{:.2f}")
    put("ConvCTOnePct", 100 * abs(sim('ct', '1')['rms_mm'] - c1_10['rms_mm'])
        / c1_10['rms_mm'], "{:.1f}")
p10 = sim('pdg', '0', 10)
if p10:
    put("RMSPDGZeroSubTen", p10['rms_mm'], "{:.2f}")
    put("ConvPDGZeroPct", 100 * abs(sim('pdg', '0')['rms_mm'] - p10['rms_mm'])
        / p10['rms_mm'], "{:.1f}")

sct = sim('ct', '0'); spdg = sim('pdg', '0'); spd = sim('pd', '0')
put("GainPDGoverPD", spd['rms_mm'] / spdg['rms_mm'], "{:.0f}")
sig = [sct['signed_peak_tau'][j] for j in range(6)]
put("TauShoulder", abs(sig[1]), "{:.2f}")
put("TauElbow", abs(sig[2]), "{:.2f}")
put("TauWristMax", max(abs(x) for x in sig[3:]), "{:.2f}")
put("CTOneVsPDGOne", sim('pdg', '1')['rms_mm'] / sim('ct', '1')['rms_mm'], "{:.2f}")

# borne d'erreur d'intégration (h = 1 ms), mesurée sur le cas au modèle exact
ie = sim('ct', '0')['rms_mm']
unc = min(sim(l, m)['rms_mm'] for l in ('ct', 'pdg') for m in ('0.5', '1', '2'))
put("IntErrPctPayload", 100 * ie / unc, "{:.0f}")
put("IntErrPctPDG", 100 * ie / sim('pdg', '0')['rms_mm'], "{:.0f}")
# raideur effective du couple calculé à l'épaule le long du mouvement (1 kg)
from agrirob_model import mass_matrix
d = np.load(f"{D}/sim_ct_Tc5_sub5_m1.npz"); q0 = np.array([0, -0.3, 0.5, 0, 0.5, 0.])
qq = q0 + np.concatenate([[np.zeros(6)], np.cumsum((d['v'][1:] + d['v'][:-1]) / 2
                                                    * np.diff(d['t'])[:, None], axis=0)])
m0 = mass_matrix(q0)[1, 1]
put("StiffGainPct", 100 * (max(mass_matrix(x)[1, 1] for x in qq[::20]) / m0 - 1), "{:.0f}")
ratios = [sim('ct', m)['rms_mm'] / sim('pdg', m)['rms_mm'] for m in ('0.5', '1', '2')]
put("PDGAdvLo", 100 * (min(ratios) - 1), "{:.0f}")
put("PDGAdvHi", 100 * (max(ratios) - 1), "{:.0f}")
es = [max(sim(l, m)['E_mech_J'] for l in ('ct','pdg','pd')) /
      min(sim(l, m)['E_mech_J'] for l in ('ct','pdg','pd')) for m in ('0','0.5','1','2')]
put("EnergySpreadPct", 100 * (max(es) - 1), "{:.0f}")

# ------------------------------------------------------------- énergie
en = json.load(open(f"{D}/energy_rev.json"))
for k, n in (('asphalt', 'Asph'), ('deck', 'Deck'), ('dior', 'Dior'), ('wet_sand', 'Wet')):
    put(f"P{n}", en['loco_W'][k], "{:.0f}")
put("RatioWet", en['ratio_wet_asphalt'], "{:.2f}")
put("RatioDior", en['ratio_dior_asphalt'], "{:.2f}")
put("SinkShare", 100 * en['asphalt_sink_share'], "{:.1f}")
put("EArm", en['E_mech_J'], "{:.2f}")
put("PArmMech", en['P_mech_mean_W'], "{:.2f}")
put("PArmLo", en['P_arm_low_W'], "{:.1f}")
put("PArmHi", en['P_arm_high_W'], "{:.0f}")
put("PAux", en['P_aux_W'], "{:.0f}")
for s, n in (('0.3', 'Slow'), ('0.5', 'Mid'), ('1.2', 'Fast')):
    put(f"End{n}Lo", en['endurance_h'][s][0], "{:.1f}")
    put(f"End{n}Hi", en['endurance_h'][s][1], "{:.1f}")
put("VcrossLo", en['v_cross_high_W'], "{:.2f}")
put("VcrossHi", en['v_cross_low_W'], "{:.2f}")
put("POpLo", en['P_op_W'][0], "{:.0f}")
put("POpHi", en['P_op_W'][1], "{:.0f}")
put("SolarLo", en['solar_extra_h'][0], "{:.1f}")
put("SolarHi", en['solar_extra_h'][1], "{:.1f}")

# ------------------------------------------------------------- MGI
ik = json.load(open(f"{D}/ik_statistics.json"))
E = ik['E']
put("IKN", E['n_targets'], "{:d}")
for mode, n in (('full_pose', 'Full'), ('position_only', 'Pos')):
    s = E[mode]
    put(f"IK{n}Conv", s['converged'], "{:d}")
    put(f"IK{n}Rate", 100 * s['rate'], "{:.1f}")
    put(f"IK{n}ItMean", s['it_mean'], "{:.1f}")
    put(f"IK{n}ItMed", s['it_median'], "{:.0f}")
    put(f"IK{n}ItPn", s['it_p95'], "{:.0f}")
wb = E['full_pose']['by_manipulability'][-1]
put("IKFullRateHighW", 100 * wb['rate'], "{:.1f}")
dist = json.load(open(f"{D}/ik_distance.json"))
put("IKFullQone", 100 * dist['full'][0][3], "{:.0f}")
put("IKFullQfour", 100 * dist['full'][3][3], "{:.0f}")
put("IKPosQone", 100 * dist['pos'][0][3], "{:.0f}")
put("IKPosQfour", 100 * dist['pos'][3][3], "{:.0f}")
ws = ik['ik_warmstart_tracking']
put("IKWarmConv", ws['converged'], "{:d}")
put("IKWarmN", ws['steps'], "{:d}")

with open(LATEX / "results_macros.tex", 'w') as f:
    f.write("% Généré automatiquement par make_macros.py — ne pas éditer.\n")
    for k, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\r{k}}}{{{v}}}\n")

# ------------------------------------------------------------- tableaux
def table(lang):
    H = {'en': ('$m_p$ (kg)', 'Law', 'RMS (mm)', 'Peak (mm)',
                'Energy (J)', 'Peak $\\tau$ (N\\,m)',
                {'ct': 'CT', 'pdg': 'PD+G', 'pd': 'PD'}),
         'fr': ('Charge (kg)', 'Loi', 'RMS (mm)', 'Max (mm)',
                'Énergie articulaire (J)', 'Couple max (N\\,m)',
                {'ct': 'Couple calculé', 'pdg': 'PD + gravité', 'pd': 'PD seul'})}[lang]
    rows = []
    for i, (m, mk) in enumerate(PAY):
        for j, (law, lk) in enumerate(LAWS):
            s = sim(law, m)
            pay = f"\\multirow{{3}}{{*}}{{\\num{{{m}}}}}" if j == 0 else ""
            rms = (f"$<$\\num{{0.02}}\\textsuperscript{{a}}"
                   if (law == 'ct' and m == '0') else f"\\num{{{s['rms_mm']:.2f}}}")
            mx = (f"$<$\\num{{0.04}}\\textsuperscript{{a}}"
                  if (law == 'ct' and m == '0') else f"\\num{{{s['max_mm']:.2f}}}")
            rows.append(f"    {pay} & {H[6][law]} & {rms} & {mx} & "
                        f"\\num{{{s['E_mech_J']:.2f}}} & "
                        f"\\num{{{max(s['peak_tau_per_joint']):.2f}}} \\\\")
        if i < len(PAY) - 1:
            rows.append("    \\midrule")
    body = "\n".join(rows)
    return (f"  \\begin{{tabular}}{{@{{}}clrrrr@{{}}}}\n    \\toprule\n"
            f"    {H[0]} & {H[1]} & {H[2]} & {H[3]} & {H[4]} & {H[5]} \\\\\n"
            f"    \\midrule\n{body}\n    \\bottomrule\n  \\end{{tabular}}\n")

for lang in ('en', 'fr'):
    open(LATEX / f"tab_control_{lang}.tex", 'w').write(table(lang))
print(f"{len(M)} macros écrites ; tableaux EN/FR écrits")
