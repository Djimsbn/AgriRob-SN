"""Bilan énergétique révisé : ratio minorant et encadrement de la puissance du bras."""
import json
import numpy as np
from agrirob_model import MobileBase
from paths import RESULTS

ct = json.load(open(RESULTS / 'sim_ct_Tc5_sub5.json'))
base = MobileBase()
P_AUX, P_HI, E_BATT = 30.0, 60.0, 1500.0
P_LO = ct['P_elec_mean_W']            # travail mécanique / rendement 0,70

loco = {}
for n, fr, sl in (('asphalt', .02, .02), ('deck', .12, .06), ('dior', .18, .10),
                  ('wet_sand', .25, .15)):
    base.f_r, base.slip_coeff = fr, sl
    loco[n] = base.power_consumption(0.5, 10)

base.f_r, base.slip_coeff = 0.18, 0.10
end = lambda v, pa: E_BATT / (base.power_consumption(v, 10) + P_AUX + pa)
vs = np.linspace(0.1, 1.2, 11001)
cross = lambda pa: float(np.interp(-4.0, -np.array([end(v, pa) for v in vs]), vs))

P_op_lo = base.power_consumption(0.5, 10) + P_AUX + P_LO
P_op_hi = base.power_consumption(0.5, 10) + P_AUX + P_HI
W = (base.M + 10) * 9.81
out = {
    'loco_W': loco,
    'ratio_wet_asphalt': loco['wet_sand'] / loco['asphalt'],
    'ratio_dior_asphalt': loco['dior'] / loco['asphalt'],
    'asphalt_sink_share': (base.k_s * 0.02 * 0.04) /
                          (W * 0.02 + base.k_s * 0.02 * 0.04),
    'E_mech_J': ct['E_mech_J'], 'P_mech_mean_W': ct['P_mech_mean_W'],
    'P_arm_low_W': P_LO, 'P_arm_high_W': P_HI, 'P_aux_W': P_AUX,
    'endurance_h': {s: [end(float(s), P_HI), end(float(s), P_LO)]
                    for s in ('0.3', '0.5', '1.2')},
    'v_cross_low_W': cross(P_LO), 'v_cross_high_W': cross(P_HI),
    'P_op_W': [P_op_lo, P_op_hi],
    'solar_extra_h': [E_BATT / P_op_hi, E_BATT / P_op_lo],
}
json.dump(out, open(RESULTS / 'energy_rev.json', 'w'), indent=2)
for k, v in out.items():
    print(f'{k:22} {v}')
