"""
lora_simulation.py
Simulates LoRa Packet Success Rate (PSR) vs. Distance
for India 866 MHz band using COST-231 Hata model + fading.
"""

import numpy as np
import matplotlib.pyplot as plt

params = dict(
    f_mhz=866.0,
    hb_m=30.0,
    hm_m=2.0,
    Pt_dbm=14.0,
    Gt_db=2.0,
    Gr_db=2.0,
    L_cable_db=0.5,
    bw=125e3,
    NF_db=6.0,
    sf=7,
    cr=1,
    preamble=8,
    snr_th=-7.5
)

def cost231_hata_suburban(d_km, f_mhz, hb_m, hm_m):
    a = (1.1*np.log10(f_mhz)-0.7)*hm_m - (1.56*np.log10(f_mhz)-0.8)
    Lurban = 46.3 + 33.9*np.log10(f_mhz) - 13.82*np.log10(hb_m) - a + (44.9 - 6.55*np.log10(hb_m))*np.log10(d_km)
    return Lurban - 2*(np.log10(f_mhz/28))**2 - 5.4

def per_from_snr(snr_db, snr_th=-7.5, k=1.2):
    return 1 / (1 + np.exp(-(snr_db - snr_th)/k))

def simulate_psr(distance_km, trials=2000, env='suburban', p=params):
    rng = np.random.default_rng(0)
    if env == 'rural': delta, sigma, m = -3.0, 4.0, 2.5
    elif env == 'urban': delta, sigma, m = +6.0, 8.0, 1.3
    else: delta, sigma, m = 0.0, 6.0, 1.8

    PL = cost231_hata_suburban(distance_km, p['f_mhz'], p['hb_m'], p['hm_m']) + delta
    shadow = rng.normal(0, sigma, size=trials)
    fading_db = 10*np.log10(rng.gamma(shape=m, scale=1.0/m, size=trials))
    Pr = p['Pt_dbm'] + p['Gt_db'] + p['Gr_db'] - p['L_cable_db'] - PL + shadow + fading_db
    N = -174 + 10*np.log10(p['bw']) + p['NF_db']
    snr = Pr - N
    psr = 1 - per_from_snr(snr, p['snr_th'])
    return np.mean(psr)

if __name__ == "__main__":
    distances = np.linspace(0.2, 15, 40)
    terrains = ['rural', 'suburban', 'urban']

    plt.figure()
    for env in terrains:
        psr_vals = [simulate_psr(d, env=env) for d in distances]
        plt.plot(distances, psr_vals, label=env.title())

    plt.xlabel("Distance (km)")
    plt.ylabel("Packet Success Rate")
    plt.title("LoRa PSR vs Distance (India 866 MHz, SF7, TX=14 dBm)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("IN_LoRa_PSR_vs_Distance_SF7_TX14.pdf", bbox_inches='tight')
