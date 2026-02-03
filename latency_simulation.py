"""
latency_simulation.py
Simulates system latency CCDF for multi-UAV LoRa telemetry.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

params = dict(
    payload_B=40, sf=7, bw=125e3, cr=1, preamble=8,
    mac_jitter_s=0.03, serial_s=0.003, backend_med_s=0.006,
    ui_base_s=0.004, ui_slope_s=0.00005, ui_jitter_s=0.001
)

def lora_airtime_bytes(payload, sf, bw, cr, preamble):
    Ts = (2**sf)/bw
    payloadSymbNb = 8 + max(np.ceil((8*payload - 4*sf + 28 + 16 - 20) / (4*(sf))) * (cr + 4), 0)
    return (preamble + 4.25)*Ts + payloadSymbNb*Ts

def simulate_latency(N=10, duration_s=180, hz=1.0, p=params):
    rng = np.random.default_rng(1)
    ToA = lora_airtime_bytes(p['payload_B'], p['sf'], p['bw'], p['cr'], p['preamble'])
    times = []
    for _ in range(int(duration_s*hz)):
        for i in range(N):
            mac = rng.uniform(0, p['mac_jitter_s'])
            backend = rng.lognormal(mean=np.log(p['backend_med_s']), sigma=0.25)
            ui = p['ui_base_s'] + p['ui_slope_s']*N + rng.normal(0, p['ui_jitter_s'])
            total = ToA + mac + p['serial_s'] + backend + max(ui, 0)
            times.append(total)
    return np.array(times)

if __name__ == "__main__":
    Ns = [1, 10, 50]
    df = pd.DataFrame(columns=["UAV Count", "Median (ms)", "P95 (ms)", "P99 (ms)"])
    plt.figure()

    for N in Ns:
        arr_ms = simulate_latency(N) * 1000
        arr_sorted = np.sort(arr_ms)
        ccdf = 1.0 - np.arange(1, len(arr_sorted)+1)/len(arr_sorted)
        plt.plot(arr_sorted, ccdf, label=f"N={N}")
        df.loc[len(df)] = [N, np.median(arr_ms), np.percentile(arr_ms,95), np.percentile(arr_ms,99)]

    plt.xlabel("Latency (ms)")
    plt.ylabel("CCDF")
    plt.title("Latency CCDF (India 866 MHz, SF7, TX=14 dBm)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.savefig("IN_Latency_CCDF_SF7_TX14.pdf", bbox_inches='tight')
    df.to_csv("IN_Latency_Budget_Table_SF7_TX14.csv", index=False)
