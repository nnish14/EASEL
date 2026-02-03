"""
formative_analysis.py
Robust formative usability analysis for EASEL dataset.
Automatically adapts to missing columns (e.g., no T2_time).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Load CSV ===
df = pd.read_csv("EASEL_UAV_Formative_Study_Results.csv")

# === Detect available timed tasks ===
task_map = {
    "T1_time": "Alert Triage",
    "T2_time": "Geofence Breach",
    "T3_time": "Track & Predict"
}

available_tasks = [(col, label) for col, label in task_map.items() if col in df.columns]

print(f"[INFO] Detected tasks: {[label for _, label in available_tasks]}")

# === Summaries ===
summary_rows = []

for col, label in available_tasks:
    times = df[col].dropna()
    med = np.median(times)
    q1, q3 = np.percentile(times, [25, 75])
    
    # Detect success if it exists
    success_col = f"{col.split('_')[0]}_success"
    if success_col in df.columns:
        success_rate = (df[success_col] == "Y").mean() * 100
        success_str = f"{success_rate:.0f}%"
    else:
        success_str = "N/A"
    
    summary_rows.append([
        label,
        round(med, 2),
        f"{q1:.1f}-{q3:.1f}",
        success_str
    ])

# === Export Summary Table ===
summary_df = pd.DataFrame(summary_rows, columns=["Task", "Median (s)", "IQR (s)", "Success Rate"])
summary_df.to_csv("Formative_Summary.csv", index=False)
print("[INFO] Exported Formative_Summary.csv")
print(summary_df)

# === Boxplot (only for available timed tasks) ===
if available_tasks:
    plt.figure(figsize=(8, 5))
    plt.boxplot([df[col] for col, _ in available_tasks], labels=[label for _, label in available_tasks], showmeans=True)
    plt.ylabel("Time (s)")
    plt.title(f"Task Completion Time Distribution (N={len(df)})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("task_time_boxplot.png", dpi=300, bbox_inches='tight')
    print("[INFO] Saved task_time_boxplot.png")
else:
    print("[WARN] No timed tasks found — skipping boxplot.")

