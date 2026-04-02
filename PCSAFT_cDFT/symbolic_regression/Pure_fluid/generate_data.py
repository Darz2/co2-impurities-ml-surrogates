"""
Generate pure_component_results.csv for each impurity using PC-SAFT + cDFT.
Run from the Pure_fluid directory:
    python generate_data.py
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "../../thermoift/src"))
import thermoift.semi_emperical_correlations as SEC

model = SEC.semi_emperical_correlations()

# component name (feos) → (directory, T_min, T_max_exclusive)
COMPONENTS = {
    "argon":            ("Ar",   85,  149),
    "carbon monoxide":  ("CO",   78,  132),
    "hydrogen":         ("H2",   15,   64),
    "hydrogen sulfide": ("H2S", 190,  374),
    "methane":          ("CH4",  92,  190),
    "nitrogen":         ("N2",   65,  125),
    "oxygen":           ("O2",   80,  153),
}

base_dir = os.path.dirname(os.path.abspath(__file__))

for name, (subdir, T_min, T_max) in COMPONENTS.items():
    print(f"\n{'='*60}")
    print(f"Generating data for {name}  (T = {T_min} … {T_max-1} K)")
    print(f"{'='*60}")

    T_vals = np.arange(T_min, T_max, 1)
    rows = []

    for T_K in T_vals:
        gamma0, rhoL0, rhoV0, Tc, Pc, Psat = model._pure_component_cDFT(name, float(T_K))
        if np.isnan(gamma0):
            continue
        rows.append((T_K, gamma0, rhoL0, rhoV0, Tc, Pc, Psat))

    if not rows:
        print(f"  WARNING: no valid rows for {name}, skipping.")
        continue

    df = pd.DataFrame(rows, columns=["T_K", "gamma", "rhoL", "rhoV", "Tc", "Pc", "Psat"])
    out_path = os.path.join(base_dir, subdir, "pure_component_results.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")

print("\nDone.")
