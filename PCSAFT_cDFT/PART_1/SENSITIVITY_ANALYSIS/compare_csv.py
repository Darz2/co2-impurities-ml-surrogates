import sys, pandas as pd, numpy as np
from pathlib import Path

V4 = Path(sys.argv[1]); V5 = Path(sys.argv[2])
files = sorted(p.name for p in V4.glob("*.csv"))
ok = True
for name in files:
    a = pd.read_csv(V4 / name)
    b = pd.read_csv(V5 / name)
    if list(a.columns) != list(b.columns):
        print(f"[{name}] column mismatch:\n  V4: {list(a.columns)}\n  V5: {list(b.columns)}"); ok=False; continue
    # Sort by all numeric-looking keys so row order doesn't matter.
    sort_cols = [c for c in ("T_K","P_bar","T","P") if c in a.columns] or list(a.columns)
    a = a.sort_values(sort_cols).reset_index(drop=True)
    b = b.sort_values(sort_cols).reset_index(drop=True)
    if a.shape != b.shape:
        print(f"[{name}] shape mismatch V4={a.shape} V5={b.shape}"); ok=False; continue
    num = a.select_dtypes(include=[np.number]).columns
    diff = (a[num] - b[num]).abs()
    rel  = diff / a[num].abs().replace(0, np.nan)
    print(f"[{name}] max |Δ| = {diff.max().max():.3e}   max relΔ = {rel.max().max():.3e}")
    if not np.allclose(a[num].values, b[num].values, rtol=1e-6, atol=1e-9, equal_nan=True):
        ok=False
        bad = (~np.isclose(a[num], b[num], rtol=1e-6, atol=1e-9, equal_nan=True)).any(axis=1)
        print(a[bad].head().to_string())
print("MATCH" if ok else "MISMATCH")
