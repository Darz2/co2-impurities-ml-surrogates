"""
Post-processing: render the γ-vs-(T,P) contour with linear γ(P) closure on
each isotherm, so the contour fill reaches the bubble and dew curves.

Reads interfacial + envelope CSVs from a run directory, fits γ ≈ a + b·P
for each isotherm (needs cDFT_NP ≥ 2 P points per T), evaluates at the
bubble and dew pressures of that T, and feeds the augmented (T, P, γ) set
to scipy.interpolate.griddata cubic — same machinery as
PlottingEngine.plot_interfacial_tension_map, just with the boundary
extrapolation pre-applied.

Usage:
    python plot_gamma_closure.py LOCAL_RUN/0
    python plot_gamma_closure.py LOCAL_RUN/0 --out LOCAL_RUN/0/CSV/PNG/Gamma_cDFT_closed_feed_1.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion

# Reuse the project's plot styling so the new figure matches the rest
THERMOIFT_SRC = Path(__file__).parent / "../thermoift/src"
sys.path.insert(0, str(THERMOIFT_SRC.resolve()))
import thermoift.PLOT_SETTINGS as ps  # noqa: E402


def fit_linear_closure(ift_df, env_df):
    """For each isotherm, fit γ(P)=a+b·P and evaluate at P_bub/P_dew.

    Boundary anchors are placed at EVERY envelope T (not just cDFT T) by
    using the nearest cDFT isotherm's linear fit. This densifies the
    boundary so the contour fill is continuous along the saturation curves.

    Returns (T_aug, P_aug, gamma_aug) — original interior points plus
    dense boundary points along both bubble and dew curves.
    """
    bubble = env_df[env_df["curve"] == "bubble"].sort_values("T_K").reset_index(drop=True)
    dew    = env_df[env_df["curve"] == "dew"   ].sort_values("T_K").reset_index(drop=True)

    # Per-isotherm linear fits γ(P) = a + b·P
    fits = {}  # T -> (a, b)
    interior_T, interior_P, interior_g = [], [], []
    for T, sub in ift_df.dropna(subset=["gamma"]).groupby("temperature"):
        Ps = sub["pressure"].to_numpy()
        gs = sub["gamma"].to_numpy()
        interior_T.extend([T] * len(Ps))
        interior_P.extend(Ps.tolist())
        interior_g.extend(gs.tolist())
        if len(Ps) >= 2:
            b, a = np.polyfit(Ps, gs, 1)
            fits[float(T)] = (float(a), float(b))

    if not fits:
        return np.array(interior_T), np.array(interior_P), np.array(interior_g)

    fit_T = np.array(sorted(fits))
    fit_a = np.array([fits[T][0] for T in fit_T])
    fit_b = np.array([fits[T][1] for T in fit_T])

    def gamma_at(T, P):
        """Interpolate (a, b) to T from the cDFT isotherm fits, evaluate at P."""
        a = float(np.interp(T, fit_T, fit_a))
        b = float(np.interp(T, fit_T, fit_b))
        return max(0.0, a + b * P)

    # Dense boundary anchors at every envelope T
    bnd_T, bnd_P, bnd_g = [], [], []
    for _, row in bubble.iterrows():
        T = float(row["T_K"]); P = float(row["P_bar"])
        bnd_T.append(T); bnd_P.append(P); bnd_g.append(gamma_at(T, P))
    for _, row in dew.iterrows():
        T = float(row["T_K"]); P = float(row["P_bar"])
        bnd_T.append(T); bnd_P.append(P); bnd_g.append(gamma_at(T, P))

    T_aug = np.array(interior_T + bnd_T)
    P_aug = np.array(interior_P + bnd_P)
    g_aug = np.array(interior_g + bnd_g)
    return T_aug, P_aug, g_aug


def plot_contour(T_g, P_g, g, env_df, Tc, Pc, title, out_path):
    fig, ax = ps.plot_init()

    # Anchor γ=0 at the critical point so the fill closes there
    T_aug = np.append(T_g, Tc)
    P_aug = np.append(P_g, Pc)
    g_aug = np.append(g,   0.0)

    # Finer grid (300×300) — visible cell width was the source of leftover
    # slivers at 100×100. Switch to LINEAR interpolation so the envelope
    # anchors define the fill exactly without cubic overshoot.
    grid_size = 300
    T_grid = np.linspace(T_g.min(), max(T_g.max(), Tc), grid_size)
    P_grid = np.linspace(P_g.min(), max(P_g.max(), Pc), grid_size)
    T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

    gamma_mesh = griddata((T_aug, P_aug), g_aug, (T_mesh, P_mesh),
                          method="linear", fill_value=np.nan)
    nan = np.isnan(gamma_mesh)
    if nan.any():
        fill = griddata((T_aug, P_aug), g_aug, (T_mesh, P_mesh), method="nearest")
        gamma_mesh[nan] = fill[nan]

    # Build the envelope polygon and mask outside it
    bubble = env_df[env_df["curve"] == "bubble"].sort_values("T_K")
    dew    = env_df[env_df["curve"] == "dew"   ].sort_values("T_K")
    env_T = list(bubble["T_K"]) + list(dew["T_K"].iloc[::-1])
    env_P = list(bubble["P_bar"]) + list(dew["P_bar"].iloc[::-1])
    poly  = MplPath(np.column_stack([env_T, env_P]))
    inside = poly.contains_points(np.column_stack([T_mesh.ravel(), P_mesh.ravel()])).reshape(T_mesh.shape)

    gm = np.ma.masked_where(~inside, gamma_mesh)
    g_min, g_max = float(np.nanmin(gm)), float(np.nanmax(gm))

    levels = np.linspace(g_min, g_max, 35)
    ax.contourf(T_mesh, P_mesh, gm, levels=levels, cmap=plt.cm.Blues, extend="both")

    # Iso-γ lines (avoid the very-near-envelope band)
    inside_eroded = binary_erosion(inside, iterations=2)
    gm2 = np.ma.masked_where(~inside_eroded, gamma_mesh)
    line_lvls = [2 * n + 1 for n in range(10) if g_min <= 2 * n + 1 <= g_max]
    cs = ax.contour(T_mesh, P_mesh, gm2, levels=line_lvls,
                    colors="black", linewidths=ps.linewidth / 2, alpha=0.6)
    ax.clabel(cs, levels=line_lvls, inline=True, fontsize=ps.label_fontsize / 2,
              fmt="%.0f", inline_spacing=15, rightside_up=True)

    # Saturation curves and critical point on top
    ax.plot(bubble["T_K"], bubble["P_bar"], color="blue", lw=ps.linewidth, label="Bubble curve")
    ax.plot(dew["T_K"],    dew["P_bar"],    color="red",  lw=ps.linewidth, label="Dew curve")
    ax.scatter([Tc], [Pc], color="grey", edgecolor="black", s=40, zorder=5, label="Critical point")

    # Colour bar
    norm = plt.Normalize(vmin=g_min, vmax=g_max)
    sm   = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
    cb   = plt.colorbar(sm, ax=ax, extend="both")
    cb.set_label(r"$\gamma$ $/$ $[\mathrm{mN\,m^{-1}}]$", fontsize=ps.label_fontsize)

    ax.set_xlabel(r"$T$ $/$ $[\mathrm{K}]$",   fontsize=ps.label_fontsize)
    ax.set_ylabel(r"$P$ $/$ $[\mathrm{bar}]$", fontsize=ps.label_fontsize)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=ps.label_fontsize * 0.8)
    ps.apply_axis_style(ax)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Run directory containing CSV/InterfacialProperties/ and CSV/PhaseEnvelope/")
    p.add_argument("--feed", default="feed_1")
    p.add_argument("--out",  default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    run = Path(args.run_dir)
    ift = pd.read_csv(run / f"CSV/InterfacialProperties/{args.feed}_interfacial_results.csv")
    env = pd.read_csv(run / f"CSV/PhaseEnvelope/{args.feed}_envelope.csv")

    Tc = env["T_K"].max()
    Pc = float(env.loc[env["T_K"] == Tc, "P_bar"].iloc[0])

    pts_per_T = ift.groupby("temperature").size()
    multi     = (pts_per_T >= 2).sum()
    print(f"Loaded {len(ift)} cDFT points across {len(pts_per_T)} isotherms")
    print(f"  isotherms with >=2 P points (closure-fittable): {multi}")
    print(f"  Tc={Tc:.2f} K, Pc={Pc:.2f} bar")

    if multi == 0:
        sys.exit("ERROR: no isotherm has ≥2 P points — cannot do linear closure. "
                 "Re-run with cDFT_NP>=2.")

    T_g, P_g, g = fit_linear_closure(ift, env)

    z = ift.iloc[0]
    comp_str = ", ".join(
        f"{c.replace('z_','').upper()}={z[c]:.2f}"
        for c in ift.columns if c.startswith("z_") and z[c] > 1e-3
    ).replace("CARBON DIOXIDE", "CO_2").replace("CARBON MONOXIDE", "CO") \
     .replace("HYDROGEN SULFIDE", "H_2S").replace("HYDROGEN", "H_2") \
     .replace("NITROGEN", "N_2").replace("OXYGEN", "O_2") \
     .replace("METHANE", "CH_4").replace("ARGON", "Ar")
    title = args.title or f"${comp_str}$"

    out = args.out or (run / "CSV" / "PNG" / f"Gamma_cDFT_closed_{args.feed}.png")
    plot_contour(T_g, P_g, g, env, Tc, Pc, title, out)


if __name__ == "__main__":
    main()
