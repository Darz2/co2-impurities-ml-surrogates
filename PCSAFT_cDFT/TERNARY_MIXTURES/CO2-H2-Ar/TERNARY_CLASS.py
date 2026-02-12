#!/usr/bin/env python

############# Required Packages ############
import numpy as np, feos, ternary, matplotlib.pyplot as plt, os, re, sys, pandas as pd, si_units as si
from molmass import Formula
sys.path.append("..")
import PLOT_SETTINGS as ps

############# Helper functions ############
def print_array2d(feeds, decimals=4):
    ncols = feeds.shape[1]

    header = " | ".join([f"{'z'+str(j+1):>10}" for j in range(ncols)])
    print(f"{'i':>3} | {header}")
    print("-" * (15 + 14*ncols))

    for i, row in enumerate(feeds, start=1):
        row_str = " ".join(f"{val:12.4f}" for val in row)
        print(f"{i:3d} | {row_str}")

def formula_from_inchi(inchi: str) -> str:
    """
    Extract chemical formula from an InChI string.
    Example: 'InChI=1S/CO2/c2-1-3' -> 'CO2'
    """
    try:
        return inchi.split("InChI=1S/")[1].split("/")[0]
    except Exception:
        raise ValueError(f"Cannot parse formula from InChI: {inchi}")
    
def components(parameters):
    """
    Extract component labels (C1, C2, C3) from feos.Parameters
    using the InChI identifiers.

    Returns
    -------
    tuple of str
        (c1, c2, c3) chemical formulas in FEOS component order.
    """
    component_labels = [
        formula_from_inchi(pr.identifier.inchi)
        for pr in parameters.pure_records
    ]

    if len(component_labels) != 3:
        raise ValueError(
            "get_component_labels_from_parameters requires exactly "
            "3 components (ternary system)."
        )

    return tuple(component_labels)

def save_figure(fig, filename):
    """Saves the figure with the predefined resolution."""
    output_dir = os.getcwd()
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path, dpi=1200, bbox_inches='tight')
    fig.savefig(fr"{filename}", dpi=1200, bbox_inches='tight')
    
def save_plot(fig, filename_base, folder="PLOTS"):
    """
    Save a figure as both PNG and PDF using save_figure.
    """
    os.makedirs(folder, exist_ok=True)

    png_path = os.path.join(folder, f"{filename_base}.png")
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")

    save_figure(fig, png_path)
    save_figure(fig, pdf_path)

def latex_formula(formula: str) -> str:
    """
    Convert a chemical formula to LaTeX format.
    Example: 'CO2' -> 'CO$_2$'
             'Ar'  -> 'Ar'
    """
    return re.sub(r"(\d+)", r"$_{\1}$", formula)
   
############# Functions ############
def generate_feeds(CO2, n_points):
    """
    Generate feed compositions for a ternary mixture:
    CO2 (C1) (fixed), H2 (C2) (varied), Ar (C3) (remainder).

    Parameters
    ----------
    CO2 : float
        Mole fraction of CO2 (component 1).
    n_points : int, optional
        Number of composition points (default is 21).

    Returns
    -------
    feeds : np.ndarray
        Array of shape (n_points, 3) with columns [CO2, H2, Ar].
    """
    C2_values = np.linspace(0.0, 1.0 - CO2, n_points)
    feeds = np.array([[CO2, C2, 1.0 - CO2 - C2] for C2 in C2_values])
    # Remove rows where any component is zero (if desired)
    feeds = feeds[~np.any(np.isclose(feeds, 0.0), axis=1)]
    return feeds

def normalize_z(z):
    """Return z as float array normalized to sum to 1."""
    z = np.asarray(z, dtype=float)
    s = z.sum()
    if s == 0:
        raise ValueError("Composition z sums to 0.")
    return z / s

def compute_feed_moles(z, n_total):
    """Convert mole fractions z to mole amounts using n_total."""
    return np.asarray(z, dtype=float) * n_total

def value_in(q, unit):
    """Return the numeric value of q in the given unit."""
    if isinstance(q, (int, float, np.floating)):
        return float(q)
    return np.asarray(q / unit, dtype=float)

def molar_density_mol_m3(rho):
    """Convert molar density to mol/m^3."""
    mol_per_m3 = si.MOL / (si.METER**3)
    try:
        return value_in(rho, mol_per_m3)
    except Exception:
        pass
    try:
        kmol_per_m3 = si.KMOL / (si.METER**3)
        return value_in(rho, kmol_per_m3) * 1000.0
    except Exception:
        pass
    try:
        kmol_per_m3 = (1000.0 * si.MOL) / (si.METER**3)
        return value_in(rho, kmol_per_m3) * 1000.0
    except Exception as e:
        raise TypeError(f"Could not interpret density units") from e
    
def density_kg_m3(rho_molar, z, molar_masses, molar_mass_unit="g/mol"):
    """Convert molar density to mass density (kg/m^3)."""
    z = np.asarray(z, dtype=float)
    M = np.asarray(molar_masses, dtype=float)
    
    if molar_mass_unit.lower() == "g/mol":
        M = M / 1000.0
    elif molar_mass_unit.lower() == "kg/mol":
        pass
    else:
        raise ValueError("molar_mass_unit must be 'g/mol' or 'kg/mol'.")
    
    rho_mol_m3 = molar_density_mol_m3(rho_molar)
    M_mix = np.sum(z * M)
    return rho_mol_m3 * M_mix

########### VLE computation ############
def compute_bubble_curve(eos, T_vals, feed, verbose):
    feed = np.array(feed/si.MOL)
    bubble_pressures = []
    T_bubble = []

    for T_K in T_vals:
        T = T_K * si.KELVIN
        try:
            envelope = feos.PhaseEquilibrium.bubble_point(eos,temperature_or_pressure=T,liquid_molefracs=feed)
            P_bubble = envelope.liquid.pressure()

        except Exception as err:
            if verbose:
                print(f"Bubble calculation failed at T = {T_K} K: {err}")
            continue

        bubble_pressures.append(P_bubble / si.BAR)
        T_bubble.append(T_K)

    return T_bubble, bubble_pressures

def compute_dew_curve(eos, T_vals, feed, verbose):
    feed = np.array(feed/si.MOL)
    T_dew = []
    dew_pressures = []
    
    for T_K in T_vals:
        T = T_K * si.KELVIN
        try:
            envelope = feos.PhaseEquilibrium.dew_point(eos,temperature_or_pressure=T,vapor_molefracs=feed)
            P_dew = envelope.vapor.pressure()

        except Exception as err:
            if verbose:
                print(f"Dew calculation failed at T = {T_K} K: {err}")
            break

        dew_pressures.append(P_dew / si.BAR)
        T_dew.append(T_K)

    return T_dew, dew_pressures

def tp_flash(feos, eos, T, P, feed, molar_masses):
    """
    Run TP flash and return (eq, x, y).
    Raises exception if flash fails.
    """
    eq              = feos.PhaseEquilibrium.tp_flash(eos, T, P, feed)
    liq, vap        = eq.liquid, eq.vapor
    x               = np.array(liq.molefracs, dtype=float)
    y               = np.array(vap.molefracs, dtype=float)
    liquid_density  = density_kg_m3(liq.density, x, molar_masses) 
    vapor_density   = density_kg_m3(vap.density, y, molar_masses)
    
    return eq, x, y, liquid_density, vapor_density

############ Critical point computation ###########
def compute_CT(feos, eos, z, T_guess):
    """Compute critical point temperature for composition z."""
    cp = feos.State.critical_point(eos, z, initial_temperature=T_guess*si.KELVIN)
    return cp.temperature, cp.pressure()

############ Planar interfacial tension computation ###########
def build_planar_interface(eq, critical_temperature, n_grid, l_grid):
    """Create a planar interface object."""
    return feos.PlanarInterface.from_tanh(
        vle=eq,
        n_grid=n_grid,
        l_grid=l_grid * si.ANGSTROM,
        critical_temperature=critical_temperature,)


def solve_interface_properties(interface, si):
    """
    Solve planar interface and return:
    (gamma_mN_m, interfacial_thickness_nm, (E1, E2, E3))
    """
    sol = interface.solve()  # solve once

    surface_tension             = sol.surface_tension
    interfacial_thickness       = sol.interfacial_thickness
    enrichment                  = sol.interfacial_enrichment

    gamma_mN_m                  = float(surface_tension * 1e3 / si.NEWTON * si.METER)
    interfacial_thickness_nm    = float(interfacial_thickness() * 1e9 / si.METER)
    E_1, E_2, E_3               = enrichment()

    return gamma_mN_m, interfacial_thickness_nm, (E_1, E_2, E_3)


######################## Append data #########################
def row_append(z, x, y, liquid_density, vapor_density, critical_temperature,
               gamma_mN_m, interfacial_thickness_nm,  components, enrichment):
    """
    Build a results dictionary for one composition case.
    """
    C_1, C_2, C_3 = components
    E_1, E_2, E_3 = enrichment

    return {
        f"feed_{C_1}": z[0], f"feed_{C_2}": z[1], f"feed_{C_3}": z[2],
        f"x_{C_1}": x[0], f"x_{C_2}": x[1], f"x_{C_3}": x[2],
        f"y_{C_1}": y[0], f"y_{C_2}": y[1], f"y_{C_3}": y[2],
        "liquid_density_kg_m3": float(liquid_density), "vapor_density_kg_m3": float(vapor_density),
        "Tc_K": float(critical_temperature / si.KELVIN),
        "gamma_mN_m": gamma_mN_m,
        "interfacial_thickness_nm": interfacial_thickness_nm,
        f"E_{C_1}": E_1, f"E_{C_2}": E_2, f"E_{C_3}": E_3,
    }

######################## PT PLOT SETTINGS #########################  
def plot_PT(PT_results, feed_key, parameters):
    """
    Plot PT phase diagram (bubble + dew) for a given feed.
    Includes critical point.
    """
    if feed_key not in PT_results:
        raise KeyError(f"{feed_key} not found in results.")

    feed_data = PT_results[feed_key]

    C_1, C_2, C_3 = components(parameters)
    
    # --- Extract curves ---
    T_bub = [pt["T_K"] for pt in feed_data["bubble"]]
    P_bub = [pt["P_bar"] for pt in feed_data["bubble"]]

    T_dew = [pt["T_K"] for pt in feed_data["dew"]]
    P_dew = [pt["P_bar"] for pt in feed_data["dew"]]

    # --- Critical point ---
    Tc = feed_data["TC_K"]
    Pc = feed_data["PC_bar"]
    z  = feed_data["z"]

    # --- Plot ---
    fig, ax = ps.plot_init()

    ax.plot(T_bub, P_bub, label="Bubble curve", linestyle="-", linewidth=ps.linewidth, color="blue")
    ax.plot(T_dew, P_dew, label="Dew curve", linestyle="-", linewidth=ps.linewidth, color="red")
    ax.plot([T_dew[-1], Tc] , [P_dew[-1], Pc], linestyle="-", linewidth=ps.linewidth, color="red")

    # Critical point marker
    ax.scatter(Tc, Pc, marker="o", s=40, zorder=4, label="Critical point")

    ax.set_xlabel("$T$  / [K]", fontsize=ps.label_fontsize)
    ax.set_ylabel("$P$  / [bar]", fontsize=ps.label_fontsize)
    
    ax.set_xlim(left=200)
    
    z_str = ", ".join(f"{comp}={val:.2f}" for comp, val in zip([latex_formula(C_1), latex_formula(C_2), latex_formula(C_3)], z))
    ax.set_title(f"{z_str}", fontsize=ps.label_fontsize)
    ax.minorticks_on()

    ps.style_legend(ax, loc='best',ncol=1,borderaxespad=1.0,frame=False)
    ax.grid(False)
    fig.tight_layout()
    
    return fig,ax

######################## TERNARY PLOT SETTINGS #####################
def plot_ternary_vle(rows,feeds,components,*,permutation="210",
    scale=1.0,
    figsize=(7, 6.5),
    grid_multiple=0.1,
    tick_multiple=0.2,
    tick_format="%.2f",
    tie_lines=True,
    tie_linewidth=0.8,
    tie_color="k",
    tie_alpha=0.6,
    show=True,
    ax=None,
):
    """
    Plot ternary VLE points (bubble x, dew y) + feeds in a ternary diagram.

    Parameters
    ----------
    rows : list[dict]
        Output rows. Must contain keys: x_C1,x_C2,x_C3 and y_C1,y_C2,y_C3.
    feeds : array-like, shape (n,3)
        Feed compositions as [C1, C2, C3] (same order as components).
    parameters : feos.Parameters
        Parameter object containing component information.
    permutation : str, optional
        python-ternary permutation string (default "210").
    scale : float, optional
        Ternary scale (default 1.0).
    figsize : tuple, optional
        Figure size (default (7, 6.5)).
    grid_multiple : float, optional
        Gridline spacing (default 0.1).
    tick_multiple : float, optional
        Tick spacing (default 0.2).
    tick_format : str, optional
        Tick label format (default "%.2f").
    tie_lines : bool, optional
        Draw tie-lines between bubble and dew points (default True).
    show : bool, optional
        Call plt.show() (default True).
    ax : matplotlib axis, optional
        If you want to draw on an existing figure axis, pass it (otherwise new figure).

    Returns
    -------
    fig, tax
        Matplotlib figure and ternary tax object.
    """
    # --- component labels from FEOS parameters (C1,C2,C3 order) ---
    C_1, C_2, C_3 = components

    # Extract points from rows
    x_pts = [tuple(r[c] for c in (f"x_{C_1}", f"x_{C_2}", f"x_{C_3}")) for r in rows]
    y_pts = [tuple(r[c] for c in (f"y_{C_1}", f"y_{C_2}", f"y_{C_3}")) for r in rows]

    # Figure/axes
    if ax is None:
        fig, tax = ternary.figure(scale=scale, permutation=permutation)
        fig.set_size_inches(*figsize)
    else:
        # python-ternary supports "tax" on existing axes via TernaryAxesSubplot
        fig = ax.figure
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale, permutation=permutation)

    tax.boundary(linewidth=2)
    tax.gridlines(multiple=grid_multiple, linewidth=0.7)

    # Axis labels must follow C1,C2,C3 order you pass in `components`.
    tax.left_axis_label(f"{latex_formula(C_1)}/ [mol mol$^{{-1}}$]", offset=0.14, fontsize=20, fontweight="bold")
    tax.right_axis_label(f"{latex_formula(C_2)} / [mol mol$^{{-1}}$]", offset=0.14, fontsize=20, fontweight="bold")
    tax.bottom_axis_label(f"{latex_formula(C_3)} / [mol mol$^{{-1}}$]", offset=0.02, fontsize=20, fontweight="bold")

    tax.ticks(
        axis="lbr",
        multiple=tick_multiple,
        linewidth=1,
        fontsize=11,
        offset=0.02,
        tick_formats=tick_format,
    )

    # Scatter sets
    tax.scatter(x_pts, marker="o", s=120, facecolors="none", edgecolors="blue", label="Bubble")
    tax.scatter(y_pts, marker="o", s=120, facecolors="none", edgecolors="red", label="Dew")
    tax.scatter(feeds, marker="o", s=20, facecolors="orange", edgecolors="orange", label="Feed")

    # Tie-lines
    if tie_lines:
        for xp, yp in zip(x_pts, y_pts):
            tax.line(xp, yp, linewidth=tie_linewidth, color=tie_color, alpha=tie_alpha, permutation=permutation)

    tax.get_axes().set_axis_off()
    tax.clear_matplotlib_ticks()
    tax.legend(loc="upper right", frameon=False, fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, tax

    
    