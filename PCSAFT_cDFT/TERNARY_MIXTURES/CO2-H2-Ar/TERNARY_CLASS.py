#!/usr/bin/env python

############# Required Packages ############
import numpy as np, feos, ternary, matplotlib.pyplot as plt, os, re, sys, pandas as pd, si_units as si
from molmass import Formula
sys.path.append("..")
import PLOT_SETTINGS as ps

# For colormaps
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from matplotlib.path import Path
from scipy.ndimage import binary_erosion

############# Helper functions ############
def print_array2d(feeds, decimals=4):
    ncols = feeds.shape[1]

    header = " | ".join([f"{'z'+str(j+1):>10}" for j in range(ncols)])
    print(f"{'i':>3} | {header}")
    print("-" * (15 + 14*ncols))

    for i, row in enumerate(feeds, start=1):
        row_str = " ".join(f"{val:12.4f}" for val in row)
        print(f"{i:3d} | {row_str}")

def formula_from_inchi(inchi) -> str:
    """
    Extract chemical formula from an InChI string.
    Accepts InChI=1S/... and InChI=1/...
    """
    if inchi is None:
        raise ValueError("InChI is None")

    s = str(inchi).strip()

    if not s.startswith("InChI="):
        raise ValueError(f"Not an InChI string: {s}")

    # InChI format: InChI=1S/<formula>/<rest...>  OR  InChI=1/<formula>/<rest...>
    parts = s.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Cannot parse formula from InChI: {s}")

    return parts[1]
    
def components(parameters):
    """
    Extract component labels (C1, C2, C3) from feos.Parameters
    using the InChI identifiers.

    Returns
    -------
    tuple of str
        (c1, c2, c3) chemical formulas in FEOS component order.
    """
    component_labels = []
    
    for pr in parameters.pure_records:
        identifier = pr.identifier
        if hasattr(identifier, '__dict__'):
            inchi_str = identifier.__dict__['inchi']
        else:
            # Parse from string representation
            id_str = str(identifier)
            # Extract inchi from "Identifier(cas=..., inchi=InChI=1S/...)"
            import re
            match = re.search(r'inchi=(InChI=[^,)]+)', id_str)
            if match:
                inchi_str = match.group(1)
            else:
                raise ValueError(f"Cannot extract InChI from: {id_str}")
        
        component_labels.append(formula_from_inchi(inchi_str))

    if len(component_labels) != 3:
        raise ValueError(
            "get_component_labels_from_parameters requires exactly "
            "3 components (ternary system)."
        )

    return tuple(component_labels)

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

########### VLE computations ############
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

######################## Append and save data #########################
def row_append(x, y, liquid_density, vapor_density, critical_temperature,
               gamma_mN_m, interfacial_thickness_nm,  components, enrichment):
    """
    Build a results dictionary for one composition case.
    """
    C_1, C_2, C_3 = components
    E_1, E_2, E_3 = enrichment

    return {
        f"x_{C_1}": x[0], f"x_{C_2}": x[1], f"x_{C_3}": x[2],
        f"y_{C_1}": y[0], f"y_{C_2}": y[1], f"y_{C_3}": y[2],
        "liquid_density_kg_m3": float(liquid_density), "vapor_density_kg_m3": float(vapor_density),
        "Tc_K": float(critical_temperature / si.KELVIN),
        "gamma_mN_m": gamma_mN_m,
        "interfacial_thickness_nm": interfacial_thickness_nm,
        f"E_{C_1}": E_1, f"E_{C_2}": E_2, f"E_{C_3}": E_3,
    }

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

def VLE_DFT_summary(VLE_DFT):
    """
    Print a summary of VLE-DFT data structure.
    
    Parameters
    ----------
    VLE_DFT : dict
        Dictionary containing VLE data with structure:
        {feed_key: {
            "interfacial_data": {T_K: [data_dicts, ...]},
            "phase_envelope": {
                "bubble_curve": [...],
                "dew_curve": [...],
                "isothermal_lines": [...]
            }
        }}
    
    Returns
    -------
    dict
        Summary statistics for each feed:
        {feed_key: {
            "n_bubble_points": int,
            "n_dew_points": int,
            "n_isotherms": int,
            "total_interfacial_points": int
        }}
    """
    print("Available feeds:", list(VLE_DFT.keys()))
    
    summary = {}
    
    for feed_key in VLE_DFT:
        n_bubble = len(VLE_DFT[feed_key]["phase_envelope"]["bubble_curve"])
        n_dew = len(VLE_DFT[feed_key]["phase_envelope"]["dew_curve"])
        n_temps = len(VLE_DFT[feed_key]["interfacial_data"])
        
        # Count total interfacial data points
        total_interfacial = sum(
            len(data_list) 
            for data_list in VLE_DFT[feed_key]["interfacial_data"].values()
        )
        
        print(f"{feed_key}: {n_bubble} bubble pts, {n_dew} dew pts, {n_temps} isotherms, {total_interfacial} interfacial pts")
        
        summary[feed_key] = {
            "n_bubble_points": n_bubble,
            "n_dew_points": n_dew,
            "n_isotherms": n_temps,
            "total_interfacial_points": total_interfacial
        }
    
    return summary

def VLE_DFT_to_csv(VLE_DFT, verbose: bool = False):
    """Save VLE-DFT interfacial and phase envelope data to CSV files."""
    saved_files = {}
    
    for feed_key in VLE_DFT:
        # Save interfacial data
        all_data = []
        for T_K, data_list in VLE_DFT[feed_key]["interfacial_data"].items():
            all_data.extend(data_list)
        
        df = pd.DataFrame(all_data)
        csv_filename = f"{feed_key}_interfacial_results.csv"
        df.to_csv(csv_filename, index=False)
        
        if verbose:
            print(f"Saved {len(df)} rows to {csv_filename}")
        
        # Save phase envelope (isothermal lines only)
        envelope_filename = f"{feed_key}_phase_envelope.csv"
        pd.DataFrame(VLE_DFT[feed_key]["phase_envelope"]["isothermal_lines"]).to_csv(
            envelope_filename, index=False
        )
        
        saved_files[feed_key] = {
            "interfacial": csv_filename,
            "envelope": envelope_filename
        }
    
    return saved_files
   
######################## PT PLOT SETTINGS #########################  
def plot_PT(PT_results, feed_key, parameters):
    """
    Plot PT phase diagram (bubble + dew) for a given feed.
    Includes critical point.
    """
    if feed_key not in PT_results:
        raise KeyError(f"{feed_key} not found in PT_results dict.")

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
    ax.plot(Tc, Pc, 'o', markersize=4, zorder=4, 
               markerfacecolor="grey", markeredgewidth=1, markeredgecolor="black", 
               label="Critical point")

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

def plot_gamma_colormap(PT_results, VLE_DFT, feed_key, parameters):
    """
    Create a colormap plot of interfacial tension in PT space.
    
    Parameters
    ----------
    VLE_DFT : dict
        Full VLE dictionary containing phase envelope and interfacial data:
        {"T_K": list, "P_bar": list, "gamma_mN_m": list, "stats": dict}
    feed_key : str
        Feed composition identifier
    parameters : dict
        Plot parameters (colors, fonts, etc.)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the colormap plot
    """
    if feed_key not in PT_results:
        raise KeyError(f"{feed_key} not found in PT_results dict.")
    
    feed_data = PT_results[feed_key]
    
    C_1, C_2, C_3 = components(parameters)
        
    # Extract phase envelope
    bubble_data = VLE_DFT[feed_key]["phase_envelope"]["bubble_curve"]
    dew_data    = VLE_DFT[feed_key]["phase_envelope"]["dew_curve"]
    T_bubble    = [pt["T_K"] for pt in bubble_data]
    P_bubble    = [pt["P_bar"] for pt in bubble_data]
    T_dew       = [pt["T_K"] for pt in dew_data]
    P_dew       = [pt["P_bar"] for pt in dew_data]
    
    Tc          = feed_data["TC_K"]
    Pc          = feed_data["PC_bar"] 
    z           = feed_data["z"]
    
    # Extract interfacial tension data
    T_gamma, P_gamma, gamma_data = [], [], []
    for T_K, data_list in VLE_DFT[feed_key]["interfacial_data"].items():
        for point in data_list:
            if "gamma_mN_m" in point and not np.isnan(point["gamma_mN_m"]):
                T_gamma.append(point["T_K"])
                P_gamma.append(point["P_bar"])
                gamma_data.append(point["gamma_mN_m"])
    
    if len(gamma_data) < 4:
        print(f"Warning: Insufficient data for {feed_key}")
        return None
    
    # Create figure
    fig, ax = ps.plot_init()
    
    # Grid generation
    grid_size       = 100
    T_grid          = np.linspace(min(T_gamma), max(T_gamma), grid_size)
    P_grid          = np.linspace(min(P_gamma), max(P_gamma), grid_size )
    T_mesh, P_mesh  = np.meshgrid(T_grid, P_grid)
    
    #Interpolation
    gamma_mesh      = griddata((T_gamma, P_gamma), gamma_data, (T_mesh, P_mesh), 
                         method='cubic', fill_value=np.nan)  # CUBIC interpolation (linear, nearest)
    
    # Mask regions outside phase envelope
    envelope_T      = T_bubble + T_dew[::-1]
    envelope_P      = P_bubble + P_dew[::-1]
    envelope_path   = Path(np.column_stack([envelope_T, envelope_P]))
    
    grid_points     = np.column_stack([T_mesh.ravel(), P_mesh.ravel()])
    inside          = envelope_path.contains_points(grid_points).reshape(T_mesh.shape)
    gamma_masked    = np.ma.masked_where(~inside, gamma_mesh)
    
    gamma_min       = np.nanmin(gamma_masked)
    gamma_max       = np.nanmax(gamma_masked)

    # Isoline levels
    levels_lines    = [2*n + 1 for n in range(10)]
    levels_lines    = [lev for lev in levels_lines if gamma_min <= lev <= gamma_max]

    # Color bands
    levels          = np.linspace(gamma_min, gamma_max, 35)
    contour         = ax.contourf(T_mesh, P_mesh, gamma_masked, levels=levels, 
                                 cmap=plt.cm.Blues, extend='both')
    
    # Contour lines
    # Erode the mask slightly to avoid edge artifacts
    inside_eroded   = binary_erosion(inside, iterations=2)
    gamma_masked    = np.ma.masked_where(~inside_eroded, gamma_mesh)

    contour_lines   = ax.contour(T_mesh, P_mesh, gamma_masked, levels=levels_lines,
                            colors='black', linewidths=ps.linewidth/2, alpha=0.6)

    ax.clabel(contour_lines, levels=levels_lines, inline=True,
            fontsize=ps.label_fontsize/2, fmt='%.0f',
            inline_spacing= 15, rightside_up=True)
        
    # ax.clabel(contour_lines, levels=levels_lines, inline=True, 
    #         fontsize=ps.label_fontsize/2, fmt='%.0f',inline_spacing=5, rightside_up=True)
    
    # Colorbar legend
    cbar = fig.colorbar(contour, ax=ax, pad=0.02,  format='%.0f')
    cbar.set_ticks(levels_lines)
    cbar.set_label(r'$\gamma$ [mN/m]', fontsize=ps.label_fontsize)
    cbar.ax.tick_params(labelsize=10) 
    
    # Phase envelope curves
    ax.plot(T_bubble, P_bubble, 'b-', linewidth=ps.linewidth, label='Bubble curve', zorder=10)
    ax.plot(T_dew, P_dew, 'r-', linewidth=ps.linewidth, label='Dew curve', zorder=10)
    ax.plot([T_dew[-1], Tc] , [P_dew[-1], Pc], linestyle="-", linewidth=ps.linewidth, color="red")

    # Critical point
    ax.plot(Tc, Pc, 'o', markersize=4, zorder=15, markerfacecolor="grey", 
        markeredgewidth=1, markeredgecolor="black", label="Critical point")
    
    # Labels
    ax.set_xlabel(f'$T$ / [K]', fontsize=ps.label_fontsize)
    ax.set_ylabel(f'$P$ / [bar]', fontsize=ps.label_fontsize)
    ps.style_legend(ax,fontsize=ps.legend_fontsize, loc='best', framealpha=0)
    
    z_str = ", ".join(f"{comp}={val:.2f}" for comp, val in zip([latex_formula(C_1), 
            latex_formula(C_2), latex_formula(C_3)], z))
    ax.set_title(f"{z_str}", fontsize=ps.label_fontsize/2)
    ax.set_xlim(left=200)
    ax.minorticks_on()
    
    plt.tight_layout()
    
    return fig

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
