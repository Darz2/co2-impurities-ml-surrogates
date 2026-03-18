# VLE_IFT Notebook: Code Comparison

## Side-by-Side Comparison of Key Cells

### Cell 2: Setup and Parameters

#### Original Code (Old API)
```python
import thermoift.FeosPlugin as TC

COMPONENTS = ["carbon dioxide", "hydrogen", "argon"]
feeds = TC.generate_feeds(CO2=0.99, n_points=4)
builder = KIJ.KIJMatrixBuilder(root=str(KIJ_DIR), kij_filename="KIJ.json", verbose=False)

# Old style - function calls
parameters = TC.PARAMETERS(COMPONENTS, T_K=300.0, kij_builder=builder, model_map=KIJ_map)
components = TC.components(parameters)
KIJ_LABELS = TC._kij_labels(COMPONENTS)

TC.print_array2d(feeds)
```

#### New Code (New Class-Based API)
```python
from thermoift.FeosPlugin import (
    CompositionHandler,
    ParameterBuilder,
    RegistryManager
)

COMPONENTS = ["carbon dioxide", "hydrogen", "argon"]
feeds = CompositionHandler.generate_feeds(CO2=0.99, n_points=4)
builder = KIJ.KIJMatrixBuilder(root=str(KIJ_DIR), kij_filename="KIJ.json", verbose=False)

# New style - class methods
parameters = ParameterBuilder.build_parameters(COMPONENTS, T_K=300.0, kij_builder=builder, model_map=KIJ_map)
components = RegistryManager.get_component_names(parameters)
KIJ_LABELS = RegistryManager.kij_labels_from_names(COMPONENTS)

CompositionHandler.print_array2d(feeds)
```

**Benefits:**
- ✅ Clearer organization with named imports
- ✅ Better IDE autocomplete
- ✅ Explicit class grouping makes intent clear

---

### Cell 4: Phase Envelope Calculations

#### Original Code (Old API)
```python
for k, z in enumerate(feeds, start=1):
    z = TC.normalize_z(z)
    feed = TC.compute_feed_moles(z)

    active_z, active_components, is_reduced = TC.reduce_components(z, COMPONENTS, verbose=verbose)
    active_feed = TC.compute_feed_moles(active_z)
    active_map = TC.reduce_kij_map(active_components, KIJ_map)

    parameters_ref = TC.PARAMETERS(active_components, T_K=T_initial_guess, kij_builder=builder, model_map=active_map)
    eos = feos.HelmholtzEnergyFunctional.pcsaft(parameters_ref)

    CT, CP = TC.compute_CT(feos, eos, active_z, T_guess=T_initial_guess)

    T_values = range(int(T_initial_guess), int(CT / si.KELVIN) + 1, T_STEP)

    for T_K in T_values:
        parameters_T = TC.PARAMETERS(active_components, T_K=float(T_K), kij_builder=builder, model_map=active_map)
        eos_T = feos.HelmholtzEnergyFunctional.pcsaft(parameters_T)

        T_bub, P_bub = TC.compute_bubble_curve(eos_T, [T_K], active_feed, verbose=verbose)
        T_dew, P_dew = TC.compute_dew_curve(eos_T, [T_K], active_feed, verbose=verbose)

        T_bubble_all.extend(T_bub)
        P_bubble_all.extend(P_bub)
        T_dew_all.extend(T_dew)
        P_dew_all.extend(P_dew)
```

#### New Code (New Class-Based API)
```python
from thermoift.FeosPlugin import (
    CompositionHandler,
    ParameterBuilder,
    VLECalculator
)

for k, z in enumerate(feeds, start=1):
    z = CompositionHandler.normalize_z(z)
    feed = CompositionHandler.compute_feed_moles(z)

    active_z, active_components, is_reduced = CompositionHandler.reduce_components(z, COMPONENTS, verbose=verbose)
    active_feed = CompositionHandler.compute_feed_moles(active_z)
    active_map = ParameterBuilder.reduce_kij_map(active_components, KIJ_map)

    parameters_ref = ParameterBuilder.build_parameters(active_components, T_K=T_initial_guess, kij_builder=builder, model_map=active_map)
    eos = feos.HelmholtzEnergyFunctional.pcsaft(parameters_ref)

    CT, CP = VLECalculator.compute_critical_point(eos, active_z, T_guess=T_initial_guess)

    T_values = range(int(T_initial_guess), int(CT / si.KELVIN) + 1, T_STEP)

    for T_K in T_values:
        parameters_T = ParameterBuilder.build_parameters(active_components, T_K=float(T_K), kij_builder=builder, model_map=active_map)
        eos_T = feos.HelmholtzEnergyFunctional.pcsaft(parameters_T)

        T_bub, P_bub = VLECalculator.compute_bubble_curve(eos_T, [T_K], active_feed, verbose=verbose)
        T_dew, P_dew = VLECalculator.compute_dew_curve(eos_T, [T_K], active_feed, verbose=verbose)

        T_bubble_all.extend(T_bub)
        P_bubble_all.extend(P_bub)
        T_dew_all.extend(T_dew)
        P_dew_all.extend(P_dew)
```

**Benefits:**
- ✅ Clear grouping: `CompositionHandler` for composition operations
- ✅ Clear grouping: `ParameterBuilder` for parameter building
- ✅ Clear grouping: `VLECalculator` for VLE operations
- ✅ Easier to test each component independently

---

### Cell 5: TP Flash and cDFT Calculations

#### Original Code (Old API)
```python
# TP flash
try:
    eq, x, y, liquid_density, vapor_density = TC.tp_flash(
        feos, eos_T, T*si.KELVIN, P*si.BAR, active_z*si.MOL, molar_masses)
except Exception as e:
    print(f"TP flash failed: {e}")
    continue

# Planar interface
interface = TC.build_planar_interface(eq, critical_temperature=CT, n_grid=ngrid, l_grid=lgrid)

try:
    gamma_mN_m, interfacial_thickness_nm, enrichment = \
        TC.solve_interface_properties(interface, si)
except Exception as e:
    print(f"Surface tension calculation failed: {e}")
    gamma_mN_m = np.nan

# Data processing
row_data = TC.row_append(T, P, active_z, CT, CP, P_b, P_d, liquid_density, vapor_density,
                         x, y, gamma_mN_m, interfacial_thickness_nm, active_components, enrichment,
                         ML_mode=True)

# Plotting
fig, ax = TC.plot_PT(PT_results, feed_key, parameters)
TC.save_plot(fig, f"PT_{feed_key}")

fig_gamma = TC.plot_gamma_colormap(PT_results, VLE_DFT, feed_key, parameters)
if fig_gamma is not None:
    TC.save_plot(fig_gamma, f"Gamma_{feed_key}")

# Summary
summary = TC.VLE_DFT_summary(VLE_DFT)
saved_files = TC.VLE_DFT_to_csv(VLE_DFT, folder="CSV", verbose=verbose)
```

#### New Code (New Class-Based API)
```python
from thermoift.FeosPlugin import (
    VLECalculator,
    InterfacialTensionCalculator,
    DataProcessor,
    PlottingEngine
)

# TP flash
try:
    eq, x, y, liquid_density, vapor_density = VLECalculator.tp_flash(
        eos_T, T*si.KELVIN, P*si.BAR, active_z*si.MOL, molar_masses)
except Exception as e:
    print(f"TP flash failed: {e}")
    continue

# Planar interface
interface = InterfacialTensionCalculator.build_planar_interface(eq, critical_temperature=CT, n_grid=ngrid, l_grid=lgrid)

try:
    gamma_mN_m, interfacial_thickness_nm, enrichment = \
        InterfacialTensionCalculator.solve_interface_properties(interface)
except Exception as e:
    print(f"Surface tension calculation failed: {e}")
    gamma_mN_m = np.nan

# Data processing
row_data = DataProcessor.assemble_row(T, P, active_z, CT, CP, P_b, P_d, liquid_density, vapor_density,
                                     x, y, gamma_mN_m, interfacial_thickness_nm, active_components, enrichment,
                                     ML_mode=True)

# Plotting
fig, ax = PlottingEngine.plot_phase_diagram(PT_results, feed_key, parameters)
PlottingEngine.save_plots(fig, f"PT_{feed_key}")

fig_gamma = PlottingEngine.plot_interfacial_tension_map(PT_results, VLE_DFT, feed_key, parameters)
if fig_gamma is not None:
    PlottingEngine.save_plots(fig_gamma, f"Gamma_{feed_key}")

# Summary
summary = DataProcessor.summarize_vle_dft(VLE_DFT)
saved_files = DataProcessor.export_to_csv(VLE_DFT, folder="CSV", verbose=verbose)
```

**Benefits:**
- ✅ Each class handles its domain: VLE, IFT, data, plotting
- ✅ Function names more explicit: `plot_phase_diagram()` vs `plot_PT()`
- ✅ Method signatures clearer: `solve_interface_properties(interface)` vs `solve_interface_properties(interface, si)`
- ✅ Can import only what you need

---

## Summary of Improvements

| Aspect | Old API | New API |
|--------|---------|---------|
| **Organization** | All functions mixed | 9 functional classes |
| **Import clarity** | `import ... as TC` | `from ... import SpecificClass` |
| **IDE autocomplete** | Generic `TC.*` | Class-specific methods shown |
| **Finding functions** | Search the whole module | Search specific class |
| **Testing** | Test entire module | Test individual classes |
| **Backward Compat** | N/A | 100% compatible ✅ |
| **Migration effort** | N/A | Zero (use wrappers) |

---

## Recommendation

**For new work**: Use the class-based API (`VLE_IFT_ClassBased.ipynb`)
- Better organized
- More maintainable
- Easier to understand

**For existing work**: Use the old API (`VLE_IFT.ipynb`)
- No changes needed
- Full compatibility
- Can migrate gradually

**For mixed projects**: Use both!
- Wrapper functions allow seamless mixing
- Migrate at your own pace
- Maximum flexibility

---

## File Locations

```
TERNARY_MIXTURES/CO2-H2-Ar/
├── VLE_IFT.ipynb                    # Original notebook (old API)
├── VLE_IFT_ClassBased.ipynb         # NEW: Class-based API notebook
└── API_MIGRATION_GUIDE.md           # NEW: This comparison guide

thermoift/src/thermoift/
└── FeosPlugin.py                    # Refactored with 9 classes + backward compat wrappers
```

**Both notebooks produce identical results!** Choose based on your preference. 🚀
