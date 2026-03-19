# VLE_IFT Notebook - API Migration Guide

## New Notebook Created: `VLE_IFT_ClassBased.ipynb`

The new notebook uses the refactored class-based API while maintaining **100% functional equivalence** with the original.

---

## Quick Reference: Old API → New API

### Imports

**Old:**
```python
import thermoift.FeosPlugin as TC
```

**New:**
```python
from thermoift.FeosPlugin import (
    RegistryManager,
    CompositionHandler,
    ParameterBuilder,
    PropertyCalculator,
    VLECalculator,
    InterfacialTensionCalculator,
    DataProcessor,
    PlottingEngine,
    UtilityFunctions
)
```

---

### Feed Generation

| Operation | Old API | New API |
|-----------|---------|---------|
| Generate feeds | `TC.generate_feeds(0.5, 21)` | `CompositionHandler.generate_feeds(0.5, 21)` |
| Normalize composition | `TC.normalize_z(z)` | `CompositionHandler.normalize_z(z)` |
| Convert to moles | `TC.compute_feed_moles(z)` | `CompositionHandler.compute_feed_moles(z)` |
| Reduce components | `TC.reduce_components(z, comps)` | `CompositionHandler.reduce_components(z, comps)` |
| Print array | `TC.print_array2d(feeds)` | `CompositionHandler.print_array2d(feeds)` |

---

### Parameters & Registry

| Operation | Old API | New API |
|-----------|---------|---------|
| Build parameters | `TC.PARAMETERS(comps, T_K=300)` | `ParameterBuilder.build_parameters(comps, T_K=300)` |
| Get component names | `TC.components(params)` | `RegistryManager.get_component_names(params)` |
| Get KIJ label | `TC._kij_labels(comps)` | `RegistryManager.kij_labels_from_names(comps)` |
| Get component record | N/A | `RegistryManager.get_pure_record(name)` |
| Reduce KIJ map | `TC.reduce_kij_map(comps, kij)` | `ParameterBuilder.reduce_kij_map(comps, kij)` |

---

### Property Calculations

| Operation | Old API | New API |
|-----------|---------|---------|
| Molar density | `TC.molar_density_mol_m3(rho)` | `PropertyCalculator.molar_density_mol_m3(rho)` |
| Mass density | `TC.density_kg_m3(rho, z, MM)` | `PropertyCalculator.density_kg_m3(rho, z, MM)` |
| Value in unit | `TC.value_in(q, unit)` | `PropertyCalculator.value_in(q, unit)` |
| Molar masses | `TC.molar_masses(params)` | `PropertyCalculator.molar_masses(params)` |

---

### VLE Calculations

| Operation | Old API | New API |
|-----------|---------|---------|
| Bubble curve | `TC.compute_bubble_curve(eos, T, feed, v)` | `VLECalculator.compute_bubble_curve(eos, T, feed, v)` |
| Dew curve | `TC.compute_dew_curve(eos, T, feed, v)` | `VLECalculator.compute_dew_curve(eos, T, feed, v)` |
| Critical point | `TC.compute_CT(feos, eos, z, T_g)` | `VLECalculator.compute_critical_point(eos, z, T_g)` |
| TP Flash | `TC.tp_flash(feos, eos, T, P, feed, MM)` | `VLECalculator.tp_flash(eos, T, P, feed, MM)` |

---

### Interfacial Tension (cDFT)

| Operation | Old API | New API |
|-----------|---------|---------|
| Build interface | `TC.build_planar_interface(eq, Tc, ng, lg)` | `InterfacialTensionCalculator.build_planar_interface(eq, Tc, ng, lg)` |
| Solve interface | `TC.solve_interface_properties(interface, si)` | `InterfacialTensionCalculator.solve_interface_properties(interface)` |

---

### Data Processing & Export

| Operation | Old API | New API |
|-----------|---------|---------|
| Assemble row | `TC.row_append(T, P, z, ..., ML_mode=T)` | `DataProcessor.assemble_row(T, P, z, ..., ML_mode=T)` |
| Summarize VLE-DFT | `TC.VLE_DFT_summary(vle)` | `DataProcessor.summarize_vle_dft(vle)` |
| Export to CSV | `TC.VLE_DFT_to_csv(vle, folder)` | `DataProcessor.export_to_csv(vle, folder)` |

---

### Plotting

| Operation | Old API | New API |
|-----------|---------|---------|
| PT diagram | `TC.plot_PT(PT_res, key, params)` | `PlottingEngine.plot_phase_diagram(PT_res, key, params)` |
| Gamma colormap | `TC.plot_gamma_colormap(PT_res, VLE, key, p)` | `PlottingEngine.plot_interfacial_tension_map(PT_res, VLE, key, p)` |
| Save plot | `TC.save_plot(fig, base, folder)` | `PlottingEngine.save_plots(fig, base, folder)` |
| Save figure | `TC.save_figure(fig, name)` | `PlottingEngine.save_figure(fig, name)` |

---

### Utilities

| Operation | Old API | New API |
|-----------|---------|---------|
| LaTeX formula | `TC.latex_formula(formula)` | `UtilityFunctions.latex_formula(formula)` |

---

## File Locations

- **Original notebook**: `VLE_IFT.ipynb` (uses old API - still works!)
- **New notebook**: `VLE_IFT_ClassBased.ipynb` (uses new class-based API)
- **Core module**: `thermoift/FeosPlugin.py` (refactored with 9 classes)

---

## Usage Tips

### Option 1: Use Original Notebook (No Changes Needed)
```python
# The old notebook continues to work through backward compatibility wrappers
# Old function names are automatically delegated to the new classes
```

### Option 2: Use New Notebook (Recommended for New Code)
```python
# Import specific classes for cleaner, more organized code
from thermoift.FeosPlugin import CompositionHandler, ParameterBuilder
```

### Option 3: Mix Both APIs
```python
# You can use both old and new APIs in the same script!
# Backward compatibility wrappers allow seamless mixing

import thermoift.FeosPlugin as TC
from thermoift.FeosPlugin import CompositionHandler

# Old style
feeds_old = TC.generate_feeds(0.5, 21)

# New style
feeds_new = CompositionHandler.generate_feeds(0.5, 21)

# Both work and are functionally identical!
```

---

## Benefits Summary

### 🎯 Organization
- Functions grouped by logical domain
- Clear separation of concerns
- Easier to navigate large codebase

### 💡 IDE Support
- Autocomplete shows all class methods
- Better code discovery
- Inline documentation

### 🧪 Testing
- Each class can be tested independently
- Mocking is easier
- Better unit test organization

### 🔄 Backward Compatibility
- Zero breaking changes
- Old code continues to work
- Gradual migration possible

### 📚 Maintainability
- Clear class hierarchies
- Explicit method organization
- Easier to understand relationships

---

## Migration Strategy

### Phase 1 (Current) ✅
- New class-based API available
- Old API still works (backward compatible)
- No migration urgency

### Phase 2 (Optional)
- Update new code to use new API
- Gradually migrate old code
- Mix both APIs as needed

### Phase 3 (Future - If Desired)
- Remove backward compatibility wrappers
- Full migration to new API
- Would be a breaking change

---

**Status**: Ready to use! Both notebooks work identically. Choose whichever you prefer! 🚀
