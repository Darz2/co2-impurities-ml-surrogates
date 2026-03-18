# A6: PC-SAFT cDFT + Machine Learning Workspace

A Python workspace for computing **vapor-liquid interfacial properties** of multicomponent mixtures using **PCP-SAFT** (Perturbed Chain Polar Statistical Associating Fluid Theory) equation of state combined with **classical Density Functional Theory (cDFT)**.

## Overview

This project provides tools for:

- **Phase equilibrium calculations** (VLE - Vapor-Liquid Equilibrium) for N-component mixtures
- **Interfacial tension computations** using cDFT planar interface calculations
- **Semi-empirical IFT correlations** (Parachor model, Winterfeld-Scriven-Davis model)
- **Binary interaction parameter (kij) fitting** with temperature-dependent correlations
- **Machine learning dataset generation** for thermodynamic property prediction

The primary focus is on CO2-rich mixtures relevant to carbon capture and sequestration (CCS) applications, including binary and multicomponent systems with H2, Ar, N2, CH4, and other gases.

## Project Structure

```
A6/
├── PCSAFT_cDFT/                    # Main project directory
│   ├── thermoift/                  # Core thermodynamics library (submodule)
│   │   └── src/thermoift/
│   │       ├── FeosPlugin.py       # PC-SAFT VLE and cDFT computations via feos
│   │       ├── Analyzer.py         # Results analysis utilities
│   │       ├── ML_feeds.py         # Feed composition generation for ML
│   │       ├── PLOT_SETTINGS.py    # Publication-quality plotting
│   │       └── BINARY_INTERACTION_PARAMETERS/  # kij data and fitting
│   ├── Interfacethermopy/          # IFT correlations library (submodule)
│   │   └── src/
│   │       └── parachor.py         # Parachor & WSD models
│   ├── examples/                   # Example notebooks
│   │   ├── BINARY_MIXTURES/        # CO2-CH4, CO2-N2, CO2-Ar, CO2-H2
│   │   ├── QUATERNARY_MIXTURES/    # 4-component systems
│   │   ├── BUILD_phase/            # Development examples
│   │   └── ML_examples/            # Machine learning workflows
│   └── GENDATA/                    # Data generation scripts
├── feos_09/                        # feos library (submodule)
├── py_A6/                          # Python virtual environment
└── pyproject.toml                  # Project configuration
```

## Requirements

- Python >= 3.9
- [feos](https://github.com/feos-org/feos) >= 0.9.3 (Rust-based thermodynamics library)

### Core Dependencies

```
feos>=0.9.3
numpy
pandas
matplotlib
scipy
molmass==2026.1.8
si-units==0.11.1
sympy==1.14.0
ctREFPROP==0.10.5
scienceplots
```

### Optional Dependencies

- **REFPROP**: NIST thermodynamic database (requires separate license)
- **PyTorch/TabPFN**: For machine learning applications

## Installation

1. Clone the repository with submodules:
```bash
git clone --recurse-submodules https://github.com/darshan-raju/A6.git
cd A6
```

2. Create and activate virtual environment:
```bash
python -m venv py_A6
source py_A6/bin/activate  # Linux/macOS
```

3. Install the workspace and dependencies:
```bash
pip install -e .
```

4. Install submodules (thermoift and Interfacethermopy):
```bash
pip install -e PCSAFT_cDFT/thermoift
pip install -e PCSAFT_cDFT/Interfacethermopy
```

## Usage

### Basic VLE and IFT Calculation

```python
from thermoift import FeosPlugin as fp

# Define mixture components and composition
components = ["carbon dioxide", "hydrogen", "argon"]
feed = [0.90, 0.05, 0.05]  # Mole fractions

# Build parameters with temperature-dependent kij
params = fp.ParameterBuilder.build(components, T=280.0)

# Create equation of state
eos = fp.feos.PcSaft(params)

# Calculate VLE at specified conditions
T = 280.0  # K
P = 50e5   # Pa
vle_result = fp.VLECalculator.compute(eos, T, P, feed)

# Calculate interfacial tension using cDFT
gamma = fp.InterfacialTensionCalculator.compute(eos, vle_result)
print(f"Interfacial tension: {gamma:.4f} mN/m")
```

### Using Semi-Empirical IFT Models

```python
from Interfacethermopy import parachor as IFT

# Initialize with coefficient database
model = IFT.InterfacialTension(["Mulero_2012.json"])

# Calculate pure component IFT
gamma_pure = model.compute_gamma_pure("carbon dioxide", T=250.0)

# Calculate mixture IFT using Parachor model
gamma_mix = model.compute_gamma_mixture(
    components=["CO2", "CH4"],
    x_liquid=[0.95, 0.05],
    y_vapor=[0.80, 0.20],
    rho_liquid=1000.0,
    rho_vapor=50.0,
    T=250.0
)
```

## Key Features

### thermoift Module

- **RegistryManager**: Component parameter management with PC-SAFT coefficients
- **ParameterBuilder**: Build feos Parameters with temperature-dependent kij
- **VLECalculator**: Vapor-liquid equilibrium computations
- **InterfacialTensionCalculator**: cDFT planar interface computations
- **DataProcessor**: Results aggregation and CSV export for ML datasets
- **FeedsBuilder**: Generate random/systematic compositions for data generation

### Interfacethermopy Module

- Pure fluid IFT using empirical correlations (Mulero et al., Cachadina et al.)
- Mixture IFT using Parachor model with cross-interaction parameters
- Winterfeld-Scriven-Davis (WSD) model for multicomponent systems
- Classical Nucleation Theory (CNT) for metastable phase limits
- Integration with REFPROP, Clapeyron.jl, and feos

## Examples

Jupyter notebooks are provided in `PCSAFT_cDFT/examples/`:

| Directory | Description |
|-----------|-------------|
| `BINARY_MIXTURES/` | CO2-CH4, CO2-N2, CO2-Ar, CO2-H2 phase diagrams and IFT |
| `QUATERNARY_MIXTURES/` | 4-component CO2-H2-Ar-N2 systems |
| `BUILD_phase/` | VLE/IFT workflow development |
| `ML_examples/` | Machine learning for property prediction |

## References

1. **Raju, D.; Skartlien, R.; Ramdin, M.; Vlugt, T. J. H. (2025)**
   Vapor-Liquid Interfacial Properties of CO2 Mixtures for Sequestration Applications: Molecular Simulations, Classical Density Functional Theory, and Equations of State
   *Industrial & Engineering Chemistry Research*
   https://doi.org/10.1021/acs.iecr.5c04932

2. **Gross, J.; Sadowski, G. (2001)**
   Perturbed-Chain SAFT: An Equation of State Based on a Perturbation Theory for Chain Molecules
   *Ind. Eng. Chem. Res.*, 40, 1244-1260

3. **feos documentation**: https://feos-org.github.io/feos/

## Author

**Darshan Raju**
Process and Energy Department
Delft University of Technology
Email: d.raju@tudelft.nl

## License

Proprietary - All rights reserved.
