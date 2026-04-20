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
│   │   └── ML_examples/            # ML workflows, symbolic regression (PySR/Julia)
│   ├── GENDATA/                    # Data generation scripts
│   ├── DATASET_A1/                 # Dataset A1: VLE+IFT for random feeds (100 SLURM jobs)
│   │   ├── CombinedDataset_A1.csv  # Aggregated results
│   │   └── VLE_IFT_V4.ipynb        # Generation notebook
│   ├── DATASET_A2/                 # Dataset A2: extended feed space (100 SLURM jobs)
│   │   ├── CombinedDataset_A2.csv  # Aggregated results
│   │   └── VLE_IFT_V4.ipynb        # Generation notebook
│   ├── DATASET_A3/                 # Dataset A3: saturation curves and bulk feeds
│   │   ├── ALL_FEED_RUN/           # All-feed sweep
│   │   └── BULK_RUN/               # Bulk/saturation conditions
│   ├── SEC/                        # Saturation envelope calculations
│   │   ├── SEC_PURE.ipynb          # Pure fluid saturation
│   │   ├── SEC_MULIT.ipynb         # Multicomponent saturation envelope
│   │   └── SEC_REFPROP.ipynb       # REFPROP-based saturation reference
│   ├── ML/                         # Machine learning models (trained on DATASET_A1)
│   │   ├── PRE_ML.ipynb            # Data pre-processing and feature engineering
│   │   ├── interfacial_results_dataset_A1.csv  # ML-ready dataset
│   │   ├── RF/                     # Random Forest (γ, P_bubble, P_dew)
│   │   ├── SVM/                    # Support Vector Machine (γ, P_bubble, P_dew)
│   │   ├── XGBoost/                # XGBoost (γ, P_bubble, P_dew)
│   │   └── TabPFN/                 # TabPFN transformer (γ, P_bubble, P_dew)
│   ├── GaussianProcessRegression/  # GPR and sparse GP models
│   │   ├── GPR.ipynb               # Standard GP regression
│   │   ├── SVGP_GPR_RESIDUAL.ipynb # Sparse Variational GP on residuals
│   │   └── SLURM_GPR_residual_*/   # GPR runs at 1000–5000 inducing points
│   └── symbolic_regression/        # Symbolic regression (PySR)
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

## Datasets

| Dataset | Description | Key file |
|---------|-------------|----------|
| `DATASET_A1/` | Random feed compositions, VLE+IFT via 100 SLURM jobs | `CombinedDataset_A1.csv` |
| `DATASET_A2/` | Extended feed space, same workflow | `CombinedDataset_A2.csv` |
| `DATASET_A3/` | Saturation-curve sweeps and bulk-condition runs | `ALL_FEED_RUN/`, `BULK_RUN/` |

Features: **T, P, z_CO2, z_H2, z_N2, z_Ar, z_CH4, z_O2, z_CO, z_H2S**.
Targets: **γ (mN/m)**, **P_bubble (Pa)**, **P_dew (Pa)**.

## Machine Learning Models

All models are in `PCSAFT_cDFT/ML/` and trained on DATASET_A1 (pre-processed via `PRE_ML.ipynb`).

| Model | Folder | γ test R² | Notes |
|-------|--------|-----------|-------|
| Random Forest | `ML/RF/` | 0.9995 | 500 estimators |
| SVM | `ML/SVM/` | — | Radial basis kernel |
| XGBoost | `ML/XGBoost/` | — | depth=6, lr=0.05 |
| TabPFN | `ML/TabPFN/` | **0.99995** | Transformer-based tabular model |

Each model folder contains local notebooks and `SLURM_*/` subdirectories with HPC-run outputs (predictions CSV + metrics JSON).

### Gaussian Process Regression

`PCSAFT_cDFT/GaussianProcessRegression/` contains:

- `GPR.ipynb` — standard exact GP regression
- `SVGP_GPR_RESIDUAL.ipynb` — Sparse Variational GP trained on residuals from another model, for scalable uncertainty quantification
- `SLURM_GPR_residual_{1000..5000}/` — GPR HPC runs at varying inducing-point counts
- `SLURM_SVGP_residual/` — SVGP HPC run

## Examples

Jupyter notebooks are provided in `PCSAFT_cDFT/examples/`:

| Directory | Description |
|-----------|-------------|
| `BINARY_MIXTURES/` | CO2-CH4, CO2-N2, CO2-Ar, CO2-H2 phase diagrams and IFT |
| `QUATERNARY_MIXTURES/` | 4-component CO2-H2-Ar-N2 systems |
| `BUILD_phase/` | VLE/IFT workflow development |
| `ML_examples/` | ML workflows and symbolic regression (PySR/Julia) |

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
