# GENDATA: VLE-IFT Dataset Generation Pipeline

This directory contains the automated pipeline for generating **vapor-liquid equilibrium (VLE)** and **Interfacial properties** datasets for multicomponent CO2-rich mixtures using PCP-SAFT cDFT calculations.

## Purpose

Generate large-scale thermodynamic property datasets for:
- Machine learning model training (surrogate models for Interfacial tension, saturation pressure prediction)
- Validation of semi-empirical correlations
- Phase behavior analysis across composition space

## Directory Structure

```
GENDATA/
├── Feed.ipynb              # Composition generation notebook
├── VLE_IFT_V2.ipynb        # VLE/IFT computation (papermill-parameterized)
├── VLE_IFT_V3.ipynb        # CSV-driven batch processing version
├── Analyze.ipynb           # Results analysis and visualization
├── RUN.sh                  # SLURM batch job submission script
├── CSV_feeds/              # Generated feed compositions
│   ├── Random_compositions.csv
│   ├── Systematic_compositions.csv
│   ├── Industrial_compositions.csv
│   └── Combined_compositions.csv
├── CSV/                    # Output VLE-IFT results per job
│   └── CSV_<JOB_ID>/
│       ├── feed_<N>_interfacial_results.csv
│       └── feed_<N>_phase_envelope.csv
└── PLOTS/                  # Generated phase diagrams and IFT maps
    └── PDF/ & PNG/
```

## Notebook Versions: V2 vs V3

Two computation notebooks are provided for different use cases:

### VLE_IFT_V2.ipynb — Papermill-Parameterized (Targeted Runs)

**Use case:** Small, targeted studies with specific ternary mixtures and controlled kij settings.

| Feature | Description |
|---------|-------------|
| **Input method** | Parameters defined in notebook cells (papermill-injectable) |
| **Components** | Manually specified list (e.g., `["carbon dioxide", "hydrogen", "argon"]`) |
| **Compositions** | Auto-generated via `CompositionHandler.generate_feeds(CO2=0.99, n_points=4)` |
| **KIJ handling** | Explicit `KIJ_map` dictionary with per-pair model selection |
| **Typical use** | SLURM batch jobs via `RUN.sh` with papermill parameter injection |

```python
# V2 configuration example
COMPONENTS = ["carbon dioxide", "hydrogen", "argon"]
KIJ_map = {
    ("CO2", "Ar"): "constant",   # Uses fitted kij value
    ("H2", "Ar"): "zero",        # kij = 0
    ("CO2", "H2"): "zero"
}
feeds = CompositionHandler.generate_feeds(CO2=0.99, n_points=4)
```

### VLE_IFT_V3.ipynb — CSV-Driven (Large-Scale Batch)

**Use case:** Large-scale dataset generation with diverse compositions from pre-generated CSV files.

| Feature | Description |
|---------|-------------|
| **Input method** | Loads compositions from CSV files in `CSV_feeds/` |
| **Components** | Full 8-component system auto-mapped via `RegistryManager.map_csv_to_components()` |
| **Compositions** | Loaded via `CompositionHandler.load_compositions(csv_path, n=100)` |
| **KIJ handling** | Auto-generates `KIJ_map` for all 28 binary pairs (defaults to `"zero"`) |
| **Typical use** | Processing thousands of compositions from `Feed.ipynb` output |

```python
# V3 configuration example
csv_path = "CSV_feeds/Combined_compositions.csv"
COMPONENTS = RegistryManager.map_csv_to_components()  # All 8 components
compositions = CompositionHandler.load_compositions(csv_path, n=100)

# Auto-build KIJ_map for all pairs (can override specific pairs)
KIJ_LABELS = RegistryManager.kij_labels_from_names(COMPONENTS)
KIJ_map = {pair: "zero" for pair in combinations(KIJ_LABELS, 2)}
# KIJ_map[("CO2", "Ar")] = "constant"  # Optional override
```

## Workflow

### Step 1: Generate Feed Compositions

Run `Feed.ipynb` to create mixture compositions using `FeedsBuilder`:

```python
from thermoift import FeedsBuilder

builder = FeedsBuilder(rng_type="PCG64")

# Random feeds via Dirichlet sampling
df_random = builder.generate_random_feeds(
    components=["CO2", "H2", "Ar", "N2", "CH4", "O2", "CO", "H2S"],
    mixture_sizes=(2, 3),
    co2_levels=(0.95, 0.96, 0.97, 0.98, 0.99),
    n_random_samples=200,
    rng=550055,
)

# Systematic grid feeds
df_systematic = builder.generate_systematic_feeds(...)

# Industrial template feeds (NETL limits, Sleipner, etc.)
df_industrial = builder.industrial_feeds_from_bounds(...)
```

**Output:** CSV files in `CSV_feeds/` with columns:
- Component mole fractions (CO2, H2, Ar, N2, CH4, O2, CO, H2S)
- `mixture_size`, `active_components`, `feed_source`, `template_name`

### Step 2: Compute VLE and IFT

Run `VLE_IFT_V2.ipynb` (or submit via `RUN.sh`) with parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `COMPONENTS` | List of component names | `["carbon dioxide", "hydrogen", "argon"]` |
| `KIJ_map` | Binary interaction parameter models | `{("CO2","Ar"): "constant", ...}` |
| `CO2_comp` | Minimum CO2 mole fraction | `0.99` |
| `n_feeds` | Number of feed compositions | `4` |
| `TEST_RUN` | Use coarse grid (fast) or fine grid | `True` |
| `CSV_FOLDER` | Output folder name | `"CSV"` |

**Computation steps per feed:**
1. Compute critical point (T_c, P_c)
2. Trace bubble and dew curves (phase envelope)
3. For each isotherm within two-phase region:
   - TP flash at multiple pressures
   - Solve cDFT planar interface
   - Extract IFT, interfacial thickness, surface enrichment

### Step 3: Batch Submission (HPC)

Use `RUN.sh` for SLURM cluster submission:

```bash
sbatch RUN.sh
```

The script uses [papermill](https://papermill.readthedocs.io/) for parameterized notebook execution:

```bash
papermill VLE_IFT_V2.ipynb VLE_IFT_V2_output.ipynb \
  -p CO2_comp 0.99 \
  -p n_feeds 4 \
  -p TEST_RUN True \
  -p CSV_FOLDER "CSV_${SLURM_JOB_ID}" \
  -y "
COMPONENTS:
  - carbon dioxide
  - hydrogen
  - argon
KIJ_map:
  ? [CO2, Ar]
  : constant
  ? [H2, Ar]
  : zero
  ? [CO2, H2]
  : zero
"
```

## Output Data Format

### Phase Envelope CSV (`feed_<N>_phase_envelope.csv`)

| Column | Description |
|--------|-------------|
| `T_K` | Temperature (K) |
| `P_bubble_bar` | Bubble pressure (bar) |
| `P_dew_bar` | Dew pressure (bar) |

### Interfacial Results CSV (`feed_<N>_interfacial_results.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `T_K` | scalar | Temperature (K) |
| `P_bar` | scalar | Pressure (bar) |
| `z_<comp>` | array | Feed mole fractions (one column per active component: `z_CO2`, `z_H2`, `z_Ar`, ...) |
| `x_<comp>` | array | Liquid phase mole fractions (one column per active component) |
| `y_<comp>` | array | Vapor phase mole fractions (one column per active component) |
| `rho_L_mol_cm3` | scalar | Liquid phase molar density (mol/cm³) |
| `rho_V_mol_cm3` | scalar | Vapor phase molar density (mol/cm³) |
| `gamma_mN_m` | scalar | Interfacial tension (mN/m) |
| `thickness_nm` | scalar | Interfacial thickness via 10-90 criterion (nm) |
| `enrichment_<comp>` | array | Surface enrichment factor per component (Γᵢ / Γᵢ,bulk) |
| `Tc_K` | scalar | Mixture critical temperature (K) |
| `Pc_bar` | scalar | Mixture critical pressure (bar) |
| `Tr` | scalar | Reduced temperature (T / Tc) |
| `Pr` | scalar | Reduced pressure (P / Pc) |
| `gamma0_CO2` | scalar | Pure CO2 reference IFT at same T (mN/m) |
| `rhoL0_CO2_mol_cm3` | scalar | Pure CO2 saturated liquid density (mol/cm³) |
| `rhoV0_CO2_mol_cm3` | scalar | Pure CO2 saturated vapor density (mol/cm³) |
| `Psat_CO2_bar` | scalar | Pure CO2 saturation pressure (bar) |

**Note:** The `<comp>` suffix corresponds to active components in the mixture (e.g., `CO2`, `H2`, `Ar`, `N2`, `CH4`, `O2`, `CO`, `H2S`). Only non-zero components are included.

## Grid Settings

| Mode | `T_STEP` | `P_STEP` | `ngrid` | `lgrid` | Use Case |
|------|----------|----------|---------|---------|----------|
| `TEST_RUN=True` | 10 K | 5 bar | 500 | 100 A | Development/debugging |
| `TEST_RUN=False` | 1 K | 1 bar | 2048 | 100 A | Production runs |

## Requirements

- Python virtual environment: `source /home/darshan/A6/py_A6/bin/activate`
- LaTeX (for plot generation): TexLive 2025
- thermoift package: `pip install -e ../thermoift`
- papermill: `pip install papermill`

## Notes

- KIJ models: `"constant"` uses fitted value, `"zero"` sets kij=0, `"Tdep"` uses T-dependent correlation
- Phase envelope computation rebuilds EOS at each temperature for T-dependent kij
- cDFT solver uses planar interface approximation with specified grid resolution
- Output plots are publication-ready (PDF + PNG) via `scienceplots` styling
