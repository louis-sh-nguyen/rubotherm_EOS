# CO2-HDPE Sorption Modeling Tool

This repository contains code to model and analyze CO2 sorption in High-Density Polyethylene (HDPE) using various equations of state (EOS), including SAFT-γ Mie and Span-Wagner.

## Overview

This project implements thermodynamic modeling for CO2 sorption in HDPE polymers based on experimental data from a Rubotherm sorption apparatus. The code allows calculation of:

- Solubility of CO2 in HDPE at different temperatures and pressures
- Swelling ratio predictions
- Density comparisons between experimental and model data
- Partial molar volumes of solute and polymer

## Requirements

The project requires several Python packages including:
- NumPy
- Pandas
- Matplotlib
- SciPy
- sgtpy_NETGP (SAFT-γ Mie implementation)

## Installation

### Setting up the environment

1. Clone the repository
2. Create a conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate rubotherm_eos
```

## Usage

### Main Scripts

- `solubility_master.py`: Core module containing classes and functions for thermodynamic calculations
- `plot_results.py`: Generate plots comparing model predictions with experimental data
- `plot_solutions.py`: Plot detailed solutions for specific conditions
- `plot_rhoSolPpol.py`: Compare density predictions for CO2 and HDPE
- `plot_VsVp.py`: Plot partial molar volumes

### Running the scripts

To generate model results at 50°C (comparison between experimental data and SW EoS):

```bash
python plot_results.py
```

To evaluate solubility solutions at specific conditions:

```bash
python plot_solutions.py
```

To compare density predictions:

```bash
python plot_rhoSolPpol.py
```

## Data Files

The code uses several Excel files to store and process data:
- `data_CO2-HDPE.xlsx`: Primary experimental data
- `model_lit_data.xlsx`: Literature data and model predictions

## Classes and Functions

### Key Classes

- `BaseSolPol`: Base class for solute-polymer mixture properties
- `DetailedSolPol`: Extends BaseSolPol with additional thermodynamic calculations
- `SolPolExpData`: Handles experimental data loading and processing

### Main Functions

- `solve_solubility()`: Calculate CO2 solubility in HDPE
- `solve_solubility_plot_SwR()`: Plot swelling ratio evaluations
- `plot_modelResults()`: Generate comparison plots between models and experiments

## License

[Include license information here]

## Contributors

Louis Nguyen (sn621@ic.ac.uk)
Department of Chemical Engineering, Imperial College London