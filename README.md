# HPvanRhee_MScThesis_Documentation

This repository contains all relevant coding and modelling aspects for my Systems & Control MSc Thesis on MPC-based FFR in the 2035 Dutch power system.
## Repository structure

```text
HPvanRhee_MScThesis_Documentation/
├── ID_code/
│   └── COI_models/
│   └── id_data/
├── MPC_code/
│   └── renewable/
│           └── cen_helpers/
│           └── dec_helpers/
│           └── dis_helpers/
│           └── sta_helpers/
├── PFlow_2021/
├── PFlow_2035/
└── TDS_simulation_code/
```

## Setup

### 1) For `PFlow_2021` and `PFlow_2035`

| Package | Version |
|---|---:|
| `python` | `3.10.19` |
| `numpy` | `1.23.5` |
| `matplotlib` | `3.10.8` |
| `pandas` | `1.5.3` |
| `scipy` | `1.9.3` |
| `andes` | `1.10.0` |
| `pandapower` | `2.10.0` |

See `environment_PFlow.yml` for the full environment.

### 2) For all other Python code

| Package | Version |
|---|---:|
| `python` | `3.10.19` |
| `numpy` | `2.2.6` |
| `matplotlib` | `3.10.8` |
| `pandas` | `2.3.3` |
| `scipy` | `1.13.1` |
| `andes` | `1.10.0` |
| `cvxpy` | `1.7.5` |
| `control` | `0.10.2` |

See `environment.yml` for the full environment.

### 3) For MATLAB code

| Component | Version |
|---|---:|
| `MATLAB` | `23.2.0.2391609 (R2023b) Update 2` |
| `System Identification Toolbox` | `23.2 (R2023b)` |
| `Control System Toolbox` | `23.2 (R2023b)` |
| `Optimization Toolbox` | `23.2 (R2023b)` |

See `matlab_requirements.txt` for the full setup used.

## ANDES setup

After installing `ANDES`, copy the contents of the `MPC_code/renewable` directory into your local package installation under `andes/models/renewable`.

Then replace the `__init__.py` file inside `andes/models` with the `__init__.py` file inside `MPC_code`.

After doing this, restart your kernel and run:

```python
ss = andes.prepare(models=[
    'REGF1W',
    'REGF1W_MPC_DEC',
    'REGF1W_MPC_CEN',
    'REGF1W_MPC_DIS',
    'REGF1W_MPC_STA',
    'REGF1W_MPC_ID'
])
```

## Contents

### Simulation model
The folders `PFlow_2021` and `PFlow_2035` contain both `pandapower` and `ANDES` versions of the simulation model of the Dutch power system for power flow comparisons.

### MPC code
The `MPC_code` folder contains all Python code used to run the controllers in `ANDES`.

### Identification code
The `ID_code` folder contains the Python code used to generate the identification data and the MATLAB code used to identify the prediction models.

### Time domain simulation code
The `TDS_simulation_code` folder can be used to simulate the responses of the different controllers.