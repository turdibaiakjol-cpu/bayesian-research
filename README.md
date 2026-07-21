# Nowcasting GDP in a Data-Constrained Commodity Economy

**A Bayesian vs. Classical Comparative Framework for Kazakhstan**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository implements a complete econometric pipeline benchmarking two nowcasting architectures for real GDP growth in Kazakhstan (Q1 2010 – Q4 2025):

| Model | Architecture | Key Feature |
|---|---|---|
| **B-VAR** | Bayesian Vector Autoregression | Minnesota prior shrinkage (λ₁ = 0.10) on K = 7 endogenous variables |
| **MIDAS** | Mixed Data Sampling | Exponential Almon polynomial lag weighting on 3 high-frequency predictors |

### Core Findings

- **MIDAS achieves a 61.9% RMSE improvement** over the B-VAR baseline on a 28-quarter recursive pseudo out-of-sample evaluation (Q1 2019 – Q4 2025), with a Diebold-Mariano p-value of 0.0154.
- **Brent crude oil transmission**: A one standard deviation structural shock produces a GDP impact response equivalent to **0.47 standard deviations**, persisting for seven quarters — confirming Kazakhstan's hydrocarbon dependency channel.
- **STEI as contemporaneous proxy**: The Short-Term Economic Indicator concentrates **89% of predictive weight on the most recent monthly observation** (lag-1 correlation r = 0.96 with quarterly GDP), functioning less as a distributed lag predictor and more as a high-frequency GDP proxy.

---

## Repository Structure

```
bayesian-research/
├── data/
│   ├── data_provenance.csv          # Variable sourcing and metadata
│   ├── raw/
│   │   └── raw_quarterly.csv        # Pre-transformation quarterly aggregates
│   ├── clean/
│   │   └── clean_quarterly.csv      # Log-differenced, seasonally adjusted data
│   └── results/
│       ├── rolling_forecasts_extended.csv
│       ├── accuracy_metrics.csv
│       ├── dm_test.csv
│       ├── irf_data.csv
│       ├── robustness_lambda.csv
│       └── weights_*.csv
├── src/
│   ├── 01_data_engineering.py       # Data collection, IPI corrections, transformations
│   ├── 02_bvar_model.py             # Minnesota Prior B-VAR estimation and IRF
│   ├── 03_midas_model.py            # MIDAS with Exponential Almon weights
│   ├── 04_evaluate_results.py       # Accuracy metrics, DM test, figure generation
│   └── legacy/                      # Original monolithic pipeline scripts (archived)
├── outputs/
│   ├── metrics/                     # Accuracy tables (CSV)
│   ├── weights/                     # Almon weight profiles
│   └── figures/                     # Publication-ready PNG figures
├── paper/
│   ├── manuscript.docx              # Current working manuscript
│   └── *.tex                        # LaTeX appendix tables
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Replication Instructions

### 1. Clone the repository

```bash
git clone https://github.com/turdibaiakjol-cpu/bayesian-research.git
cd bayesian-research
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
# Step 1: Data engineering (IPI corrections, seasonal adjustment, log-differencing)
python src/01_data_engineering.py

# Step 2: Estimate B-VAR with Minnesota prior
python src/02_bvar_model.py

# Step 3: Estimate MIDAS with Almon weights
python src/03_midas_model.py

# Step 4: Evaluate, generate figures and accuracy tables
python src/04_evaluate_results.py
```

All outputs are saved to `outputs/figures/` and `outputs/metrics/`.

---

## Data Sources

| Source | Variables | Frequency |
|---|---|---|
| [Bureau of National Statistics (stat.gov.kz)](https://stat.gov.kz) | GDP, STEI, IPI, Freight Turnover | Quarterly / Monthly |
| [National Bank of Kazakhstan (nationalbank.kz)](https://nationalbank.kz) | USD/KZT, Aggregate Credit | Monthly |
| [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Brent Crude, Copper, Zinc, Uranium | Monthly |

Six deterministic IPI corrections were applied based on verified BNS Express Information publications. Full documentation is in `paper/data_appendix_table_A1.tex`.

---

## Key Metrics

| Model | Horizon | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| MIDAS | Nowcast | 0.0078 | 0.0059 | 18.3 |
| B-VAR | Nowcast | 0.0205 | 0.0131 | 45.5 |
| AR(1) | Nowcast | 0.0193 | 0.0111 | 36.3 |
| MIDAS | Backcast | 0.0063 | 0.0047 | 14.0 |
| B-VAR | Backcast | 0.0177 | 0.0114 | 40.0 |
| AR(1) | Backcast | 0.0176 | 0.0102 | 33.6 |

Diebold-Mariano test statistic: **2.423** (p = 0.0154), significant at the 5% level with Harvey-Leybourne-Newbold (1997) small-sample correction applied.

---

## Citation

If you use this code or data in academic work, please cite:

```
@unpublished{bayesian_research_2025,
  title   = {Nowcasting GDP in a Data-Constrained Commodity Economy:
             A Bayesian vs. Classical Comparative Framework for Kazakhstan},
  year    = {2025},
  note    = {Working paper}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
