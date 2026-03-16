# Precision Dynamics Framework for Computational Psychiatry

A coupled ordinary differential equation model of neuromodulator-mediated precision regulation, implementing a unified computational account of sleep architecture, psychedelic pharmacology, and psychiatric conditions.

## Overview

This repository contains the ODE model described in:

> Manhães de Souza, G. (2026). A Unified Precision Dynamics Framework for Computational Psychiatry: Mechanism Fixes, Clinical Predictions, and Cross-Domain Synthesis.

The model fits 19 empirical targets across six measurement domains (sleep architecture, psychedelic pharmacology, psychiatric conditions, stress physiology, autonomic function, neuroplasticity) using 9 fitted parameters, achieving MAPE=9.9%, Train R²=0.9509, Test R²=0.9601.

## Key Results

- **Model fit:** 9 fitted parameters, 19 empirical targets, constraint ratio 2.11:1
- **Three mechanism fixes:** Psilocybin cortisol direction, REM alpha magnitude, NREM LZW ordering
- **Blind predictions:** 7/8 directional correct (88%)
- **Confirmed clinical prediction:** mPFC metabolic rate predicts sleep deprivation antidepressant response (Wu 1992/1999, Ebert 1991, Clark 2006 — 5 independent groups)
- **Structural prediction:** Atomoxetine and psilocybin produce 6/6 opposing biomarker signatures from opposing ODE mechanisms

## Installation

```bash
git clone https://github.com/gabrielmanhaes/precision-dynamics
cd precision-dynamics
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib
- pandas

## Quick Start

```python
from model import simulate_v2
from validation_summary import BEST_FIT

# Run 24-hour simulation with validated parameters
results = simulate_v2(BEST_FIT)

# Results contains:
# - neuromodulator time courses (NE, 5HT, ACh, DA)
# - precision dynamics (P_sensory, P_conceptual, P_selfmodel)
# - biomarker predictions (alpha, LZW, HRV, cortisol, BDNF)
```

## Repository Structure

```
precision-dynamics/
├── model.py          # Core ODE model
├── parameters.py             # All 163 parameters with biological sources
├── validation_summary.py     # Definitive 9-parameter validated set (BEST_FIT)
├── fitting_v2.py             # Differential evolution optimization
├── fitting_v3.py             # Extended fitting variant
├── paper_figures.py          # Figure generation for manuscript
├── sensitivity_analysis.py   # Parameter sensitivity analysis
├── sim_atomoxetine.py        # Atomoxetine simulation scenarios
├── sim_hypotheses.py         # Hypothesis testing simulations
├── sim_sleep_profiles.py     # Sleep quality moderator simulations
├── sim_microdose_depression.py  # Microdosing protocol comparison
└── sim_sensory_deprivation.py   # Float tank simulations
```

## Reproducing Key Results

### Model validation
```bash
python fitting_v2.py --validate
```

### Generate paper figures
```bash
python paper_figures.py --output figures/paper/
```

### Run blind prediction tests
```bash
python validation_summary.py --blind-predictions
```

## Parameter Sets

The definitive parameter set is `BEST_FIT` in `validation_summary.py`:
- 9 fitted parameters (differential evolution, seed=42)
- 154 motivated parameters (biological literature)
- 12 phantom parameters (<1% impact on all targets, documented)
- dt=0.1 hours required for valid results

## Known Limitations

- ATX alpha operationalization disconfirmed (NE suppresses thalamocortical alpha via burst-to-tonic shift not captured by P^exponent mapping)
- 12 phantom parameters could be removed without affecting predictions
- 4 partially tautological targets (direction forced by operationalization sign conventions)
- Sloppy model eigenspectrum: condition number 1.66×10^35 (consistent with Brown & Sethna 2003 biophysical model class)
- PTSD_DISSOC_COEFF shows 30.6% seed instability
- dt=0.1 required; dt=0.5 degrades phasic coupling

## Citation

```bibtex
@preprint{lastname2026precision,
  title={A Unified Precision Dynamics Framework for Computational Psychiatry},
  author={Manh\~{a}es de Souza, Gabriel},
  year={2026},
  note={bioRxiv preprint}
}
```

## License

MIT License — see LICENSE file.

## Contact

Gabriel Manhães de Souza — gabriel@manhaes.dev
