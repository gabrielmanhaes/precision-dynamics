# Precision Dynamics

A quantitative ODE implementation of the CANAL framework
(Carhart-Harris et al. 2022, *Neuropharmacology* 226, 109398)

> "Dynamical methods will be required to better capture the emergence of
> different free-energy minimizing processes."
> — Carhart-Harris et al. 2022, Box 1

---

## What this is

A coupled ordinary differential equation system implementing the canalization
and TEMP framework of Carhart-Harris et al. (2022) quantitatively. The model
is fitted against 19 empirical targets from published literature spanning sleep
architecture, psychedelic pharmacology, psychiatric biomarkers, stress
physiology, and autonomic function.

**Fit statistics (9 free parameters, 19 targets):**
- Train R² = 0.9509 (11 targets)
- Test R² = 0.9601 (8 held-out targets)
- MAPE = 9.9%

---

## Terminology mapping

| This repository | CANAL (Carhart-Harris et al. 2022) |
|----------------|-----------------------------------|
| Attractor deepening | Canalization |
| Precision reduction event | TEMP |
| Plasticity maintenance cycle | Endogenous TEMP via REM sleep |
| High-P state (pathological) | Canalized state |
| P_sensory / P_conceptual / P_selfmodel | Single precision dimension (extended here to three hierarchical levels) |

---

## Two gaps this addresses

**Gap 1 — Dynamical implementation (Box 1):**
The CANAL paper notes that state-space representations "remain static images"
and that "dynamical methods will be required." This repository provides a
running ODE implementation producing numerical predictions against published
empirical targets.

**Gap 2 — Precision biomarker (Section 1.15):**
The CANAL paper identifies the failure of RDoC biomarker discovery as stemming
from cross-sectional measurement of symptom outputs rather than the underlying
precision variable. This framework proposes baseline HRV as a candidate
biomarker that is continuous, ambulatory, and diagnostic-category agnostic —
directly measuring the autonomic correlate of precision level.

---

## Key results

**1. Model fit**
The model achieves R²=0.9601 on 8 held-out empirical targets not used in
fitting, with 9 free parameters against 19 total targets (constraint ratio
2.11:1). Zero parameters at bounds.

**2. Bidirectional treatment prediction**
Atomoxetine (NE reuptake inhibitor, increases P) and psilocybin (5-HT2A
agonist, decreases P) produce opposing biomarker signatures across 6/6
dynamical variables, emerging from opposing ODE mechanisms without hardcoded
directions.

**3. HRV treatment selection prediction**
Baseline HRV operationalizes canalization depth. High-HRV depression (lower
canalization, sufficient residual plasticity) is predicted to respond better
to psilocybin (TEMP event can restructure attractors). Low-HRV depression
(deeper canalization) is predicted to respond better to SSRIs (gradual
precision recalibration). This prediction is testable in existing clinical
datasets.

**4. P_selfmodel as consciousness amplitude**
P_selfmodel is proposed as the variable constituting conscious experience
amplitude. Ego dissolution marks P_selfmodel approaching zero, explaining
why mystical experience depth predicts therapeutic outcome in psilocybin
trials. DMN coherence is the predicted neural proxy, testable against
existing neuroimaging datasets without model refitting.

**5. Robustness to parameter uncertainty**
The 3/3 bidirectional ATX vs psilocybin dynamical prediction (P_conceptual,
EEG alpha, LZW complexity) was tested across 1000 joint parameter
perturbations (±30% all 9 parameters simultaneously). 100% of perturbations
maintained correct directional predictions across all 3 dynamical biomarkers,
confirming the result is structurally robust to the parameter uncertainty
implied by the sloppy model eigenspectrum.

---

## Known limitations

- All validation against published literature — no new empirical data collected
- 7/9 parameters not individually identifiable (sloppy model eigenspectrum)
- Atomoxetine alpha operationalization disconfirmed (NE→thalamocortical
  suppression pathway missing)
- Psychosis characterization requires hierarchical precision account
  (P_conceptual low, not globally low-P)
- Model predicts population means, not individual responses
- No prospective empirical validation

---

## How to run
```bash
pip install numpy scipy matplotlib
python run.py
```

Requires dt=0.1h for accurate phasic neuromodulator dynamics. Coarser
timesteps degrade phasic coupling parameters.

---

## Methodology note

This framework was developed using AI-assisted literature synthesis by
an independent researcher with no formal neuroscience training. All
mechanisms were verified against primary literature after identification.
See `docs/methodology_statement.md` for full details.

The bidirectional HRV treatment prediction is the key empirical test.
It is testable in existing clinical trial datasets without new data
collection.

---

## Relation to CANAL

This repository is an extension of, not a replacement for, the CANAL
framework. The conceptual core — precision as the key variable, canalization
as pathological entrenchment, TEMP as therapeutic mechanism — is Carhart-Harris
et al. (2022). The contributions here are: quantitative implementation, HRV
operationalization, bidirectional treatment selection prediction, REM sleep as
endogenous TEMP, and the P_selfmodel consciousness amplitude extension.

---

## Citation

Carhart-Harris, R.L. et al. (2022). Canalization and plasticity in
psychopathology. *Neuropharmacology*, 226, 109398.
https://doi.org/10.1016/j.neuropharm.2022.109398
