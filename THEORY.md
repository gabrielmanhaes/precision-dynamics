# Theoretical Background and Relation to CANAL

## Core framework

This implementation builds on the free-energy principle (Friston 2010, 2013),
the REBUS model (Carhart-Harris and Friston 2019), and the CANAL framework
(Carhart-Harris et al. 2022).

The CANAL framework proposed two key constructs:
- **Canalization**: pathological precision entrenchment via Hebbian
  over-potentiation, narrowing the phenotypic state-space
- **TEMP**: Temperature or Entropy Mediated Plasticity — therapeutic
  precision reduction counteracting canalization via simulated annealing

This repository implements these constructs quantitatively.

## What CANAL does not include (this framework's extensions)

**Quantitative ODE implementation**
CANAL is a theoretical framework with static state-space representations.
This repository provides the dynamical implementation Carhart-Harris et al.
identified as needed.

**HRV as precision biomarker**
CANAL proposes measuring belief-precision change via subjective confidence
ratings (Zeifman et al. 2022). This framework proposes baseline HRV as an
objective, ambulatory, diagnostic-category agnostic proxy for canalization
depth.

**Bidirectional treatment selection**
CANAL does not make specific predictions about which patients should receive
which treatment. This framework predicts differential response based on
baseline HRV (canalization depth).

**REM sleep as endogenous TEMP**
CANAL does not address sleep. This framework proposes REM sleep as the
brain's scheduled endogenous TEMP mechanism — a nightly precision reduction
cycle preventing progressive canalization. Disruption of this cycle is
proposed as the mechanism of treatment resistance.

**P_selfmodel as consciousness amplitude**
CANAL treats precision as a single dimension. This framework decomposes it
into three hierarchical levels (P_sensory, P_conceptual, P_selfmodel) and
proposes P_selfmodel as the variable constituting conscious experience
amplitude.

## Direct conflict with CANAL: psychosis direction

CANAL proposes a single p-factor — all psychopathology as too-high precision
(canalization). This framework treats psychosis as a distinct failure mode:
P_conceptual insufficiently precise, causing high-level priors to fail to
weight down sensory prediction errors (aberrant salience).

Resolution: psychosis may represent canalization of sensory prediction errors
under insufficiently precise high-level priors. Different failure mode from
depression's rigid conceptual attractor, but not an exception to the general
principle that psychopathology involves pathological precision dynamics. The
hierarchical decomposition (P_sensory / P_conceptual / P_selfmodel) allows
both accounts to coexist: sensory-level canalization with conceptual-level
de-canalization.

## DMN Coherence Validation (held-out test)

P_selfmodel predicts DMN coherence ordering across 8 conditions without
refitting. Results: 5/7 direction matches. Key psilocybin dose-response
ordering correct (depression > healthy > low-dose > high-dose > ego
dissolution). Two failures: propofol (model lacks thalamocortical
suppression pathway — GABA parameter operates differently) and meditation
(model has no cognitive engagement reduction mechanism for P_selfmodel).
Spearman rho = 0.143 (weak; driven by propofol and meditation failures).

**Falsifiability criterion:** If DMN coherence does not decrease
monotonically with psychedelic dose across published neuroimaging
datasets, the P_selfmodel consciousness amplitude claim is disconfirmed.

**Propofol dissociation test:** Propofol and high-dose psilocybin produce
similar DMN coherence reductions (both ~0.45) but via different mechanisms:
- Psilocybin: TEMP mechanism — acute entropy followed by post-acute
  heterosynaptic plasticity window (therapeutic)
- Propofol: thalamocortical suppression — no TEMP, no plasticity window,
  no therapeutic effect
This dissociation is the key falsifiability test: similar P_selfmodel
reduction, different post-acute outcomes, different mechanisms.

**Honest assessment:** The current model does not capture propofol or
meditation effects on P_selfmodel correctly. These would require additional
ODE pathways (thalamocortical suppression for propofol, cognitive
engagement modulation for meditation) not currently implemented.
