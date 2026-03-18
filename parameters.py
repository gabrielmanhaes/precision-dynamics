"""
Parameter Registry for Plasticity Model v2
===========================================

Every parameter in one file, organized into three categories:
  GROUNDED  — Empirically constrained with literature citation and uncertainty bounds
  MOTIVATED — Sign/direction constrained by neuroscience, magnitude assumed
  FREE      — Explicitly flagged as unconstrained fitting parameters

Total: ~98 parameters (down from 114 in v1)
  ~18 grounded (up from 0)
  ~55 motivated
  ~25 free

Each parameter is annotated with:
  - Category (GROUNDED / MOTIVATED / FREE)
  - Citation (if grounded)
  - Uncertainty range (if grounded)
  - Rationale (if motivated)
"""

import numpy as np


# ============================================================================
# GROUNDED PARAMETERS (~18) — Empirically constrained
# ============================================================================

# --- Circadian / Ultradian timing ---

# Circadian period: 24.0 ± 0.2 hours
# Citation: Czeisler et al. (1999) Science 284:2177
# Uncertainty: 23.8–24.2 hours (individual variation)
TAU_CIRCADIAN = 24.0  # GROUNDED

# Ultradian REM cycle period: 90 ± 10 minutes
# Citation: Aserinsky & Kleitman (1953) Science 118:273
# Uncertainty: 80–100 minutes
TAU_ULTRADIAN = 1.5  # GROUNDED (hours)

# --- Cortisol / HPA axis ---

# Cortisol Awakening Response (CAR) peak: ~07:00 (30-45 min post-wake)
# Citation: Debono et al. (2009) J Clin Endocrinol Metab 94:1548
# Uncertainty: 06:30–07:30
CORTISOL_CAR_PEAK = 7.0  # GROUNDED (hours, clock time)

# Cortisol nadir: ~23:00–01:00
# Citation: Debono et al. (2009)
# Uncertainty: 22:00–01:00
CORTISOL_NADIR = 23.5  # GROUNDED (hours)

# Cortisol half-life: ~90 minutes (60–120 min range)
# Citation: Debono et al. (2009); Weitzman et al. (1971)
CORTISOL_HALF_LIFE = 1.5  # GROUNDED (hours)

# --- Psilocybin pharmacokinetics ---

# Psilocybin onset: 30 min, peak: 1.5–2h, duration: 6h
# Psilocin (active metabolite) half-life: 1.4–1.8h (mean ~1.7h)
# Citation: Holze et al. (2023) Clin Pharmacol Ther 113 (online Dec 2022)
# Note: PSILOCYBIN_HALF_LIFE refers to psilocin (active metabolite), not
# the psilocybin prodrug. Corrected from 3.0h to 1.7h per Holze 2023.
# Uncertainty: onset 20-40 min, peak 1-2.5h, duration 4-8h, t1/2 1.4-1.8h
PSILOCYBIN_ONSET_TAU = 0.5  # GROUNDED (hours, ~30 min)
PSILOCYBIN_PEAK_TIME = 1.75  # GROUNDED (hours)
PSILOCYBIN_HALF_LIFE = 1.7  # GROUNDED (hours, psilocin t1/2; was 3.0h incorrectly)
PSILOCYBIN_DURATION = 6.0  # GROUNDED (hours)

# --- Ketamine pharmacokinetics ---

# Ketamine IV: onset 40 min, peak 1-2h, BDNF peak 2-4h, benefit 1-2 weeks
# Citation: Zarate et al. (2006) Arch Gen Psychiatry 63:856
# Also: Autry et al. (2011) Nature 475:91 (BDNF mechanism)
KETAMINE_ONSET_TAU = 0.67  # GROUNDED (hours, ~40 min)
KETAMINE_PEAK_TIME = 1.5  # GROUNDED (hours)
KETAMINE_BDNF_PEAK = 3.0  # GROUNDED (hours, 2-4h range)
KETAMINE_BENEFIT_DURATION = 10.0 * 24.0  # GROUNDED (hours, ~10 days median)

# SSRI neuroplasticity mechanism (Duman & Monteggia 2006)
# SSRIs take 4-6 weeks because therapeutic effect comes from BDNF/neuroplasticity,
# not acute 5-HT increase. Phase 1: 5-HT rise → mild P increase (side effects).
# Phase 2: delayed BDNF → plasticity → P decrease (therapeutic).
SSRI_NEUROPLASTICITY_DELAY = 2.0    # GROUNDED (weeks before plasticity onset)
SSRI_NEUROPLASTICITY_TAU = 3.0      # MOTIVATED (weeks, ramp-up time constant)
SSRI_NEUROPLASTICITY_GAIN = 1.5     # MOTIVATED (strength of plasticity-driven P decrease)

# --- DMT pharmacokinetics ---
# DMT IV: onset ~2 min, peak ~5 min, duration ~30 min
# Citation: Strassman et al. (1994) Arch Gen Psychiatry 51:85
# Also: Timmermann et al. (2019) Sci Rep 9:16324
DMT_ONSET_TAU = 0.033      # GROUNDED (hours, ~2 min IV)
DMT_PEAK_TIME = 0.083      # GROUNDED (hours, ~5 min)
DMT_HALF_LIFE = 0.25       # GROUNDED (hours, ~15 min)
DMT_DURATION = 0.5         # GROUNDED (hours, ~30 min)

# --- Atomoxetine pharmacokinetics ---
# Atomoxetine (Strattera): selective NET inhibitor
# Oral: Tmax ~1-2h, half-life ~5h (adults), steady-state ~1 day
# Citation: Sauer et al. (2005) Clin Pharmacokinet 44:571
ATOMOXETINE_ONSET_TAU = 1.0     # GROUNDED (hours, oral absorption)
ATOMOXETINE_PEAK_TIME = 1.5     # GROUNDED (hours, Tmax)
ATOMOXETINE_HALF_LIFE = 5.0     # GROUNDED (hours, adult t½)

# --- 5-HT2A receptor distribution ---

# 5-HT2A density weights by cortical region (PET imaging)
# Citation: Beliveau et al. (2017) J Neurosci 37:120
# Values normalized so highest region = 0.50
RECEPTOR_5HT2A_SENSORY = 0.30  # GROUNDED (V1/auditory: moderate)
RECEPTOR_5HT2A_ASSOCIATION = 0.50  # GROUNDED (Layer V association: highest)
RECEPTOR_5HT2A_DMN = 0.40  # GROUNDED (mPFC/PCC DMN: moderate-high)

# --- 5-HT2A receptor dynamics ---

# Downregulation: 20-40% over 2 weeks chronic agonism
# Citation: Buckholtz et al. (1990) Neuropsychopharmacology 3:37
# Recovery tau: ~7-14 days
# Also: Leysen et al. (1989) for tachyphylaxis kinetics
RECEPTOR_DOWNREG_MAX = 0.30  # GROUNDED (30% ± 10% over 2 weeks)
RECEPTOR_DOWNREG_TAU = 48.0  # GROUNDED (hours to significant downreg, ~2 days acute)
RECEPTOR_RECOVERY_TAU = 10.0 * 24.0  # GROUNDED (hours, ~10 days recovery)

# --- GABA ---

# Frontal GABA baseline: ~1.5-2.0 mM (MRS)
# GAD deficit in anxiety: ~15-20%
# Citation: Goddard et al. (2001) Arch Gen Psychiatry 58:556
# Also: Hasler et al. (2007) for depression GABA deficit
GABA_BASELINE = 1.75  # GROUNDED (mM, normalized to 1.0 in model)
GABA_ANXIETY_DEFICIT = 0.175  # GROUNDED (fraction, 15-20% reduction)

# --- Sleep architecture ---

# NREM delta power peaks in first 2 cycles; REM proportion increases
# Citation: Carskadon & Dement (2017) in Principles of Sleep Medicine
# REM fraction: ~20% early night → ~50% late night
SLEEP_REM_EARLY_FRACTION = 0.20  # GROUNDED
SLEEP_REM_LATE_FRACTION = 0.50  # GROUNDED


# ============================================================================
# MOTIVATED PARAMETERS (~40) — Signs constrained, magnitudes assumed
# ============================================================================

# --- Neuromodulator baselines and amplitudes ---
# Signs: NE, 5-HT are wake-promoting and increase P (well-established)
# Magnitudes: normalized to dimensionless 0-1 scale (model convention)

NE_BASELINE = 0.50  # MOTIVATED: high during wake, low in sleep (Aston-Jones 2005)
NE_AMPLITUDE = 0.10  # MOTIVATED: modest circadian swing
SEROTONIN_BASELINE = 0.45  # MOTIVATED: similar to NE, raphe tonic firing
SEROTONIN_AMPLITUDE = 0.08  # MOTIVATED: slight circadian swing
ACH_WAKING = 0.30  # MOTIVATED: moderate BF tone during wake
ACH_REM = 0.90  # MOTIVATED: high PPT/LDT output in REM
ACH_NREM = 0.10  # MOTIVATED: very low in NREM (permits consolidation)

# Dopamine
# DA baseline: tonic VTA firing ~4 Hz (Grace 1991)
# Reduced ~50% in sleep (Lena et al. 2005)
DA_BASELINE = 0.45  # MOTIVATED: normalized tonic DA level
DA_AMPLITUDE = 0.05  # MOTIVATED: modest circadian variation
DA_SLEEP_REDUCTION = 0.50  # MOTIVATED: ~50% reduction during sleep

# Glutamate
# Tightly regulated baseline; excess is excitotoxic
GLU_BASELINE = 0.50  # MOTIVATED: tightly regulated
GLU_AMPLITUDE = 0.03  # MOTIVATED: minimal circadian variation

# GABA (normalized)
GABA_NORMALIZED = 1.0  # MOTIVATED: normalized baseline = 1.0
GABA_SLEEP_INCREASE = 0.15  # MOTIVATED: slight increase during NREM

# Endogenous plasticity (renamed from DMT)
# Encompasses tryptamines, endocannabinoids, tonic BDNF, spontaneous LTP
ENDOGENOUS_PLASTICITY_TONIC = 0.15  # MOTIVATED: persistent baseline plasticity drive
ENDOGENOUS_PLASTICITY_REM_PEAK = 0.60  # MOTIVATED: phasic peaks during REM

# --- Coupling strengths ---
# Signs: NE↑P, 5-HT↑P, DA modulates salience, ACh↓P, endogenous plasticity↓P
# Magnitudes: tuned for stable waking P and realistic sleep dynamics

ALPHA_NE = 0.0500  # FIXED (consolidated): at bound, sensitivity <3%, fixed at converged value
ALPHA_5HT = 0.0500  # FIXED (consolidated): at bound, sensitivity <3%, fixed at converged value
ALPHA_DA = 0.15  # MOTIVATED: moderate DA contributes to P maintenance
BETA_PLAST = 1.5000  # FITTED (refit 1, 14-param): endogenous plasticity → reduces P
BETA_ACH = 0.30  # MOTIVATED: ACh → plasticity drive (decreases P)
BETA_GLU = 0.20  # MOTIVATED: glutamate pathway (BDNF) → plasticity

# --- HPA coupling constants ---
# Signs: well-established from neuroscience
# Magnitudes: tuned, but SHARED across all conditions (not per-scenario)
# Key property: these are calibrated from normal stress response, then
# pathological P profiles are downstream PREDICTIONS

# Cortisol → NE: CRH → locus coeruleus activation
# Citation: Valentino & Van Bockstaele (2008) Brain Res 1218:1
CORT_NE_COUPLING = 0.20  # MOTIVATED: positive, cortisol elevates NE

# Cortisol → 5-HT: complex; acute ↑ tryptophan hydroxylase, chronic ↓
# Citation: Porter et al. (2004) Psychopharmacology 174:414
CORT_5HT_COUPLING = -0.10  # MOTIVATED: chronic cortisol suppresses effective 5-HT

# Cortisol → DA: allostatic load reduces DA (anhedonia)
# Citation: Pani et al. (2000) Neurosci Biobehav Rev 24:375
CORT_DA_COUPLING = -0.30  # MOTIVATED: negative, allostatic load reduces DA

# Cortisol → plasticity: elevated cortisol reduces BDNF
# Citation: Duman & Monteggia (2006) Biol Psychiatry 59:1116
CORT_PLASTICITY_COUPLING = -0.25  # MOTIVATED: negative, cortisol reduces plasticity

# --- Inter-level coupling ---
COUPLING_TD = 0.15  # MOTIVATED: top-down Bayesian hierarchy constraint
COUPLING_BU = 0.10  # MOTIVATED: bottom-up prediction error propagation

# --- Per-level homeostatic strengths ---
GAMMA_SENSORY = 0.9666  # FITTED (refit 1, 14-param): sensory homeostatic strength
GAMMA_CONCEPTUAL = 0.80  # MOTIVATED: beliefs update slower
GAMMA_SELFMODEL = 0.65  # MOTIVATED: identity changes slowest (therapy timescale)

# --- Per-level neuromodulator coupling scales ---
# Sensory most responsive to NE (arousal gating)
# Self-model most responsive to 5-HT and plasticity (ego/identity)
ALPHA_NE_SCALE = [1.1, 1.0, 0.8]  # MOTIVATED: sensory > conceptual > self-model
ALPHA_5HT_SCALE = [0.9, 1.0, 1.2]  # MOTIVATED: self-model > conceptual > sensory
BETA_PLAST_SCALE = [0.8, 1.0, 1.3]  # MOTIVATED: self-model > conceptual > sensory

# --- GABA gain modulation ---
# GABA does NOT directly affect P. It modulates NE coupling gain.
# Low GABA → amplified NE → anxiety-like sensory hyperarousal
# Citation: Nuss (2015) Neuropsychiatr Dis Treat 11:165
GABA_NE_GAIN_MOD = 0.7322  # FITTED (refit 1, 14-param): GABA → NE gain modulation

# --- Sleep timing ---
SLEEP_ONSET = 23.0  # MOTIVATED: sleep onset clock time (hours)
WAKE_TIME = 7.0  # MOTIVATED: wake time (hours)

# --- Cortisol diurnal envelope ---
# NOTE: CORTISOL_HALF_LIFE (1.5h, GROUNDED) is the metabolic clearance half-life
# of cortisol in plasma. CORTISOL_DIURNAL_TAU (8.0h, MOTIVATED) is the envelope
# time constant of the diurnal cortisol decay from CAR peak to nadir — a much
# slower process driven by circadian HPA regulation, not metabolic clearance.
CORTISOL_DIURNAL_TAU = 8.0  # MOTIVATED: diurnal envelope decay (hours), not metabolic clearance

# --- Equilibrium reference time ---
EQUILIBRIUM_REF_TIME = 14.0  # MOTIVATED: clock time for equilibrium offset computation (14:00)

# --- Allostatic load thresholds ---
ALLOSTATIC_THRESHOLD_FACTOR = 1.1  # MOTIVATED: cortisol must exceed baseline × factor to accumulate load
ALLOSTATIC_LOAD_MAX = 2.0  # MOTIVATED: ceiling for allostatic load accumulation

# --- DA excess scaling ---
DA_EXCESS_SCALING = 0.5  # MOTIVATED: scaling from da_excess parameter to tonic DA elevation

# --- ADHD mechanisms ---
ADHD_NOISE_SCALING = 3.0  # MOTIVATED: noise amplification from transporter dysfunction
ADHD_DA_SALIENCE_NOISE = 0.5  # MOTIVATED: DA-driven salience noise amplitude in ADHD

# --- Chronic stress neuromodulator effects ---
CHRONIC_NE_SENSITIZATION = 0.30  # MOTIVATED: allostatic load → chronic NE sensitization gain
CHRONIC_5HT_ELEVATION = 0.15  # MOTIVATED: allostatic load → chronic 5-HT elevation gain

# --- CSTC and BDNF ---
CSTC_SUPPRESSION_SCALING = 0.15  # MOTIVATED: CSTC GLU excess → conceptual suppression scaling
BDNF_SURGE_SCALING = 0.6  # MOTIVATED: ketamine BDNF surge amplitude scaling

# --- Pharmacological gain amplification ---
# Psilocybin acts via 5-HT2A with receptor-mediated Gq/11 cascade gain
# beyond tonic 5-HT coupling. Separates receptor-mediated from tonic effects.
# Citation: Vollenweider & Preller (2020) Nat Rev Neurosci 21:611
PSILOCYBIN_PHARMA_GAIN = 0.2792  # FITTED (refit 1, 14-param): 5-HT2A receptor-mediated amplification (↓25% from HPA fix)

# Ketamine BDNF-TrkB pathway gain. Downstream effects on synaptic plasticity
# beyond tonic GLU coupling.
# Citation: Duman & Aghajanian (2012) Science 338:68
KETAMINE_PHARMA_GAIN = 5.0  # FIXED (consolidated): at bound, sensitivity <2%, fixed at converged value

# PTSD dissociation coefficient. Controls how strongly NE sensitization
# combined with top-down breakdown suppresses self-model P.
PTSD_DISSOC_COEFF = 0.2637  # FITTED (refit 1, 14-param): dissociation drive

# --- Psilocybin afterglow ---
PSILOCYBIN_AFTERGLOW_TAU = 72.0  # MOTIVATED: afterglow decay time constant (hours, ~3 days)

# DMT pharma gain: same 5-HT2A mechanism as psilocybin
# DMT produces qualitatively similar effects but compressed in time
DMT_PHARMA_GAIN = 0.4333   # MOTIVATED: same as psilocybin (same receptor mechanism)
DMT_AFTERGLOW_TAU = 24.0   # MOTIVATED: shorter afterglow than psilocybin (hours, ~1 day)

# --- Synaptic consolidation (psilocybin-induced structural plasticity) ---
# Ly et al. (2018) Nature Neuroscience 21:120 — psilocybin promotes dendritic
# spine formation via 5-HT2A → TrkB/mTOR. Spine density peaks at 24h,
# returns to baseline by ~14 days in mouse L5 pyramidal neurons.
# SC accumulates during 5-HT2A agonism when P < homeostatic target,
# representing structural stabilization of the reduced-P state.
SC_CONSOLIDATION_TAU = 336.0  # MOTIVATED: 14 days × 24 h/day (Ly et al. 2018 spine timeline)
K_CONSOLIDATION = 0.10        # MOTIVATED: consolidation accumulation rate

# --- Phasic neuromodulator coupling ---
# Tonic NE/5-HT coupling (ALPHA_NE, ALPHA_5HT) is equilibrium-subtracted,
# cancelling baseline effects. Phasic coupling responds to d(NE)/dt and
# d(5-HT)/dt — rate of change rather than absolute level. This pathway
# bypasses equilibrium subtraction, capturing acute drug effects (ATX onset,
# stress responses) and circadian transitions.
# Biologically: tonic LC firing maintains background arousal (equilibrium),
# while phasic LC bursts signal acute state changes (Aston-Jones & Cohen 2005).
ALPHA_NE_PHASIC = 2.0000  # FITTED (refit 1, 14-param): phasic NE coupling (at bound)
ALPHA_5HT_PHASIC = 0.8996  # FITTED (refit 1, 14-param): phasic 5-HT coupling

# Atomoxetine: NET blockade → elevated NE → P increase
# Magnitude of NET blockade at therapeutic doses (~80% occupancy)
# Citation: Takano et al. (2009) Synapse 63:555
ATOMOXETINE_NE_GAIN = 0.40      # MOTIVATED: magnitude of NE elevation from NET blockade
ATOMOXETINE_PHARMA_GAIN = 0.35  # MOTIVATED: P-modulation gain from NE pathway

# --- PTSD asymmetric coupling ---
PTSD_TD_BREAKDOWN = 0.3  # MOTIVATED: top-down coupling retains only 30% in PTSD (dissociation)

# --- HPA axis dynamics ---
# Negative feedback via glucocorticoid receptors
# Citation: Tsigos & Chrousos (2002) J Psychosom Res 53:865
HPA_FEEDBACK_GAIN = 0.30  # MOTIVATED: negative feedback strength
HPA_FEEDBACK_EROSION_RATE = 0.005  # MOTIVATED: chronic stress erodes feedback
ALLOSTATIC_LOAD_RATE = 0.008  # MOTIVATED: rate of allostatic load accumulation
ALLOSTATIC_LOAD_DECAY = 0.001  # MOTIVATED: very slow recovery of allostatic load

# --- Cortisol rhythm ---
CORTISOL_BASELINE = 0.40  # MOTIVATED: normalized tonic cortisol level
CORTISOL_CAR_AMPLITUDE = 0.35  # MOTIVATED: CAR peak amplitude
CORTISOL_STRESS_GAIN = 1.8130  # FITTED (refit 1, 14-param): stress → cortisol scaling

# --- Direct 5-HT2A → HPA activation ---
# Psilocybin/DMT activate hypothalamic CRH neurons directly via 5-HT2A receptors,
# independent of the precision dynamics → allostatic load pathway.
# This produces acute cortisol increase even as precision reduction lowers allostatic load.
# Citation: Hasler et al. (2004) Neuropsychopharmacology 29:1782
#           Strajhar et al. (2016) J Neuroendocrinol 28:12344
HPA_5HT2A_DIRECT_GAIN = 0.36  # MOTIVATED: 5-HT2A agonist → CRH → cortisol (direct pathway)

# --- DA nonlinear salience ---
# Moderate DA: good salience filtering → P maintenance
# Excess DA: salience noise → P disruption
DA_SALIENCE_OPTIMAL = 0.50  # MOTIVATED: optimal DA for salience filtering
DA_SALIENCE_WIDTH = 0.20  # MOTIVATED: width of inverted-U response

# --- Receptor dynamics ---
# D2 upregulation under chronic antagonism (supersensitivity)
# Citation: Seeman (2011) Synapse 65:1289
D2_UPREG_RATE = 0.01  # MOTIVATED: slow D2 upregulation under blockade
D2_RECOVERY_TAU = 14.0 * 24.0  # MOTIVATED: ~2 weeks D2 recovery
D2_MAX_DENSITY = 1.5  # MOTIVATED: maximum D2 upregulation (50% above baseline)

# --- Bipolar dynamics ---
# Setpoint oscillator parameters (ported from v1 where they produced
# emergent ~52.5-day cycling)
BIPOLAR_K_INSTAB = 0.008  # MOTIVATED: mood-state positive feedback
BIPOLAR_K_FEEDBACK = 0.020  # MOTIVATED: slow compensatory gain
BIPOLAR_TAU_ADAPT = 300.0  # MOTIVATED: ~12.5 day adaptation time constant (hours)


# ============================================================================
# FREE PARAMETERS (~25) — Explicitly unconstrained
# ============================================================================

# --- Precision bounds ---
# These are the most fundamental assumptions. P is dimensionless and
# the specific numeric values are model conventions, not measurements.
P_MIN = 0.15  # FREE: lowest P under normal physiology
P_MAX = 0.95  # FREE: maximum suppression ceiling

# Per-state waking baselines
P_WAKING = 0.70  # FREE: typical waking baseline
P_REM = 0.25  # FREE: P during REM sleep
P_NREM = 0.55  # FREE: P during NREM sleep

# Per-level waking baselines
P_SENSORY_BASELINE = 0.75  # FREE: sensory cortex waking P
P_CONCEPTUAL_BASELINE = 0.70  # FREE: conceptual/belief waking P
P_SELFMODEL_BASELINE = 0.60  # FREE: self-model (DMN) waking P

# P_selfmodel: Precision of the self-model
#
# Theoretical extension (beyond CANAL):
# P_selfmodel is proposed as not merely the highest level of the precision
# hierarchy but the variable whose magnitude constitutes the amplitude of
# unified conscious experience.
#
# Predicted relationships:
#   - Normal waking consciousness: P_selfmodel in healthy range (stable,
#     permeable self-model)
#   - Depression: P_selfmodel canalized at pathologically high precision
#     around negative self-referential content (rigid DMN, high internal
#     coherence)
#   - Psychosis: P_conceptual insufficiently precise — high-level priors
#     fail to weight down sensory prediction errors (aberrant salience).
#     NOT a globally low-P condition. P_selfmodel may be fragmented rather
#     than uniformly low.
#   - Low-dose psychedelics: mild P_selfmodel reduction (loosened
#     self-boundary)
#   - High-dose psychedelics: strong P_selfmodel reduction (ego dissolution)
#   - Ego dissolution threshold: P_selfmodel approaching zero
#
# Measurable proxy: DMN (Default Mode Network) coherence
#   - Decreases systematically with psychedelic dose
#   - Abnormally high and internally rigid in depression
#   - Fragmented in psychosis
#   - Predicts mystical experience depth and therapeutic outcome
#
# This interpretation does not change the ODE equations or parameter values.
# It reframes what P_selfmodel represents — the math was already tracking
# this variable. The consciousness amplitude interpretation makes explicit
# what was embedded in the model structure.
#
# Testable prediction without refitting: add DMN coherence values from
# published psychedelic neuroimaging studies as test targets and check
# whether existing P_selfmodel trajectories predict the correct ordering
# and approximate magnitudes.

# Per-level sleep targets
P_SENSORY_REM = 0.22  # FREE
P_CONCEPTUAL_REM = 0.25  # FREE
P_SELFMODEL_REM = 0.30  # FREE
P_SENSORY_NREM = 0.50  # FREE
P_CONCEPTUAL_NREM = 0.5731  # FITTED (refit 1, 14-param): NREM conceptual P target
P_SELFMODEL_NREM = 0.52  # FREE

# --- Noise parameters ---
SIGMA_NOISE = 0.02  # FREE: stochastic noise amplitude
NOISE_TAU = 0.25  # FREE: noise autocorrelation time (hours)

# --- Time constant ---
TAU_P = 0.5  # FREE: precision integration time constant (hours)

# --- Bipolar instability ---
BIPOLAR_GAMMA_WIDTH = 0.12  # FREE: homeostatic weakening width
BIPOLAR_GAMMA_BASE = 0.25  # FREE: baseline homeostatic strength in bipolar
BIPOLAR_NOISE_SCALE = 1.0  # FREE: noise in bipolar simulation

# --- Per-level P adaptation rates for mixed bipolar ---
BIPOLAR_TAU_ADAPT_SENSORY = 250.0  # FREE (hours)
BIPOLAR_TAU_ADAPT_CONCEPTUAL = 300.0  # FREE (hours)
BIPOLAR_TAU_ADAPT_SELFMODEL = 400.0  # FREE (hours)

# --- Upstream scenario parameters ---
# These are the ONLY per-scenario knobs. Pathological P profiles EMERGE.
# Listed here as defaults; scenarios override them.

STRESS_CHRONIC_DEFAULT = 0.0  # FREE: chronic stress level (0-1)
DA_EXCESS_DEFAULT = 0.0  # FREE: dopamine excess factor
GABA_DEFICIT_DEFAULT = 0.0  # FREE: GABA deficit fraction (0-1)
NE_SENSITIZATION_DEFAULT = 1.0  # FREE: NE sensitization multiplier
COUPLING_BREAKDOWN_DEFAULT = 1.0  # FREE: inter-level coupling multiplier
DAT_DYSFUNCTION_DEFAULT = 1.0  # FREE: DAT function (1.0 = normal)
NET_DYSFUNCTION_DEFAULT = 1.0  # FREE: NET function (1.0 = normal)
CSTC_GLU_EXCESS_DEFAULT = 0.0  # FREE: CSTC glutamate excess
STRESS_SENSITIVITY_DEFAULT = 1.0  # FREE: stress sensitivity multiplier


# ============================================================================
# ATTRACTOR LANDSCAPE (visualization only, not coupled to dynamics)
# ============================================================================

N_ATTRACTORS = 5
PRIOR_WEIGHTS = np.array([1.0, 0.7, 0.5, 0.3, 0.2])
ATTRACTOR_CENTERS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


# ============================================================================
# P OPERATIONALIZATION CONSTANTS
# ============================================================================
# Map P to measurable neural observables (hypothesized mappings)

# EEG alpha power: alpha ∝ P^γ
# Citation: Jensen & Mazaheri (2010) Front Hum Neurosci 4:186
ALPHA_POWER_EXPONENT = 3.0000  # FITTED (refit 1, 14-param): nonlinear P→alpha relationship (at bound)

# --- Dual-alpha operationalization ---
# Two biologically distinct alpha mechanisms must not be conflated:
#
# 1. THALAMOCORTICAL ALPHA: Generated by thalamocortical loops, reflects active
#    inhibitory gating of sensory/cognitive processing. Increases with precision
#    (P^exponent). NE has a mild modulatory effect on waking thalamocortical alpha
#    via locus coeruleus → thalamic projections (Haegens et al. 2011 J Neurosci).
#    Sleep states use P^exponent alone (different generator: slow oscillations/spindles).
#
# 2. CORTICAL IDLING ALPHA: Generated in posterior cortex when areas are not
#    actively processing (Pfurtscheller & Lopes da Silva 1999 Clin Neurophysiol).
#    Decreases with cognitive engagement regardless of precision level.
#    Dominates resting-state EEG recordings. Increases during meditation
#    (~16%, Ahani et al. 2014) and relaxation.
#
# Existing alpha targets (sleep, psilocybin, depression) use thalamocortical.
# The ATX alpha and meditation alpha failures arise from applying the
# thalamocortical formula to measurements dominated by idling alpha.

# TC_ALPHA_NE_GAIN: NE modulation of waking thalamocortical alpha.
# alpha_tc = P^exponent * (1 + TC_ALPHA_NE_GAIN * (NE_rel - 1.0))
# At NE_rel=1.0 (baseline): no modulation. At NE_rel=1.5 (ATX): +10%.
# Citation: Haegens et al. (2011) J Neurosci 31:3016 (LC→thalamus modulation)
# Set to 0.2: modest effect — NE is secondary to precision for alpha.
TC_ALPHA_NE_GAIN = 0.2  # MOTIVATED: LC→thalamic alpha modulation

# IDLING_ALPHA_GAIN: Sensitivity of cortical idling alpha to cognitive engagement.
# alpha_idling = 1.0 - IDLING_ALPHA_GAIN * cognitive_engagement
# At engagement=0.0 (full rest): alpha_idling=1.0
# At engagement=1.0 (active task): alpha_idling=0.5 (50% desynchronization)
# Citation: Pfurtscheller & Lopes da Silva (1999) Clin Neurophysiol 110:1842
# Calibrated: 30-50% ERD during cognitive tasks → gain ≈ 0.5
IDLING_ALPHA_GAIN = 0.5  # MOTIVATED: ERD magnitude during cognitive engagement

# --- REM posterior cortical alpha ---
# During REM, posterior cortical generators produce alpha independently of
# thalamocortical gating. High ACh in REM activates these generators while
# low NE/5-HT suppresses thalamocortical relay (Cantero et al. 1999, 2002).
# This produces REM alpha that is higher than P_rem alone would predict.
# Implementation: additive term alpha = P^exp + ALPHA_REM_CORTICAL_GAIN * REM_drive
# Only active during REM (REM_drive=0 during wake/NREM). Zero impact on
# waking and NREM alpha predictions.
# Citation: Cantero et al. (1999) Neuroscience 89:671 (posterior cortical alpha in REM)
#           Cantero et al. (2002) J Neurosci 22:10941 (alpha generators differ wake vs REM)
ALPHA_REM_CORTICAL_GAIN = 0.16  # MOTIVATED: posterior cortical alpha floor during REM

# LZW complexity: LZW ∝ (1 - P)^0.5
# Citation: Schartner et al. (2017) Sci Rep 7:46421 (ketamine, psilocybin)
# Note: Propofol LZW data from Schartner et al. (2015) PLoS ONE 10:e0133532
LZW_EXPONENT = 0.4338  # FITTED (refit 1, 14-param): LZW complexity exponent

# --- Slow-wave synchronization (NREM complexity fix) ---
# During NREM, reduced precision enables spontaneous bottom-up synchronization
# via Landau phase transition: when P drops below P_CRIT_SYNC, synchronization
# variable S grows through spontaneous symmetry breaking. S reduces LZW complexity
# via compressibility — synchronized traveling waves create inter-regional redundancy
# that LZW exploits, lowering measured complexity despite increased local noise.
#
# Mathematical form (Landau bifurcation):
#   dS/dt = K_SYNC * max(0, P_CRIT_SYNC - P_mean) * sleep_drive - S / TAU_SYNC
# Below P_crit: S grows. Above P_crit: S decays.
# sleep_drive gates S to only operate during sleep (prevents S during psilocybin).
#
# LZW operationalization becomes:
#   LZW = (1-P)^exp * (1 - S_COMPRESS * S)
# where S_COMPRESS controls how strongly synchronization reduces complexity.
#
# Citation: Landau theory of phase transitions (Landau & Lifshitz, 1937)
#           Tononi & Cirelli (2006) Sleep Med Rev 10:49 (synaptic homeostasis)
#           Massimini et al. (2004) J Neurosci 24:6862 (slow-wave traveling waves)
K_SYNC = 5.0              # MOTIVATED: synchronization growth rate below P_crit
P_CRIT_SYNC = 0.70        # MOTIVATED: critical P for synchronization onset (above wake/NREM boundary)
TAU_SYNC = 0.5            # MOTIVATED: synchronization decay time constant (hours)
S_COMPRESS = 0.97         # MOTIVATED: max LZW reduction at full synchronization

# fMRI prediction error BOLD: PE ∝ KL * P
# Citation: Iglesias et al. (2013) Neuron 80:519
FMRI_PE_SCALING = 1.0  # FREE: BOLD sensitivity scaling

# P300 amplitude: two-factor model — updating capacity (1-P)^γ × DA salience quality
# P300 reflects precision-weighted UPDATING, not raw precision level.
# High P (depression) → rigid priors → less updating → low P300
# DA disruption (ADHD, psychosis) → impaired salience → low P300
# Citation: Kolossa et al. (2015) Front Hum Neurosci 9:223; Nieuwenhuis et al. (2005)
P300_EXPONENT = 0.5  # FREE: P300 updating sensitivity (not fitted in v3)
P300_DA_SIGMA = 0.20  # MOTIVATED: width of DA inverted-U for P300 (Nieuwenhuis 2005)

# Pupil diameter: NE drives sympathetic pupil dilation (Joshi & Gold 2020)
# Citation: Joshi et al. (2016) eLife 5:e18547
PUPIL_BASELINE = 1.0  # MOTIVATED: normalized baseline pupil diameter
PUPIL_NE_GAIN = 0.5   # MOTIVATED: NE → pupil linear gain

# Heart rate variability: vagal withdrawal from NE + cortisol
# Citation: Kemp et al. (2010) Biol Psychiatry 67:1067
HRV_NE_GAIN = 1.0     # MOTIVATED: NE contribution to vagal withdrawal
HRV_CORT_GAIN = 0.5   # MOTIVATED: cortisol contribution to vagal withdrawal

# BDNF: serum BDNF ∝ plasticity^γ (sublinear)
# Citation: Molendijk et al. (2014) Mol Psychiatry 19:791 (d=-0.71 for MDD)
# Note: Brunoni et al. 2008 reports d=0.91; the d=-0.71 value is from Molendijk 2014
BDNF_EXPONENT = 0.8   # MOTIVATED: sublinear relationship


# ============================================================================
# PARAMETER SUMMARY
# ============================================================================

def get_parameter_table():
    """Return a structured summary of all parameters with metadata."""
    params = []

    # --- GROUNDED ---
    grounded = [
        ("TAU_CIRCADIAN", TAU_CIRCADIAN, "h", "24.0", "23.8–24.2", "Czeisler et al. 1999"),
        ("TAU_ULTRADIAN", TAU_ULTRADIAN, "h", "1.5", "1.33–1.67", "Aserinsky & Kleitman 1953"),
        ("CORTISOL_CAR_PEAK", CORTISOL_CAR_PEAK, "h", "7.0", "6.5–7.5", "Debono et al. 2009"),
        ("CORTISOL_NADIR", CORTISOL_NADIR, "h", "23.5", "22.0–1.0", "Debono et al. 2009"),
        ("CORTISOL_HALF_LIFE", CORTISOL_HALF_LIFE, "h", "1.5", "1.0–2.0", "Debono et al. 2009"),
        ("PSILOCYBIN_ONSET_TAU", PSILOCYBIN_ONSET_TAU, "h", "0.5", "0.33–0.67", "Holze et al. 2023"),
        ("PSILOCYBIN_PEAK_TIME", PSILOCYBIN_PEAK_TIME, "h", "1.75", "1.0–2.5", "Holze et al. 2023"),
        ("PSILOCYBIN_HALF_LIFE", PSILOCYBIN_HALF_LIFE, "h", "1.7", "1.4–1.8", "Holze et al. 2023"),
        ("PSILOCYBIN_DURATION", PSILOCYBIN_DURATION, "h", "6.0", "4.0–8.0", "Holze et al. 2023"),
        ("KETAMINE_ONSET_TAU", KETAMINE_ONSET_TAU, "h", "0.67", "0.5–1.0", "Zarate et al. 2006"),
        ("KETAMINE_BDNF_PEAK", KETAMINE_BDNF_PEAK, "h", "3.0", "2.0–4.0", "Autry et al. 2011"),
        ("KETAMINE_BENEFIT_DURATION", KETAMINE_BENEFIT_DURATION, "h", "240", "168–336", "Zarate et al. 2006"),
        ("RECEPTOR_5HT2A_SENSORY", RECEPTOR_5HT2A_SENSORY, "", "0.30", "0.25–0.35", "Beliveau et al. 2017"),
        ("RECEPTOR_5HT2A_ASSOCIATION", RECEPTOR_5HT2A_ASSOCIATION, "", "0.50", "0.45–0.55", "Beliveau et al. 2017"),
        ("RECEPTOR_5HT2A_DMN", RECEPTOR_5HT2A_DMN, "", "0.40", "0.35–0.45", "Beliveau et al. 2017"),
        ("RECEPTOR_DOWNREG_MAX", RECEPTOR_DOWNREG_MAX, "", "0.30", "0.20–0.40", "Buckholtz et al. 1990"),
        ("RECEPTOR_RECOVERY_TAU", RECEPTOR_RECOVERY_TAU, "h", "240", "168–336", "Buckholtz et al. 1990"),
        ("GABA_ANXIETY_DEFICIT", GABA_ANXIETY_DEFICIT, "", "0.175", "0.15–0.20", "Goddard et al. 2001"),
    ]

    # --- MOTIVATED ---
    motivated = [
        ("NE_BASELINE", NE_BASELINE, "", "0.50", "NE high during wake, near-zero in REM"),
        ("NE_AMPLITUDE", NE_AMPLITUDE, "", "0.10", "Modest circadian NE swing"),
        ("SEROTONIN_BASELINE", SEROTONIN_BASELINE, "", "0.45", "Raphe tonic firing, tracks NE"),
        ("SEROTONIN_AMPLITUDE", SEROTONIN_AMPLITUDE, "", "0.08", "Slight circadian 5-HT swing"),
        ("ACH_WAKING", ACH_WAKING, "", "0.30", "Moderate BF tone during wake"),
        ("ACH_REM", ACH_REM, "", "0.90", "High PPT/LDT in REM"),
        ("ACH_NREM", ACH_NREM, "", "0.10", "Very low in NREM"),
        ("DA_BASELINE", DA_BASELINE, "", "0.45", "Tonic VTA ~4 Hz (Grace 1991)"),
        ("GLU_BASELINE", GLU_BASELINE, "", "0.50", "Tightly regulated"),
        ("GABA_NORMALIZED", GABA_NORMALIZED, "", "1.00", "Normalized baseline"),
        ("ENDOGENOUS_PLASTICITY_TONIC", ENDOGENOUS_PLASTICITY_TONIC, "", "0.15", "Persistent plasticity floor"),
        ("ENDOGENOUS_PLASTICITY_REM_PEAK", ENDOGENOUS_PLASTICITY_REM_PEAK, "", "0.60", "Phasic REM peaks"),
        ("ALPHA_NE", ALPHA_NE, "", "0.40", "NE → suppression, sign: positive"),
        ("ALPHA_5HT", ALPHA_5HT, "", "0.35", "5-HT → suppression, sign: positive"),
        ("ALPHA_DA", ALPHA_DA, "", "0.15", "DA → salience/P maintenance"),
        ("BETA_PLAST", BETA_PLAST, "", "0.45", "Plasticity → reduces P"),
        ("BETA_ACH", BETA_ACH, "", "0.30", "ACh → plasticity drive"),
        ("BETA_GLU", BETA_GLU, "", "0.20", "GLU/BDNF → plasticity"),
        ("CORT_NE_COUPLING", CORT_NE_COUPLING, "", "0.20", "CRH → LC, sign: positive"),
        ("CORT_5HT_COUPLING", CORT_5HT_COUPLING, "", "-0.10", "Chronic cortisol ↓ 5-HT"),
        ("CORT_DA_COUPLING", CORT_DA_COUPLING, "", "-0.15", "Allostatic load ↓ DA"),
        ("CORT_PLASTICITY_COUPLING", CORT_PLASTICITY_COUPLING, "", "-0.25", "Cortisol ↓ BDNF"),
        ("COUPLING_TD", COUPLING_TD, "", "0.15", "Top-down Bayesian constraint"),
        ("COUPLING_BU", COUPLING_BU, "", "0.10", "Bottom-up PE propagation"),
        ("GAMMA_SENSORY", GAMMA_SENSORY, "", "0.90", "Fastest homeostatic recovery"),
        ("GAMMA_CONCEPTUAL", GAMMA_CONCEPTUAL, "", "0.80", "Moderate recovery"),
        ("GAMMA_SELFMODEL", GAMMA_SELFMODEL, "", "0.65", "Slowest recovery"),
        ("GABA_NE_GAIN_MOD", GABA_NE_GAIN_MOD, "", "0.50", "GABA modulates NE gain"),
        ("SLEEP_ONSET", SLEEP_ONSET, "h", "23.0", "Sleep onset clock time"),
        ("WAKE_TIME", WAKE_TIME, "h", "7.0", "Wake time"),
        ("CORTISOL_DIURNAL_TAU", CORTISOL_DIURNAL_TAU, "h", "8.0", "Diurnal cortisol envelope tau (not metabolic)"),
        ("EQUILIBRIUM_REF_TIME", EQUILIBRIUM_REF_TIME, "h", "14.0", "Equilibrium offset reference time"),
        ("ALLOSTATIC_THRESHOLD_FACTOR", ALLOSTATIC_THRESHOLD_FACTOR, "", "1.10", "Allostatic load accumulation threshold"),
        ("ALLOSTATIC_LOAD_MAX", ALLOSTATIC_LOAD_MAX, "", "2.00", "Allostatic load ceiling"),
        ("DA_EXCESS_SCALING", DA_EXCESS_SCALING, "", "0.50", "DA excess → tonic DA scaling"),
        ("ADHD_NOISE_SCALING", ADHD_NOISE_SCALING, "", "3.00", "Transporter noise amplification"),
        ("ADHD_DA_SALIENCE_NOISE", ADHD_DA_SALIENCE_NOISE, "", "0.50", "DA salience noise in ADHD"),
        ("CHRONIC_NE_SENSITIZATION", CHRONIC_NE_SENSITIZATION, "", "0.30", "Allostatic → chronic NE gain"),
        ("CHRONIC_5HT_ELEVATION", CHRONIC_5HT_ELEVATION, "", "0.15", "Allostatic → chronic 5-HT gain"),
        ("CSTC_SUPPRESSION_SCALING", CSTC_SUPPRESSION_SCALING, "", "0.15", "CSTC GLU → conceptual suppression"),
        ("BDNF_SURGE_SCALING", BDNF_SURGE_SCALING, "", "0.60", "Ketamine BDNF surge amplitude"),
        ("PSILOCYBIN_AFTERGLOW_TAU", PSILOCYBIN_AFTERGLOW_TAU, "h", "72.0", "Psilocybin afterglow decay tau"),
        ("PTSD_TD_BREAKDOWN", PTSD_TD_BREAKDOWN, "", "0.30", "Top-down coupling retention in PTSD"),
        ("PSILOCYBIN_PHARMA_GAIN", PSILOCYBIN_PHARMA_GAIN, "", "2.00", "5-HT2A receptor-mediated amplification"),
        ("KETAMINE_PHARMA_GAIN", KETAMINE_PHARMA_GAIN, "", "2.00", "BDNF-TrkB pathway amplification"),
        ("PTSD_DISSOC_COEFF", PTSD_DISSOC_COEFF, "", "0.25", "Dissociation drive strength"),
        ("HPA_FEEDBACK_GAIN", HPA_FEEDBACK_GAIN, "", "0.30", "GR negative feedback"),
        ("BIPOLAR_K_INSTAB", BIPOLAR_K_INSTAB, "", "0.008", "Mood-state positive feedback"),
        ("BIPOLAR_K_FEEDBACK", BIPOLAR_K_FEEDBACK, "", "0.020", "Slow compensatory gain"),
        ("BIPOLAR_TAU_ADAPT", BIPOLAR_TAU_ADAPT, "h", "300", "~12.5 day adaptation"),
        ("SC_CONSOLIDATION_TAU", SC_CONSOLIDATION_TAU, "h", "336", "Spine density decay (14 days, Ly 2018)"),
        ("K_CONSOLIDATION", K_CONSOLIDATION, "", "0.10", "SC accumulation rate during 5-HT2A agonism"),
        ("ALPHA_NE_PHASIC", ALPHA_NE_PHASIC, "", "0.30", "Phasic NE coupling: d(NE)/dt, not eq-subtracted"),
        ("ALPHA_5HT_PHASIC", ALPHA_5HT_PHASIC, "", "0.20", "Phasic 5-HT coupling: d(5-HT)/dt, not eq-subtracted"),
    ]

    # --- FREE ---
    free = [
        ("P_MIN", P_MIN, "", "0.15", "Lowest P, model convention"),
        ("P_MAX", P_MAX, "", "0.95", "Highest P, model convention"),
        ("P_WAKING", P_WAKING, "", "0.70", "Waking baseline, dimensionless"),
        ("P_REM", P_REM, "", "0.25", "REM target"),
        ("P_NREM", P_NREM, "", "0.55", "NREM target"),
        ("P_SENSORY_BASELINE", P_SENSORY_BASELINE, "", "0.75", "Sensory waking"),
        ("P_CONCEPTUAL_BASELINE", P_CONCEPTUAL_BASELINE, "", "0.70", "Conceptual waking"),
        ("P_SELFMODEL_BASELINE", P_SELFMODEL_BASELINE, "", "0.60", "Self-model waking"),
        ("SIGMA_NOISE", SIGMA_NOISE, "", "0.02", "Stochastic noise amplitude"),
        ("NOISE_TAU", NOISE_TAU, "h", "0.25", "Noise autocorrelation"),
        ("TAU_P", TAU_P, "h", "0.50", "P integration time constant"),
    ]

    return {"grounded": grounded, "motivated": motivated, "free": free}


def print_parameter_summary():
    """Print a formatted table of all parameters."""
    tables = get_parameter_table()

    print("\n" + "=" * 80)
    print("PARAMETER REGISTRY — Plasticity Model v2")
    print("=" * 80)

    for category, entries in tables.items():
        label = category.upper()
        print(f"\n--- {label} ({len(entries)} parameters) ---")
        if category == "grounded":
            print(f"{'Name':<30} {'Value':>8} {'Unit':>4} {'Range':>12} {'Citation'}")
            print("-" * 90)
            for name, val, unit, nominal, range_or_cite, cite in entries:
                print(f"{name:<30} {val:>8.3f} {unit:>4} {range_or_cite:>12} {cite}")
        elif category == "motivated":
            print(f"{'Name':<30} {'Value':>8} {'Unit':>4} {'Rationale'}")
            print("-" * 80)
            for name, val, unit, nominal, rationale in entries:
                print(f"{name:<30} {val:>8.3f} {unit:>4} {rationale}")
        else:
            print(f"{'Name':<30} {'Value':>8} {'Unit':>4} {'Note'}")
            print("-" * 70)
            for name, val, unit, nominal, note in entries:
                print(f"{name:<30} {val:>8.3f} {unit:>4} {note}")

    total = sum(len(v) for v in tables.values())
    print(f"\n{'Total parameters:':<30} {total}")
    print(f"{'  Grounded:':<30} {len(tables['grounded'])}")
    print(f"{'  Motivated:':<30} {len(tables['motivated'])}")
    print(f"{'  Free:':<30} {len(tables['free'])}")
    print("=" * 80)


if __name__ == "__main__":
    print_parameter_summary()
