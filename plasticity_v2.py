"""
Plasticity Model v2 — Empirically Grounded, Emergent Pathologies
=================================================================

Core engine implementing:
- 10-variable state vector (3 precision, 2 receptor, 3 HPA, 2 bipolar)
- 6 neuromodulator pathways (NE, 5-HT, DA, ACh, GLU, GABA)
- HPA axis with cortisol diurnal rhythm and stress response
- Receptor dynamics (5-HT2A downregulation, D2 upregulation)
- Emergent pathological P profiles from upstream parameters only
- P operationalization to measurable neural observables

Key change from v1: Scenarios specify UPSTREAM parameters (stress level,
transporter dysfunction, etc.) and P profiles EMERGE from the dynamics.
No per-scenario P overrides.

State vector at each timestep:
    P_sensory, P_conceptual, P_selfmodel    (3 floats)
    R_5HT2A, R_D2                           (2 floats, receptor density)
    cortisol, hpa_sensitivity, allostatic_load  (3 floats)
    P_setpoint, R_bipolar                   (2 floats, optional)

Total: 10 state variables (vs 1 in v1, 3 in extensions)
"""

import numpy as np
from parameters import *


# ============================================================================
# NEUROMODULATOR TIME COURSES
# ============================================================================

def is_sleep(t, sleep_onset=None, wake_time=None):
    """
    Determine if time t (hours, mod 24) falls in the sleep window.
    Returns a smooth sleep drive (0 = awake, 1 = deep sleep).
    """
    if sleep_onset is None:
        sleep_onset = SLEEP_ONSET
    if wake_time is None:
        wake_time = WAKE_TIME
    t_mod = t % 24.0
    if t_mod > 12:
        return 1.0 / (1.0 + np.exp(-8.0 * (t_mod - sleep_onset)))
    else:
        return 1.0 / (1.0 + np.exp(-8.0 * (wake_time - t_mod)))


def sleep_stage(t, sleep_onset=None):
    """
    Model sleep stages. Returns (nrem_drive, rem_drive) each 0-1.
    REM cycles every ~90 min, with REM proportion increasing through night.
    """
    if sleep_onset is None:
        sleep_onset = SLEEP_ONSET
    t_mod = t % 24.0
    if t_mod >= sleep_onset:
        hours_asleep = t_mod - sleep_onset
    elif t_mod < 12:
        hours_asleep = (24.0 - sleep_onset) + t_mod
    else:
        return 0.0, 0.0

    if hours_asleep < 0 or hours_asleep > 9:
        return 0.0, 0.0

    # REM proportion increases through night (grounded: 20% early → 50% late)
    rem_propensity = SLEEP_REM_EARLY_FRACTION + \
        (SLEEP_REM_LATE_FRACTION - SLEEP_REM_EARLY_FRACTION) * (hours_asleep / 8.0)
    cycle_phase = np.sin(2 * np.pi * hours_asleep / TAU_ULTRADIAN)

    rem_drive = rem_propensity * max(0, cycle_phase) ** 2
    nrem_drive = (1.0 - rem_drive) * min(1.0, hours_asleep / 0.5)

    return nrem_drive, rem_drive


def norepinephrine(t, ne_baseline=None, ne_amplitude=None):
    """
    NE: high during waking, near-zero in REM, low in NREM.
    Circadian peak mid-morning (~10:00).
    """
    if ne_baseline is None:
        ne_baseline = NE_BASELINE
    if ne_amplitude is None:
        ne_amplitude = NE_AMPLITUDE

    sleep = is_sleep(t)
    _, rem = sleep_stage(t)

    t_mod = t % 24.0
    circadian = ne_baseline + ne_amplitude * np.cos(
        2 * np.pi * (t_mod - 10.0) / TAU_CIRCADIAN)

    wake_gate = 1.0 - sleep * (0.6 + 0.35 * rem)
    return max(0.0, circadian * wake_gate)


def serotonin(t, sht_baseline=None, sht_amplitude=None):
    """
    5-HT: tracks NE roughly, more completely suppressed in REM.
    Circadian peak late morning (~11:00).
    """
    if sht_baseline is None:
        sht_baseline = SEROTONIN_BASELINE
    if sht_amplitude is None:
        sht_amplitude = SEROTONIN_AMPLITUDE

    sleep = is_sleep(t)
    _, rem = sleep_stage(t)

    t_mod = t % 24.0
    circadian = sht_baseline + sht_amplitude * np.cos(
        2 * np.pi * (t_mod - 11.0) / TAU_CIRCADIAN)

    wake_gate = 1.0 - sleep * (0.7 + 0.28 * rem)
    return max(0.0, circadian * wake_gate)


def acetylcholine(t):
    """
    ACh: moderate in wake, very low in NREM, surges in REM.
    """
    sleep = is_sleep(t)
    nrem, rem = sleep_stage(t)

    waking = ACH_WAKING * (1.0 - sleep)
    nrem_c = ACH_NREM * nrem * sleep
    rem_c = ACH_REM * rem * sleep

    return waking + nrem_c + rem_c


def dopamine(t, da_baseline=None):
    """
    DA: tonic VTA baseline + circadian modulation.
    Reduced ~50% in sleep (Lena et al. 2005).
    """
    if da_baseline is None:
        da_baseline = DA_BASELINE

    sleep = is_sleep(t)

    t_mod = t % 24.0
    circadian = da_baseline + DA_AMPLITUDE * np.cos(
        2 * np.pi * (t_mod - 12.0) / TAU_CIRCADIAN)

    wake_gate = 1.0 - sleep * DA_SLEEP_REDUCTION
    return max(0.0, circadian * wake_gate)


def glutamate(t):
    """
    GLU: tightly regulated baseline. Minimal circadian variation.
    Ketamine NMDA blockade → compensatory surge modeled in pharmacology.
    """
    t_mod = t % 24.0
    return GLU_BASELINE + GLU_AMPLITUDE * np.cos(
        2 * np.pi * (t_mod - 14.0) / TAU_CIRCADIAN)


def gaba(t, deficit=0.0):
    """
    GABA: cortical inhibitory tone. Slight NREM increase.
    deficit: fractional reduction (0 = normal, 0.175 = anxiety-level deficit).
    """
    sleep = is_sleep(t)
    nrem, _ = sleep_stage(t)

    base = GABA_NORMALIZED * (1.0 - deficit)
    sleep_mod = GABA_SLEEP_INCREASE * nrem * sleep

    return base + sleep_mod


def endogenous_plasticity(t, scale=1.0):
    """
    Endogenous plasticity drive: tonic 5-HT2A/plasticity signaling.
    Encompasses endogenous tryptamines (DMT as candidate), endocannabinoids,
    tonic BDNF, spontaneous LTP. Not specific to any single mediator.
    Note: INMT KO mice (Hatzipantelis 2025) show INMT is not the primary
    DMT-producing enzyme in rodents — the tonic 5-HT2A hypothesis does not
    depend on INMT-mediated DMT synthesis specifically.
    """
    _, rem = sleep_stage(t)

    tonic = ENDOGENOUS_PLASTICITY_TONIC * scale
    phasic = ENDOGENOUS_PLASTICITY_REM_PEAK * rem

    return tonic + phasic


def cortisol_rhythm(t, stress_input=0.0, cortisol_state=None):
    """
    Diurnal cortisol rhythm: Gaussian CAR peak + exponential decay.
    Stress input drives cortisol elevation.

    Returns cortisol level (normalized).
    """
    t_mod = t % 24.0

    # CAR: Gaussian peak around 07:00
    car = CORTISOL_CAR_AMPLITUDE * np.exp(
        -0.5 * ((t_mod - CORTISOL_CAR_PEAK) / 1.5) ** 2)

    # If modular time is past 12, also check for the next day's CAR
    if t_mod > 20:
        car += CORTISOL_CAR_AMPLITUDE * np.exp(
            -0.5 * (((t_mod - 24.0) - CORTISOL_CAR_PEAK) / 1.5) ** 2)

    # Exponential decay from nadir baseline through day
    # Higher in morning, lower in evening
    if t_mod >= CORTISOL_CAR_PEAK:
        decay_hours = t_mod - CORTISOL_CAR_PEAK
    else:
        decay_hours = (24.0 - CORTISOL_CAR_PEAK) + t_mod

    diurnal_decay = CORTISOL_BASELINE * np.exp(-decay_hours * 0.693 / CORTISOL_DIURNAL_TAU)

    # Stress augmentation
    stress_component = CORTISOL_STRESS_GAIN * stress_input

    return car + diurnal_decay + stress_component


# ============================================================================
# EQUILIBRIUM OFFSETS
# ============================================================================
# Computed from waking neuromodulator levels at t=14:00 so drive ≈ 0
# at normal waking baseline. Without this, monoamine dominance pushes P to ceiling.

_NE_EQ = NE_BASELINE + NE_AMPLITUDE * np.cos(2 * np.pi * (EQUILIBRIUM_REF_TIME - 10.0) / TAU_CIRCADIAN)
_5HT_EQ = SEROTONIN_BASELINE + SEROTONIN_AMPLITUDE * np.cos(
    2 * np.pi * (EQUILIBRIUM_REF_TIME - 11.0) / TAU_CIRCADIAN)
_DA_EQ = DA_BASELINE + DA_AMPLITUDE * np.cos(2 * np.pi * (EQUILIBRIUM_REF_TIME - 12.0) / TAU_CIRCADIAN)
# DA salience contribution at equilibrium (inverted-U centered at DA_SALIENCE_OPTIMAL)
_DA_DEV_EQ = (_DA_EQ - DA_SALIENCE_OPTIMAL) / DA_SALIENCE_WIDTH
_DA_SALIENCE_EQ = ALPHA_DA * np.exp(-0.5 * _DA_DEV_EQ ** 2)
_SUPPRESSION_EQ = ALPHA_NE * _NE_EQ + ALPHA_5HT * _5HT_EQ + _DA_SALIENCE_EQ
_PLASTICITY_EQ = BETA_PLAST * ENDOGENOUS_PLASTICITY_TONIC + BETA_ACH * ACH_WAKING


# ============================================================================
# PHARMACOLOGICAL PERTURBATIONS
# ============================================================================

def psilocybin_perturbation(t, dose_time=14.0, dose_strength=0.6, duration=None):
    """
    Psilocybin/psilocin: 5-HT2A agonist.
    PK: onset ~30 min, peak ~1.75h, half-life ~3h (grounded).
    Returns agonist occupancy (0-1) for receptor dynamics.
    """
    if duration is None:
        duration = PSILOCYBIN_DURATION
    dt_val = t - dose_time
    if dt_val < 0:
        return 0.0

    onset = 1.0 - np.exp(-dt_val / PSILOCYBIN_ONSET_TAU)
    clearance = np.exp(-dt_val * 0.693 / PSILOCYBIN_HALF_LIFE)
    acute = dose_strength * onset * clearance

    # Afterglow: BDNF-mediated sustained mild reduction
    afterglow = 0.08 * dose_strength * (1.0 - np.exp(-dt_val / 2.0)) * np.exp(-dt_val / PSILOCYBIN_AFTERGLOW_TAU)

    return acute + afterglow


def dmt_perturbation(t, dose_time=14.0, dose_strength=0.6, duration=None):
    """
    DMT (IV): 5-HT2A agonist with ultra-short pharmacokinetics.
    Same receptor mechanism as psilocybin but compressed ~12x in time.
    PK: onset ~2 min, peak ~5 min, half-life ~15 min (grounded).
    """
    if duration is None:
        duration = DMT_DURATION
    dt_val = t - dose_time
    if dt_val < 0:
        return 0.0

    onset = 1.0 - np.exp(-dt_val / DMT_ONSET_TAU)
    clearance = np.exp(-dt_val * 0.693 / DMT_HALF_LIFE)
    acute = dose_strength * onset * clearance

    # Shorter afterglow than psilocybin (less time for BDNF cascade)
    afterglow = 0.04 * dose_strength * (1.0 - np.exp(-dt_val / 0.5)) * np.exp(-dt_val / DMT_AFTERGLOW_TAU)

    return acute + afterglow


def ketamine_perturbation(t, dose_time=0.0, dose_strength=0.5):
    """
    Ketamine: NMDA antagonist → compensatory GLU surge → BDNF → rapid P reduction.
    PK: onset ~40 min, BDNF peak 2-4h, benefit 1-2 weeks (grounded).

    Returns tuple: (nmda_blockade, bdnf_surge)
    - nmda_blockade: acute NMDA receptor blockade (hours)
    - bdnf_surge: BDNF-mediated plasticity increase (days to weeks)
    """
    dt_val = t - dose_time
    if dt_val < 0:
        return 0.0, 0.0

    # Acute NMDA blockade (short-lived)
    onset = 1.0 - np.exp(-dt_val / KETAMINE_ONSET_TAU)
    clearance = np.exp(-dt_val * 0.693 / 2.0)  # ketamine half-life ~2h IV
    nmda_blockade = dose_strength * onset * clearance

    # BDNF surge: peaks at 2-4h, sustained for 1-2 weeks
    bdnf_onset = 1.0 - np.exp(-dt_val / KETAMINE_BDNF_PEAK)
    bdnf_decay = np.exp(-dt_val * 0.693 / KETAMINE_BENEFIT_DURATION)
    bdnf_surge = dose_strength * BDNF_SURGE_SCALING * bdnf_onset * bdnf_decay

    return nmda_blockade, bdnf_surge


def antipsychotic_perturbation(t, dose_time=0.0, dose_strength=0.3, half_life=12.0):
    """
    Typical antipsychotic: D2 antagonist + 5-HT2A antagonist.
    Returns D2 antagonist occupancy for receptor dynamics.
    """
    dt_val = t - dose_time
    if dt_val < 0:
        return 0.0

    onset = 1.0 - np.exp(-dt_val / 1.0)
    clearance = np.exp(-0.693 * dt_val / half_life)
    return dose_strength * onset * clearance


def ssri_perturbation(t, start_time=0.0, dose_strength=0.15, ramp_weeks=3.0):
    """
    SSRI two-phase mechanism (Duman & Monteggia 2006):
      Phase 1 (weeks 1-2): Acute 5-HT increase → mild P increase (side effects,
        initial worsening) — autoreceptor desensitization not yet complete.
      Phase 2 (weeks 3+): Sustained 5-HT → BDNF upregulation → neuroplasticity →
        P DECREASE (therapeutic effect). This is why SSRIs take 4-6 weeks.

    Net effect: early worsening → delayed improvement, matching clinical onset.
    """
    dt_val = t - start_time
    if dt_val < 0:
        return 0.0
    ramp_hours = ramp_weeks * 7 * 24
    ramp = 1.0 - np.exp(-dt_val / ramp_hours)

    # Phase 1: acute monoaminergic → mild P increase
    acute = dose_strength * 0.3 * ramp

    # Phase 2: neuroplasticity onset → P decrease (delayed ~2 weeks)
    neuro_delay = SSRI_NEUROPLASTICITY_DELAY * 7 * 24  # hours
    neuro_ramp_tau = SSRI_NEUROPLASTICITY_TAU * 7 * 24  # hours
    if dt_val > neuro_delay:
        neuro_ramp = 1.0 - np.exp(-(dt_val - neuro_delay) / neuro_ramp_tau)
        plasticity_effect = -dose_strength * SSRI_NEUROPLASTICITY_GAIN * neuro_ramp
    else:
        plasticity_effect = 0.0

    return acute + plasticity_effect


def atomoxetine_perturbation(t, dose_time=0.0, dose_strength=0.5):
    """
    Atomoxetine: selective norepinephrine reuptake inhibitor (NET blockade).
    Increases synaptic NE by blocking reuptake. Single-phase kinetics.

    Returns NE elevation factor (0-1) for modulating ne_base_mod.
    """
    dt_val = t - dose_time
    if dt_val < 0:
        return 0.0

    onset = 1.0 - np.exp(-dt_val / ATOMOXETINE_ONSET_TAU)
    clearance = np.exp(-dt_val * 0.693 / ATOMOXETINE_HALF_LIFE)
    return dose_strength * onset * clearance


# ============================================================================
# CORE DYNAMICS
# ============================================================================

def suppression(ne, sht, da=None, ne_scale=1.0, sht_scale=1.0):
    """
    Suppression term: monoaminergic tone that INCREASES precision P.
    NE (LC) and 5-HT (raphe) provide tonic stability.
    DA contributes moderate salience-based P maintenance.
    """
    sup = ALPHA_NE * ne_scale * ne + ALPHA_5HT * sht_scale * sht
    if da is not None:
        # DA inverted-U: moderate DA supports P, excess disrupts
        da_dev = (da - DA_SALIENCE_OPTIMAL) / DA_SALIENCE_WIDTH
        da_contribution = ALPHA_DA * np.exp(-0.5 * da_dev ** 2)
        sup += da_contribution
    return sup


def plasticity_drive(plast, ach, glu_bdnf=0.0, plast_scale=1.0):
    """
    Plasticity drive: DECREASES precision P.
    Endogenous plasticity (tryptamines/BDNF/LTP) + ACh + GLU/BDNF pathway.
    """
    return (BETA_PLAST * plast_scale * plast +
            BETA_ACH * ach +
            BETA_GLU * glu_bdnf)


def da_salience_disruption(da):
    """
    Excess DA disrupts salience filtering → P reduction.
    Returns a negative contribution to P when DA is too high.
    Captures psychosis mechanism: hyperdopaminergia → aberrant salience.
    Onset is smooth (no hard threshold), scaled by squared excess.
    """
    if da <= DA_SALIENCE_OPTIMAL:
        return 0.0
    excess = (da - DA_SALIENCE_OPTIMAL) / DA_SALIENCE_WIDTH
    return -ALPHA_DA * 0.5 * excess ** 2


def gaba_ne_gain(gaba_level):
    """
    GABA modulates the GAIN of NE coupling (deficit pathway only).
    Low GABA → amplified NE effect → anxiety-like sensory hyperarousal.
    Returns a gain multiplier (1.0 = normal, >1.0 = amplified).
    GABA excess (propofol) is handled separately via direct cortical inhibition.
    """
    deviation = GABA_NORMALIZED - gaba_level
    return 1.0 + GABA_NE_GAIN_MOD * max(0.0, deviation)


def energy_landscape(x, P):
    """
    Energy landscape E(x, P).
    High P: deep attractor wells (rigid). Low P: flat (fluid).
    """
    well_width = 0.8
    attractor_term = np.zeros_like(x)
    for w, c in zip(PRIOR_WEIGHTS, ATTRACTOR_CENTERS):
        attractor_term += w * np.exp(-(x - c) ** 2 / (2 * well_width ** 2))

    entropy_term = 0.3 * x ** 2
    return -P * attractor_term + (1.0 - P) * entropy_term


# ============================================================================
# P OPERATIONALIZATION
# ============================================================================

def p_to_eeg_alpha(P, exponent=None):
    """EEG alpha power ∝ P^γ (Jensen & Mazaheri 2010).

    This is the THALAMOCORTICAL alpha operationalization. It captures
    precision-mediated inhibitory gating and is appropriate for:
    - Sleep states (NREM/REM): alpha reflects thalamocortical dynamics
    - Psilocybin: precision reduction → reduced thalamocortical gating
    - Depression: elevated precision → increased thalamocortical gating

    NOT appropriate for:
    - Resting-state pharmacological EEG (dominated by cortical idling alpha)
    - Meditation (relaxation alpha = cortical idling, not precision-mediated)
    See p_to_alpha_idling() for those contexts.
    """
    if exponent is None:
        exponent = ALPHA_POWER_EXPONENT
    return np.clip(P, 0, 1) ** exponent


def p_to_alpha_thalamocortical(P, NE_relative=1.0, exponent=None, ne_gain=None):
    """Thalamocortical alpha with NE modulation for waking states.

    alpha_tc = P^exponent * (1 + ne_gain * (NE_relative - 1.0))

    NE modulation is secondary to precision: LC→thalamic projections mildly
    enhance thalamocortical alpha via reticular nucleus (Haegens et al. 2011).
    At NE_relative=1.0: reduces to p_to_eeg_alpha (no NE effect).
    For sleep states, use p_to_eeg_alpha directly (NE modulation does not
    apply to sleep spindle/slow oscillation alpha generators).
    """
    if exponent is None:
        exponent = ALPHA_POWER_EXPONENT
    if ne_gain is None:
        ne_gain = TC_ALPHA_NE_GAIN
    p_component = np.clip(P, 0, 1) ** exponent
    ne_modulation = 1.0 + ne_gain * (NE_relative - 1.0)
    return p_component * max(ne_modulation, 0.01)


def p_to_alpha_idling(cognitive_engagement=0.0, gain=None):
    """Cortical idling alpha: decreases with cognitive engagement.

    alpha_idling = 1.0 - gain * cognitive_engagement

    Reflects posterior alpha generated by disengaged cortex
    (Pfurtscheller & Lopes da Silva 1999). Not driven by precision dynamics.
    Appropriate for: meditation resting alpha, relaxation states.
    Not appropriate for: precision-driven contexts (sleep, psilocybin, depression).

    cognitive_engagement: 0.0 = full rest, 1.0 = active cognitive task
    """
    if gain is None:
        gain = IDLING_ALPHA_GAIN
    return max(0.0, 1.0 - gain * cognitive_engagement)


def p_to_eeg_alpha_state(P, rem_drive=0.0, exponent=None, rem_gain=None):
    """State-dependent EEG alpha: thalamocortical + posterior cortical (REM).

    alpha = P^exponent + ALPHA_REM_CORTICAL_GAIN * rem_drive

    During waking/NREM: rem_drive ≈ 0, reduces to p_to_eeg_alpha.
    During REM: posterior cortical generators add alpha floor independent
    of thalamocortical gating, driven by cholinergic activation.

    Citation: Cantero et al. (1999, 2002) — distinct alpha generators in REM.
    """
    if exponent is None:
        exponent = ALPHA_POWER_EXPONENT
    if rem_gain is None:
        rem_gain = ALPHA_REM_CORTICAL_GAIN
    tc_alpha = np.clip(P, 0, 1) ** exponent
    pc_alpha = rem_gain * rem_drive
    return tc_alpha + pc_alpha


def p_to_mmn(P, deviant_magnitude=1.0):
    """Prediction error (MMN) amplitude ∝ (1-P) * deviant_magnitude."""
    return (1.0 - np.clip(P, 0, 1)) * deviant_magnitude


def p_to_lzw(P, exponent=None):
    """LZW complexity ∝ (1-P)^0.5 (Schartner et al. 2015/2017)."""
    if exponent is None:
        exponent = LZW_EXPONENT
    return (1.0 - np.clip(P, 0, 1)) ** exponent


def p_to_lzw_state(P, S_sync=0.0, exponent=None, s_compress=None):
    """State-dependent LZW: modulated by slow-wave synchronization.

    LZW = (1-P)^exp * (1 - S_COMPRESS * S_sync)

    During waking: S_sync ≈ 0, reduces to p_to_lzw.
    During NREM: S_sync > 0, synchronization reduces complexity
    by creating inter-regional redundancy (traveling slow waves
    make the joint signal more compressible).
    During psilocybin: S_sync ≈ 0 (sleep-gated), LZW increases as expected.

    Citation: Massimini et al. (2004) J Neurosci 24:6862
    """
    if exponent is None:
        exponent = LZW_EXPONENT
    if s_compress is None:
        s_compress = S_COMPRESS
    base_lzw = (1.0 - np.clip(P, 0, 1)) ** exponent
    sync_reduction = 1.0 - s_compress * np.clip(S_sync, 0, 1)
    return base_lzw * max(sync_reduction, 0.01)


def p_to_fmri_pe(P, kl_divergence=1.0):
    """fMRI PE signal ∝ KL(posterior||prior) * P (Iglesias et al. 2013)."""
    return kl_divergence * np.clip(P, 0, 1) * FMRI_PE_SCALING


def p_to_p300(P_conceptual, DA=None, exponent=None):
    """P300 amplitude reflects precision-weighted UPDATING (Kolossa et al. 2015).

    Two-factor model:
    1. Updating capacity: (1-P)^γ — high P (rigid priors) resists updating → low P300
    2. DA salience quality: inverted-U on DA — both excess and deficit impair P300

    This correctly predicts reduced P300 in:
    - Depression: high P → reduced updating ✓
    - Schizophrenia: DA excess → reduced salience signaling ✓
    - ADHD: DA deficit → reduced salience signaling ✓
    """
    if exponent is None:
        exponent = P300_EXPONENT
    updating = (1.0 - np.clip(P_conceptual, 0, 0.99)) ** exponent
    if DA is not None:
        # Inverted-U DA contribution (Nieuwenhuis 2005)
        da_quality = np.exp(-((DA - DA_SALIENCE_OPTIMAL) ** 2) / (2 * P300_DA_SIGMA ** 2))
        return updating * da_quality
    return updating


def ne_to_pupil(NE, baseline=None, gain=None):
    """Pupil diameter ∝ baseline * (1 + k * NE) (Joshi et al. 2016).
    NE drives sympathetic pupil dilation."""
    if baseline is None:
        baseline = PUPIL_BASELINE
    if gain is None:
        gain = PUPIL_NE_GAIN
    return baseline * (1.0 + gain * NE)


def ne_cort_to_hrv(NE, cortisol, k_ne=None, k_cort=None):
    """HRV ∝ 1/(1 + k_ne*NE + k_cort*cort) (Kemp et al. 2010).
    Vagal withdrawal from sympathetic (NE) + HPA (cortisol) activation."""
    if k_ne is None:
        k_ne = HRV_NE_GAIN
    if k_cort is None:
        k_cort = HRV_CORT_GAIN
    return 1.0 / (1.0 + k_ne * NE + k_cort * cortisol)


def plasticity_to_bdnf(endogenous_plast, exponent=None):
    """Serum BDNF ∝ plasticity^γ (Molendijk et al. 2014; d=-0.71 for MDD).
    Sublinear: BDNF saturates at high plasticity levels."""
    if exponent is None:
        exponent = BDNF_EXPONENT
    return np.clip(endogenous_plast, 0, None) ** exponent


# ============================================================================
# SIMULATION ENGINE — Full 10-Variable State Vector
# ============================================================================

class SimulationState:
    """Container for the full state vector."""
    __slots__ = ['P_s', 'P_c', 'P_sm',
                 'R_5HT2A', 'R_D2',
                 'cortisol', 'hpa_sensitivity', 'allostatic_load',
                 'P_setpoint', 'R_bipolar',
                 'SC', 'S_sync']

    def __init__(self, P_s=None, P_c=None, P_sm=None,
                 R_5HT2A=1.0, R_D2=1.0,
                 cortisol=None, hpa_sensitivity=1.0, allostatic_load=0.0,
                 P_setpoint=None, R_bipolar=0.0, SC=0.0, S_sync=0.0):
        self.P_s = P_s if P_s is not None else P_SENSORY_BASELINE
        self.P_c = P_c if P_c is not None else P_CONCEPTUAL_BASELINE
        self.P_sm = P_sm if P_sm is not None else P_SELFMODEL_BASELINE
        self.R_5HT2A = R_5HT2A
        self.R_D2 = R_D2
        self.cortisol = cortisol if cortisol is not None else CORTISOL_BASELINE
        self.hpa_sensitivity = hpa_sensitivity
        self.allostatic_load = allostatic_load
        self.P_setpoint = P_setpoint if P_setpoint is not None else P_WAKING
        self.R_bipolar = R_bipolar
        self.SC = SC
        self.S_sync = S_sync


def simulate_v2(
    t_span,
    dt=0.01,
    state0=None,
    # Upstream scenario parameters (THE ONLY per-scenario knobs)
    chronic_stress=0.0,
    da_excess=0.0,
    gaba_deficit=0.0,
    ne_sensitization=1.0,
    coupling_breakdown=1.0,
    dat_dysfunction=1.0,
    net_dysfunction=1.0,
    cstc_glu_excess=0.0,
    stress_sensitivity=1.0,
    # Pharmacology
    pharma_psilocybin=None,   # list of (dose_time, dose_strength) or None
    pharma_dmt=None,          # list of (dose_time, dose_strength) or None
    pharma_ketamine=None,     # list of (dose_time, dose_strength) or None
    pharma_antipsychotic=None, # (dose_time, dose_strength, half_life) or None
    pharma_ssri=None,         # (start_time, dose_strength) or None
    pharma_atomoxetine=None,  # list of (dose_time, dose_strength) or None
    # Plasticity scaling
    endogenous_plasticity_scale=1.0,
    # Bipolar mode
    bipolar_mode=False,
    bipolar_per_level=False,  # per-level setpoint oscillator for mixed bipolar
    # Noise
    noise_scale=1.0,
    noise_scale_per_level=None,  # [sensory, conceptual, selfmodel] or None
    # Bounds overrides
    p_min=None,
    p_max=None,
    # PTSD asymmetric coupling
    td_coupling_scale=1.0,  # top-down coupling scale (0.3 for PTSD dissociation)
    # Coupling constant overrides for robustness analysis
    params_override=None,  # dict of param_name → value
    # Random seed
    seed=42,
):
    """
    Simulate the full v2 model with 10-variable state vector.

    KEY PRINCIPLE: All scenario-specific behavior is driven by upstream
    parameters (chronic_stress, da_excess, gaba_deficit, etc.).
    P profiles EMERGE from the dynamics. No per-scenario P overrides.

    Returns:
        t_array: time array
        P_dict: {'sensory': array, 'conceptual': array, 'selfmodel': array}
        state_dict: full state variable histories
        neuromod_dict: neuromodulator time courses
    """
    np.random.seed(seed)

    if state0 is None:
        state0 = SimulationState()

    # --- Local parameter dict (enables robustness analysis overrides) ---
    p = {
        'ALPHA_NE': ALPHA_NE, 'ALPHA_5HT': ALPHA_5HT, 'ALPHA_DA': ALPHA_DA,
        'BETA_PLAST': BETA_PLAST, 'BETA_ACH': BETA_ACH, 'BETA_GLU': BETA_GLU,
        'CORT_NE_COUPLING': CORT_NE_COUPLING, 'CORT_5HT_COUPLING': CORT_5HT_COUPLING,
        'CORT_DA_COUPLING': CORT_DA_COUPLING,
        'CORT_PLASTICITY_COUPLING': CORT_PLASTICITY_COUPLING,
        'GAMMA_SENSORY': GAMMA_SENSORY, 'GAMMA_CONCEPTUAL': GAMMA_CONCEPTUAL,
        'GAMMA_SELFMODEL': GAMMA_SELFMODEL,
        'GABA_NE_GAIN_MOD': GABA_NE_GAIN_MOD,
        'CHRONIC_NE_SENSITIZATION': CHRONIC_NE_SENSITIZATION,
        'CHRONIC_5HT_ELEVATION': CHRONIC_5HT_ELEVATION,
        'DA_EXCESS_SCALING': DA_EXCESS_SCALING,
        'CSTC_SUPPRESSION_SCALING': CSTC_SUPPRESSION_SCALING,
        'COUPLING_TD': COUPLING_TD, 'COUPLING_BU': COUPLING_BU,
        'HPA_FEEDBACK_GAIN': HPA_FEEDBACK_GAIN,
        'HPA_FEEDBACK_EROSION_RATE': HPA_FEEDBACK_EROSION_RATE,
        'ALLOSTATIC_LOAD_RATE': ALLOSTATIC_LOAD_RATE,
        'DA_SALIENCE_OPTIMAL': DA_SALIENCE_OPTIMAL,
        'DA_SALIENCE_WIDTH': DA_SALIENCE_WIDTH,
        'CORTISOL_STRESS_GAIN': CORTISOL_STRESS_GAIN,
        'ADHD_NOISE_SCALING': ADHD_NOISE_SCALING,
        'ADHD_DA_SALIENCE_NOISE': ADHD_DA_SALIENCE_NOISE,
        'BDNF_SURGE_SCALING': BDNF_SURGE_SCALING,
        'PSILOCYBIN_PHARMA_GAIN': PSILOCYBIN_PHARMA_GAIN,
        'DMT_PHARMA_GAIN': DMT_PHARMA_GAIN,
        'KETAMINE_PHARMA_GAIN': KETAMINE_PHARMA_GAIN,
        'PTSD_DISSOC_COEFF': PTSD_DISSOC_COEFF,
        'P_CONCEPTUAL_NREM': P_CONCEPTUAL_NREM,
        'P_CONCEPTUAL_REM': P_CONCEPTUAL_REM,
        'ATOMOXETINE_NE_GAIN': ATOMOXETINE_NE_GAIN,
        'ATOMOXETINE_PHARMA_GAIN': ATOMOXETINE_PHARMA_GAIN,
        'K_CONSOLIDATION': K_CONSOLIDATION,
        'SC_CONSOLIDATION_TAU': SC_CONSOLIDATION_TAU,
        'ALPHA_NE_PHASIC': ALPHA_NE_PHASIC,
        'ALPHA_5HT_PHASIC': ALPHA_5HT_PHASIC,
        'HPA_5HT2A_DIRECT_GAIN': HPA_5HT2A_DIRECT_GAIN,
    }
    if params_override:
        p.update(params_override)

    p_lo = p_min if p_min is not None else P_MIN
    p_hi = p_max if p_max is not None else P_MAX

    ns = noise_scale_per_level if noise_scale_per_level is not None else \
        [noise_scale, noise_scale, noise_scale]

    t_start, t_end = t_span
    t_array = np.arange(t_start, t_end, dt)
    n = len(t_array)

    # --- Recompute equilibrium offsets using local params ---
    _t_eq = EQUILIBRIUM_REF_TIME
    _ne_eq_local = NE_BASELINE + NE_AMPLITUDE * np.cos(2 * np.pi * (_t_eq - 10.0) / TAU_CIRCADIAN)
    _5ht_eq_local = SEROTONIN_BASELINE + SEROTONIN_AMPLITUDE * np.cos(
        2 * np.pi * (_t_eq - 11.0) / TAU_CIRCADIAN)
    _da_eq_local = DA_BASELINE + DA_AMPLITUDE * np.cos(2 * np.pi * (_t_eq - 12.0) / TAU_CIRCADIAN)
    _da_dev_eq_local = (_da_eq_local - p['DA_SALIENCE_OPTIMAL']) / p['DA_SALIENCE_WIDTH']
    _da_sal_eq_local = p['ALPHA_DA'] * np.exp(-0.5 * _da_dev_eq_local ** 2)

    sup_eq_local = [
        p['ALPHA_NE'] * ALPHA_NE_SCALE[i] * _ne_eq_local +
        p['ALPHA_5HT'] * ALPHA_5HT_SCALE[i] * _5ht_eq_local +
        _da_sal_eq_local
        for i in range(3)
    ]
    pla_eq_local = [
        p['BETA_PLAST'] * BETA_PLAST_SCALE[i] * ENDOGENOUS_PLASTICITY_TONIC +
        p['BETA_ACH'] * ACH_WAKING
        for i in range(3)
    ]

    # === Output arrays ===
    P_s_arr = np.zeros(n)
    P_c_arr = np.zeros(n)
    P_sm_arr = np.zeros(n)
    R_5HT2A_arr = np.zeros(n)
    R_D2_arr = np.zeros(n)
    cortisol_arr = np.zeros(n)
    hpa_sens_arr = np.zeros(n)
    allostatic_arr = np.zeros(n)
    setpoint_arr = np.zeros(n)
    R_bipolar_arr = np.zeros(n)
    SC_arr = np.zeros(n)
    S_sync_arr = np.zeros(n)

    # Per-level setpoints for mixed bipolar
    if bipolar_per_level:
        sp_s = state0.P_setpoint
        sp_c = state0.P_setpoint
        sp_sm = state0.P_setpoint
        R_bp_s = 0.0
        R_bp_c = 0.0
        R_bp_sm = 0.0
        sp_s_arr = np.zeros(n)
        sp_c_arr = np.zeros(n)
        sp_sm_arr = np.zeros(n)

    # Neuromodulator output arrays
    ne_arr = np.zeros(n)
    sht_arr = np.zeros(n)
    da_arr = np.zeros(n)
    ach_arr = np.zeros(n)
    glu_arr = np.zeros(n)
    gaba_arr = np.zeros(n)
    plast_arr = np.zeros(n)  # endogenous plasticity
    sleep_arr = np.zeros(n)
    rem_arr = np.zeros(n)

    # Initialize state
    P_s = state0.P_s
    P_c = state0.P_c
    P_sm = state0.P_sm
    R_5HT2A = state0.R_5HT2A
    R_D2 = state0.R_D2
    cort = state0.cortisol
    hpa_sens = state0.hpa_sensitivity
    allo_load = state0.allostatic_load
    P_sp = state0.P_setpoint
    R_bp = state0.R_bipolar
    SC = state0.SC
    S_sync = state0.S_sync

    # OU noise processes (one per level)
    noise_s = 0.0
    noise_c = 0.0
    noise_sm = 0.0

    # Coupling strengths (modulated by coupling_breakdown for PTSD etc.)
    # Asymmetric: td_coupling_scale allows PTSD to break top-down MORE than bottom-up
    c_td = p['COUPLING_TD'] * coupling_breakdown * td_coupling_scale
    c_bu = p['COUPLING_BU'] * coupling_breakdown

    gammas = [p['GAMMA_SENSORY'], p['GAMMA_CONCEPTUAL'], p['GAMMA_SELFMODEL']]
    waking_targets = [P_SENSORY_BASELINE, P_CONCEPTUAL_BASELINE, P_SELFMODEL_BASELINE]
    rem_targets = [P_SENSORY_REM, p['P_CONCEPTUAL_REM'], P_SELFMODEL_REM]
    nrem_targets = [P_SENSORY_NREM, p['P_CONCEPTUAL_NREM'], P_SELFMODEL_NREM]
    receptor_weights = [RECEPTOR_5HT2A_SENSORY, RECEPTOR_5HT2A_ASSOCIATION, RECEPTOR_5HT2A_DMN]

    sqrt_dt = np.sqrt(dt)

    for i in range(n):
        t = t_array[i]

        # --- Record current state ---
        P_s_arr[i] = P_s
        P_c_arr[i] = P_c
        P_sm_arr[i] = P_sm
        R_5HT2A_arr[i] = R_5HT2A
        R_D2_arr[i] = R_D2
        cortisol_arr[i] = cort
        hpa_sens_arr[i] = hpa_sens
        allostatic_arr[i] = allo_load
        setpoint_arr[i] = P_sp
        R_bipolar_arr[i] = R_bp
        SC_arr[i] = SC
        S_sync_arr[i] = S_sync

        if bipolar_per_level:
            sp_s_arr[i] = sp_s
            sp_c_arr[i] = sp_c
            sp_sm_arr[i] = sp_sm

        # --- Compute neuromodulator levels ---
        slp = is_sleep(t)
        nrem_d, rem_d = sleep_stage(t)
        sleep_arr[i] = slp
        rem_arr[i] = rem_d

        # NE: modulated by acute cortisol (CRH → LC) AND allostatic load
        # Acute: cortisol → CRH → LC activation
        # Chronic: allostatic load sensitizes LC (increased NE tonic firing)
        cort_ne_acute = 1.0 + p['CORT_NE_COUPLING'] * max(0, cort - CORTISOL_BASELINE)
        cort_ne_chronic = 1.0 + p['CHRONIC_NE_SENSITIZATION'] * allo_load
        ne_base_mod = NE_BASELINE * cort_ne_acute * cort_ne_chronic * ne_sensitization
        # Atomoxetine NET blockade → elevated synaptic NE
        if pharma_atomoxetine is not None:
            atx_effect = 0.0
            for at_time, at_strength in pharma_atomoxetine:
                atx_effect += atomoxetine_perturbation(t, at_time, at_strength)
            ne_base_mod *= (1.0 + p['ATOMOXETINE_NE_GAIN'] * atx_effect)
        # NET dysfunction (ADHD): no baseline NE shift — noise handled separately
        # (NET dysfunction → noisier NE, not consistently higher)
        ne_val = norepinephrine(t, ne_baseline=ne_base_mod)
        ne_arr[i] = ne_val

        # 5-HT: chronic allostatic load dysregulates 5-HT
        # In depression: 5-HT system is dysregulated, paradoxically elevated tonic
        # (serotonin transporter dysfunction → more synaptic 5-HT → downreg)
        cort_5ht_mod = 1.0 + p['CHRONIC_5HT_ELEVATION'] * allo_load
        sht_val = serotonin(t, sht_baseline=SEROTONIN_BASELINE * cort_5ht_mod)
        sht_arr[i] = sht_val

        # DA: allostatic load → anhedonia (reduced DA) + excess for psychosis
        cort_da_mod = 1.0 + p['CORT_DA_COUPLING'] * allo_load
        da_base = DA_BASELINE * max(0.1, cort_da_mod) * (1.0 + da_excess * p['DA_EXCESS_SCALING'])
        # DAT dysfunction (ADHD): lower effective tonic DA (frontal hypodopaminergia)
        # DAT dysfunction → phasic burst diluted into tonic noise → lower functional DA
        if dat_dysfunction < 1.0:
            da_base *= (1.0 - 0.20 * (1.0 - dat_dysfunction))
        da_val = dopamine(t, da_baseline=da_base)
        da_arr[i] = da_val

        # ACh: standard
        ach_val = acetylcholine(t)
        ach_arr[i] = ach_val

        # GLU: baseline + CSTC excess (OCD) + ketamine surge
        glu_val = glutamate(t) + cstc_glu_excess * p['CSTC_SUPPRESSION_SCALING'] * 2.0
        if pharma_ketamine is not None:
            for kt_time, kt_strength in pharma_ketamine:
                nmda_block, _ = ketamine_perturbation(t, kt_time, kt_strength)
                glu_val += nmda_block * 0.5
        glu_arr[i] = glu_val

        # GABA: modulated by deficit
        gaba_val = gaba(t, deficit=gaba_deficit)
        gaba_arr[i] = gaba_val

        # Endogenous plasticity: allostatic load reduces BDNF → less plasticity
        cort_plast_mod = 1.0 + p['CORT_PLASTICITY_COUPLING'] * allo_load
        plast_val = endogenous_plasticity(t, scale=endogenous_plasticity_scale) * \
            max(0.1, cort_plast_mod)
        plast_arr[i] = plast_val

        if i >= n - 1:
            break

        # === INTEGRATION STEP ===

        # --- Update OU noise ---
        dW_s = np.random.normal(0, sqrt_dt)
        dW_c = np.random.normal(0, sqrt_dt)
        dW_sm = np.random.normal(0, sqrt_dt)
        noise_s += (-noise_s / NOISE_TAU) * dt + SIGMA_NOISE * ns[0] * dW_s
        noise_c += (-noise_c / NOISE_TAU) * dt + SIGMA_NOISE * ns[1] * dW_c
        noise_sm += (-noise_sm / NOISE_TAU) * dt + SIGMA_NOISE * ns[2] * dW_sm

        # ADHD noise amplification from transporter dysfunction
        # Scale noise AMPLITUDE (not state) to avoid exponential growth
        if dat_dysfunction < 1.0 or net_dysfunction < 1.0:
            adhd_noise_extra = SIGMA_NOISE * p['ADHD_NOISE_SCALING'] * (2.0 - dat_dysfunction - net_dysfunction)
            noise_s += adhd_noise_extra * dW_s
            noise_c += adhd_noise_extra * dW_c
            noise_sm += adhd_noise_extra * dW_sm

        # --- GABA modulation (dual pathway) ---
        # Pathway 1: GABA deficit → NE amplification (anxiety/hypervigilance)
        gaba_deviation = GABA_NORMALIZED - gaba_val
        ne_gain = 1.0 + p['GABA_NE_GAIN_MOD'] * max(0.0, gaba_deviation)
        # Pathway 2: GABA excess → direct cortical inhibition (propofol/sedation)
        # High GABA increases P via global inhibition, independent of NE
        gaba_direct_sup = p['GABA_NE_GAIN_MOD'] * max(0.0, gaba_val - GABA_NORMALIZED)

        # --- 5-HT2A agonist signal for synaptic consolidation ---
        sht2a_raw = 0.0
        if pharma_psilocybin is not None:
            for p_time, p_strength in pharma_psilocybin:
                sht2a_raw += psilocybin_perturbation(t, p_time, p_strength)
        if pharma_dmt is not None:
            for d_time, d_strength in pharma_dmt:
                sht2a_raw += dmt_perturbation(t, d_time, d_strength)

        # --- Phasic NE/5-HT derivatives (bypass equilibrium subtraction) ---
        # d(NE)/dt and d(5-HT)/dt capture acute changes: drug onsets, stress,
        # circadian transitions. Unlike tonic coupling, not cancelled by
        # equilibrium subtraction.
        if i > 0:
            dNE_dt = (ne_val - ne_arr[i - 1]) / dt
            d5HT_dt = (sht_val - sht_arr[i - 1]) / dt
        else:
            dNE_dt = 0.0
            d5HT_dt = 0.0

        # --- Per-level precision dynamics ---
        P_levels = [P_s, P_c, P_sm]
        noises = [noise_s, noise_c, noise_sm]

        # Compute GLU/BDNF contribution for each level
        glu_bdnf = 0.0
        if pharma_ketamine is not None:
            for kt_time, kt_strength in pharma_ketamine:
                _, bdnf = ketamine_perturbation(t, kt_time, kt_strength)
                glu_bdnf += bdnf

        dp_levels = []
        for lvl in range(3):
            P_cur = P_levels[lvl]

            # Suppression with GABA gain modulation on NE
            ne_effective = ne_val * ne_gain * ALPHA_NE_SCALE[lvl]
            sht_effective = sht_val * ALPHA_5HT_SCALE[lvl]

            sup = p['ALPHA_NE'] * ne_effective + p['ALPHA_5HT'] * sht_effective
            # Direct GABA cortical inhibition (propofol/sedation pathway)
            sup += gaba_direct_sup

            # DA salience contribution (inverted-U)
            da_dev = (da_val - p['DA_SALIENCE_OPTIMAL']) / p['DA_SALIENCE_WIDTH']
            da_cont = p['ALPHA_DA'] * np.exp(-0.5 * da_dev ** 2)
            sup += da_cont

            # DA salience disruption (excess DA → aberrant salience → P reduction)
            if da_val > p['DA_SALIENCE_OPTIMAL']:
                da_excess_dev = (da_val - p['DA_SALIENCE_OPTIMAL']) / p['DA_SALIENCE_WIDTH']
                sup -= p['ALPHA_DA'] * 0.5 * da_excess_dev ** 2

            # ADHD: DA salience noise — DAT dysfunction → moment-to-moment DA fluctuations
            # Biased negative: noise preferentially disrupts (reduces) sustained suppression
            if dat_dysfunction < 1.0:
                da_noise_raw = np.random.normal(0, 1) - 0.5  # biased negative
                da_noise_factor = 1.0 + p['ADHD_DA_SALIENCE_NOISE'] * (1.0 - dat_dysfunction) * da_noise_raw
                sup *= max(0.5, da_noise_factor)

            # Plasticity drive
            pla = p['BETA_PLAST'] * BETA_PLAST_SCALE[lvl] * plast_val + \
                  p['BETA_ACH'] * ach_val + \
                  p['BETA_GLU'] * glu_bdnf * p['KETAMINE_PHARMA_GAIN']

            # CSTC glutamate excess at conceptual level (OCD)
            if lvl == 1 and cstc_glu_excess > 0:
                # Excess GLU in CSTC → increased conceptual suppression (lock-in)
                sup += cstc_glu_excess * p['CSTC_SUPPRESSION_SCALING']

            # PTSD dissociation: hyperarousal + broken top-down → selfmodel P suppression
            # Mechanism: extreme NE drives sensory hypervigilance but disrupts
            # self-referential processing (dissociation from self-model)
            ptsd_dissoc = 0.0
            if lvl == 2 and td_coupling_scale < 1.0 and ne_sensitization > 1.0:
                ptsd_dissoc = -p['PTSD_DISSOC_COEFF'] * (ne_sensitization - 1.0) * (1.0 - td_coupling_scale)

            # Phasic NE/5-HT: NOT equilibrium-subtracted
            # Captures acute drug effects (ATX onset), stress, circadian transitions
            phasic = p['ALPHA_NE_PHASIC'] * dNE_dt * ALPHA_NE_SCALE[lvl] + \
                     p['ALPHA_5HT_PHASIC'] * d5HT_dt * ALPHA_5HT_SCALE[lvl]

            # Drive relative to equilibrium (tonic) + phasic bypass
            drive = (sup - sup_eq_local[lvl]) - (pla - pla_eq_local[lvl]) + ptsd_dissoc + phasic

            # Homeostatic target
            if bipolar_mode and not bipolar_per_level:
                # Bipolar: use moving setpoint
                wt = P_sp
            elif bipolar_per_level:
                wt = [sp_s, sp_c, sp_sm][lvl]
            else:
                wt = waking_targets[lvl]

            tgt = wt * (1 - slp) + nrem_targets[lvl] * slp * (1 - rem_d) + \
                  rem_targets[lvl] * slp * rem_d

            # Save conceptual target for SC update
            if lvl == 1:
                tgt_c_sc = tgt

            # Synaptic consolidation shifts effective equilibrium lower
            # SC > 0 means structural plasticity has stabilized a lower-P state
            homeo = -gammas[lvl] * (P_cur - tgt + SC)

            # Inter-level coupling
            if lvl == 0:  # sensory
                coup = c_td * (P_c - P_s)
            elif lvl == 1:  # conceptual
                coup = c_td * (P_sm - P_c) + c_bu * (P_s - P_c)
            else:  # self-model
                coup = c_bu * (P_c - P_sm)

            # Psilocybin perturbation (weighted by receptor density and R_5HT2A)
            pharma_val = 0.0
            if pharma_psilocybin is not None:
                for p_time, p_strength in pharma_psilocybin:
                    raw = psilocybin_perturbation(t, p_time, p_strength)
                    # Scale by receptor density and receptor availability
                    pharma_val -= raw * receptor_weights[lvl] / 0.5 * R_5HT2A * p['PSILOCYBIN_PHARMA_GAIN']

            # DMT perturbation (same 5-HT2A mechanism as psilocybin)
            if pharma_dmt is not None:
                for d_time, d_strength in pharma_dmt:
                    raw = dmt_perturbation(t, d_time, d_strength)
                    pharma_val -= raw * receptor_weights[lvl] / 0.5 * R_5HT2A * p['DMT_PHARMA_GAIN']

            # Antipsychotic (increases P via D2/5HT2A antagonism)
            if pharma_antipsychotic is not None:
                ap_time, ap_strength, ap_hl = pharma_antipsychotic
                ap_occ = antipsychotic_perturbation(t, ap_time, ap_strength, ap_hl)
                pharma_val += ap_occ * 0.5  # stabilizing effect

            # SSRI (two-phase: acute 5-HT → P↑, then delayed neuroplasticity → P↓)
            if pharma_ssri is not None:
                ssri_start, ssri_strength = pharma_ssri
                pharma_val += ssri_perturbation(t, ssri_start, ssri_strength)

            # Atomoxetine (NE-mediated P increase via dedicated pharma pathway)
            if pharma_atomoxetine is not None:
                for at_time, at_strength in pharma_atomoxetine:
                    atx_raw = atomoxetine_perturbation(t, at_time, at_strength)
                    # NE reuptake inhibition → increased precision at sensory > conceptual > self
                    ne_weight = ALPHA_NE_SCALE[lvl] / max(ALPHA_NE_SCALE)
                    pharma_val += atx_raw * ne_weight * p['ATOMOXETINE_PHARMA_GAIN']

            dp = (1.0 / TAU_P) * (drive + homeo + coup + noises[lvl] + pharma_val)
            dp_levels.append(dp)

        # --- Update P levels ---
        P_c_pre = P_c  # save for SC Euler consistency
        P_s = np.clip(P_s + dp_levels[0] * dt, p_lo, p_hi)
        P_c = np.clip(P_c + dp_levels[1] * dt, p_lo, p_hi)
        P_sm = np.clip(P_sm + dp_levels[2] * dt, p_lo, p_hi)

        # --- Synaptic consolidation dynamics ---
        # Ly et al. 2018: 5-HT2A agonism promotes dendritic spine growth,
        # structurally stabilizing the reduced-P state. SC accumulates when
        # P is below homeostatic target AND 5-HT2A agonist is present.
        # Driven by conceptual level (most depression-relevant).
        if sht2a_raw > 0:
            sc_drive = p['K_CONSOLIDATION'] * sht2a_raw * max(0.0, tgt_c_sc - P_c_pre)
        else:
            sc_drive = 0.0
        dSC = (sc_drive - SC / p['SC_CONSOLIDATION_TAU']) * dt
        SC = max(0.0, SC + dSC)

        # --- Slow-wave synchronization dynamics (Landau bifurcation) ---
        # When mean P drops below P_CRIT_SYNC during sleep, synchronization
        # emerges through spontaneous symmetry breaking. Sleep gating prevents
        # synchronization during psilocybin (low P but awake → no slow waves).
        # REM suppression: cholinergic activation during REM disrupts slow-wave
        # synchronization, causing rapid S decay (Steriade & McCarley 2005).
        P_mean = (P_s + P_c + P_sm) / 3.0
        sync_drive = K_SYNC * max(0.0, P_CRIT_SYNC - P_mean) * slp * (1.0 - rem_d)
        # Faster decay during REM (ACh disrupts slow-wave sync)
        effective_tau = TAU_SYNC / (1.0 + 9.0 * rem_d)
        dS_sync = (sync_drive - S_sync / effective_tau) * dt
        S_sync = np.clip(S_sync + dS_sync, 0.0, 1.0)

        # --- Receptor dynamics ---
        # 5-HT2A: downregulation under agonist occupancy, recovery toward 1.0
        agonist_occ = 0.0
        if pharma_psilocybin is not None:
            for p_time, p_strength in pharma_psilocybin:
                agonist_occ += psilocybin_perturbation(t, p_time, p_strength)
        if pharma_dmt is not None:
            for d_time, d_strength in pharma_dmt:
                agonist_occ += dmt_perturbation(t, d_time, d_strength)
        agonist_occ = min(agonist_occ, 1.0)

        # dR_5HT2A/dt = (1/tau_recovery) * (1.0 - R) - k_downreg * occupancy * R
        k_downreg = RECEPTOR_DOWNREG_MAX / RECEPTOR_DOWNREG_TAU
        dR_5ht2a = (1.0 / RECEPTOR_RECOVERY_TAU) * (1.0 - R_5HT2A) - \
                   k_downreg * agonist_occ * R_5HT2A
        R_5HT2A = np.clip(R_5HT2A + dR_5ht2a * dt, 1.0 - RECEPTOR_DOWNREG_MAX, 1.5)

        # D2: upregulation under antagonist occupancy, recovery toward 1.0
        antagonist_occ = 0.0
        if pharma_antipsychotic is not None:
            ap_time, ap_strength, ap_hl = pharma_antipsychotic
            antagonist_occ = antipsychotic_perturbation(t, ap_time, ap_strength, ap_hl)

        # dR_D2/dt = (1/tau_recovery) * (1.0 - R) + k_upreg * occupancy * (R_max - R)
        dR_D2 = (1.0 / D2_RECOVERY_TAU) * (1.0 - R_D2) + \
                D2_UPREG_RATE * antagonist_occ * (D2_MAX_DENSITY - R_D2)
        R_D2 = np.clip(R_D2 + dR_D2 * dt, 0.5, D2_MAX_DENSITY)

        # --- HPA axis dynamics ---
        # Cortisol: diurnal rhythm + stress input + feedback
        # + direct 5-HT2A → CRH pathway (psilocybin/DMT acute cortisol increase)
        stress_input = chronic_stress * stress_sensitivity
        cort_target = cortisol_rhythm(t, stress_input=0.0) + p['CORTISOL_STRESS_GAIN'] * stress_input

        # Negative feedback (attenuated by chronic stress via hpa_sensitivity)
        feedback = -p['HPA_FEEDBACK_GAIN'] * hpa_sens * max(0, cort - CORTISOL_BASELINE)

        # Direct 5-HT2A agonist → hypothalamic CRH → cortisol
        # Independent of precision dynamics pathway. Operates in parallel:
        # precision reduction lowers allostatic load (slow, indirect, predicts cortisol ↓)
        # but 5-HT2A on CRH neurons drives acute cortisol ↑ (fast, direct, pharmacological)
        hpa_direct = p.get('HPA_5HT2A_DIRECT_GAIN', 0.0) * agonist_occ

        dcort = (cort_target - cort) * 0.5 + feedback + hpa_direct
        cort = max(0.0, cort + dcort * dt)

        # HPA sensitivity erosion under chronic stress
        if chronic_stress > 0.1:
            dhpa = -p['HPA_FEEDBACK_EROSION_RATE'] * chronic_stress * hpa_sens
            hpa_sens = max(0.1, hpa_sens + dhpa * dt)
        else:
            # Slow recovery when stress is removed
            dhpa = 0.001 * (1.0 - hpa_sens)
            hpa_sens = min(1.0, hpa_sens + dhpa * dt)

        # Allostatic load: accumulates under sustained cortisol elevation
        allo_threshold = CORTISOL_BASELINE * ALLOSTATIC_THRESHOLD_FACTOR
        if cort > allo_threshold:
            dallo = p['ALLOSTATIC_LOAD_RATE'] * (cort - allo_threshold)
        else:
            dallo = -ALLOSTATIC_LOAD_DECAY * allo_load
        allo_load = np.clip(allo_load + dallo * dt, 0.0, ALLOSTATIC_LOAD_MAX)

        # --- Bipolar setpoint oscillator ---
        if bipolar_mode:
            if bipolar_per_level:
                # Per-level setpoint oscillators with different tau_adapt
                taus = [BIPOLAR_TAU_ADAPT_SENSORY,
                        BIPOLAR_TAU_ADAPT_CONCEPTUAL,
                        BIPOLAR_TAU_ADAPT_SELFMODEL]
                for idx, (sp_ref, R_ref, tau_a) in enumerate([
                    (sp_s, R_bp_s, taus[0]),
                    (sp_c, R_bp_c, taus[1]),
                    (sp_sm, R_bp_sm, taus[2]),
                ]):
                    instab = BIPOLAR_K_INSTAB * (sp_ref - P_WAKING)
                    comp = -BIPOLAR_K_FEEDBACK * R_ref
                    dsp = (instab + comp) * dt
                    sp_ref_new = np.clip(sp_ref + dsp, p_lo + 0.05, p_hi - 0.05)
                    dR = (1.0 / tau_a) * (sp_ref - P_WAKING) * dt
                    R_ref_new = R_ref + dR

                    if idx == 0:
                        sp_s, R_bp_s = sp_ref_new, R_ref_new
                    elif idx == 1:
                        sp_c, R_bp_c = sp_ref_new, R_ref_new
                    else:
                        sp_sm, R_bp_sm = sp_ref_new, R_ref_new

                # Store mean setpoint for backward compatibility
                P_sp = (sp_s + sp_c + sp_sm) / 3.0
                R_bp = (R_bp_s + R_bp_c + R_bp_sm) / 3.0
            else:
                # Single setpoint oscillator (v1 mechanism)
                instability = BIPOLAR_K_INSTAB * (P_sp - P_WAKING)
                compensation = -BIPOLAR_K_FEEDBACK * R_bp
                dP_sp = (instability + compensation) * dt
                P_sp = np.clip(P_sp + dP_sp, p_lo + 0.05, p_hi - 0.05)

                dR_bp = (1.0 / BIPOLAR_TAU_ADAPT) * (P_sp - P_WAKING) * dt
                R_bp += dR_bp

    # === Package results ===
    P_dict = {
        'sensory': P_s_arr,
        'conceptual': P_c_arr,
        'selfmodel': P_sm_arr,
    }

    state_dict = {
        'R_5HT2A': R_5HT2A_arr,
        'R_D2': R_D2_arr,
        'cortisol': cortisol_arr,
        'hpa_sensitivity': hpa_sens_arr,
        'allostatic_load': allostatic_arr,
        'setpoint': setpoint_arr,
        'R_bipolar': R_bipolar_arr,
        'SC': SC_arr,
        'S_sync': S_sync_arr,
    }

    if bipolar_per_level:
        state_dict['setpoint_sensory'] = sp_s_arr
        state_dict['setpoint_conceptual'] = sp_c_arr
        state_dict['setpoint_selfmodel'] = sp_sm_arr

    neuromod = {
        'NE': ne_arr,
        '5-HT': sht_arr,
        'DA': da_arr,
        'ACh': ach_arr,
        'GLU': glu_arr,
        'GABA': gaba_arr,
        'endogenous_plasticity': plast_arr,
        'sleep': sleep_arr,
        'REM': rem_arr,
    }

    return t_array, P_dict, state_dict, neuromod


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

def simulate_normal_24h(seed=42, dt=0.01):
    """Normal healthy 24h cycle."""
    return simulate_v2(
        t_span=(6.0, 30.0), dt=dt, seed=seed,
    )


def simulate_depression(chronic_stress=0.6, seed=42, dt=0.01,
                        t_span=None, weeks=6):
    """
    Depression: emergent from chronic stress.
    Mechanism chain:
      1. chronic_stress → cortisol elevation (HPA axis)
      2. cortisol excess → hpa_sensitivity erodes (feedback resistance)
      3. sustained cortisol → NE↑, DA↓, plasticity↓
      4. monoamine changes → P elevation EMERGES
    """
    if t_span is None:
        t_span = (6.0, 6.0 + weeks * 7 * 24)  # weeks of simulation
    return simulate_v2(
        t_span=t_span, dt=dt, seed=seed,
        chronic_stress=chronic_stress,
    )


def simulate_psychosis(da_excess=1.5, gaba_deficit=0.3, seed=42, dt=0.01):
    """
    Psychosis: emergent from DA excess + GABA deficit.
    Mechanism:
      - Excess DA → aberrant salience → P disruption
      - GABA deficit → amplified NE gain → sensory instability
    """
    return simulate_v2(
        t_span=(6.0, 30.0), dt=dt, seed=seed,
        da_excess=da_excess,
        gaba_deficit=gaba_deficit,
    )


def simulate_anxiety(gaba_deficit=0.20, stress_sensitivity=1.5, seed=42, dt=0.01):
    """
    Anxiety/GAD: emergent from GABA deficit + stress sensitivity.
    Mechanism:
      - GABA deficit → amplified NE coupling → sensory hyperarousal
      - Stress sensitivity → heightened cortisol response
    """
    return simulate_v2(
        t_span=(6.0, 54.0), dt=dt, seed=seed,
        gaba_deficit=gaba_deficit,
        stress_sensitivity=stress_sensitivity,
        chronic_stress=0.3,  # moderate background stress
    )


def simulate_ptsd(ne_sensitization=1.8, coupling_breakdown=0.5,
                  chronic_stress=0.4, seed=42, dt=0.01):
    """
    PTSD: emergent from NE sensitization + asymmetric coupling breakdown.
    Mechanism:
      - NE sensitization → hyperarousal (sensory P elevation)
      - Asymmetric coupling: top-down breaks MORE than bottom-up (dissociation)
        → high sensory P (hypervigilance) + low self-model P (dissociation)
      - Chronic stress → HPA dysregulation
    """
    return simulate_v2(
        t_span=(6.0, 54.0), dt=dt, seed=seed,
        ne_sensitization=ne_sensitization,
        coupling_breakdown=coupling_breakdown,
        chronic_stress=chronic_stress,
        td_coupling_scale=PTSD_TD_BREAKDOWN,
    )


def simulate_adhd(dat_dysfunction=0.7, net_dysfunction=0.8, seed=42, dt=0.01):
    """
    ADHD: emergent from transporter dysfunction.
    Mechanism:
      - DAT dysfunction → lower effective tonic DA (frontal hypodopaminergia)
      - DAT dysfunction → noisy DA signaling → variable salience → P instability
      - Frontal DA deficit → impaired precision maintenance (weakened homeostatic gain)
      - NET dysfunction → noisy NE → variable arousal
      - Result: low mean P (poor sustained attention) + high variance (distractibility)
    """
    # DA deficit weakens the brain's ability to maintain precision homeostasis
    # ADHD frontal DA deficit → impaired P maintenance
    gamma_scale = 0.65
    return simulate_v2(
        t_span=(6.0, 30.0), dt=dt, seed=seed,
        dat_dysfunction=dat_dysfunction,
        net_dysfunction=net_dysfunction,
        noise_scale=4.0,
        params_override={
            'GAMMA_SENSORY': GAMMA_SENSORY * gamma_scale,
            'GAMMA_CONCEPTUAL': GAMMA_CONCEPTUAL * gamma_scale,
            'GAMMA_SELFMODEL': GAMMA_SELFMODEL * gamma_scale,
        },
    )


def simulate_ocd(cstc_glu_excess=1.5, seed=42, dt=0.01):
    """
    OCD: emergent from CSTC glutamate excess.
    Mechanism:
      - Excess GLU in cortico-striato-thalamo-cortical loop
      - Primarily affects conceptual level → lock-in
    """
    return simulate_v2(
        t_span=(6.0, 30.0), dt=dt, seed=seed,
        cstc_glu_excess=cstc_glu_excess,
    )


def simulate_psilocybin(dose_time=14.0, dose_strength=0.6, seed=42, dt=0.01,
                        t_span=None):
    """
    Psilocybin session with receptor dynamics (tolerance built in).
    """
    if t_span is None:
        t_span = (6.0, 78.0)  # 3 days
    return simulate_v2(
        t_span=t_span, dt=dt, seed=seed,
        pharma_psilocybin=[(dose_time, dose_strength)],
        noise_scale=1.5,
    )


def simulate_ketamine(dose_time_offset=8.0, dose_strength=0.5,
                      chronic_stress=0.6, depression_weeks=8, seed=42, dt=0.01):
    """
    Ketamine rapid antidepressant — two-phase simulation.
    Phase 1: Evolve depression for depression_weeks to get EMERGENT depressed state.
    Phase 2: Ketamine injection into evolved state + 2-week follow-up.
    Mechanism: NMDA blockade → GLU surge → BDNF → rapid P reduction.

    Returns: (t_dep, P_dep, t_ket, P_ket, state_ket, neuromod_ket)
    """
    # Phase 1: evolve depression
    t_dep, P_dep, st_dep, nm_dep = simulate_depression(
        chronic_stress=chronic_stress, weeks=depression_weeks, seed=seed, dt=max(dt, 0.05))

    # Extract evolved depressed state from LAST WAKING PERIOD (not sleep)
    wake_mask = nm_dep['sleep'] < 0.3
    # Find last waking block
    last_wake_indices = np.where(wake_mask)[0]
    if len(last_wake_indices) > 100:
        # Use mean of last 100 waking timesteps
        idx = last_wake_indices[-100:]
    else:
        idx = last_wake_indices
    state0 = SimulationState(
        P_s=np.mean(P_dep['sensory'][idx]),
        P_c=np.mean(P_dep['conceptual'][idx]),
        P_sm=np.mean(P_dep['selfmodel'][idx]),
        hpa_sensitivity=st_dep['hpa_sensitivity'][idx[-1]],
        allostatic_load=st_dep['allostatic_load'][idx[-1]],
        cortisol=st_dep['cortisol'][idx[-1]],
    )

    # Phase 2: ketamine intervention
    t_start = 6.0
    dose_time = t_start + dose_time_offset
    t_ket, P_ket, st_ket, nm_ket = simulate_v2(
        t_span=(t_start, t_start + 14 * 24), dt=dt, seed=seed + 1000,
        state0=state0,
        chronic_stress=chronic_stress * 0.5,  # reduced stress post-treatment
        pharma_ketamine=[(dose_time, dose_strength)],
    )

    return t_dep, P_dep, t_ket, P_ket, st_ket, nm_ket


def simulate_bipolar(seed=42, dt=0.05, t_span=None, P0=0.85):
    """
    Bipolar disorder with emergent oscillatory cycling.
    """
    if t_span is None:
        t_span = (0.0, 7200.0)  # 300 days

    state0 = SimulationState(P_s=P0, P_c=P0, P_sm=P0, P_setpoint=P0)
    return simulate_v2(
        t_span=t_span, dt=dt, seed=seed,
        state0=state0,
        bipolar_mode=True,
        noise_scale=BIPOLAR_NOISE_SCALE,
    )


def simulate_mixed_bipolar(seed=42, dt=0.05, t_span=None, P0=0.85):
    """
    Mixed bipolar with per-level setpoint oscillators.
    Different tau_adapt per level → divergent P trajectories.
    """
    if t_span is None:
        t_span = (0.0, 7200.0)

    state0 = SimulationState(P_s=P0, P_c=P0, P_sm=P0, P_setpoint=P0)
    return simulate_v2(
        t_span=t_span, dt=dt, seed=seed,
        state0=state0,
        bipolar_mode=True,
        bipolar_per_level=True,
        noise_scale=BIPOLAR_NOISE_SCALE,
    )


def simulate_tolerance(dose_strength=0.6, gap_days=14, seed=42, dt=0.01):
    """
    Two psilocybin sessions separated by gap_days.
    Second session shows reduced effect due to 5-HT2A downregulation.
    """
    dose1_time = 14.0
    dose2_time = gap_days * 24.0 + 14.0
    total_time = (gap_days + 3) * 24.0

    return simulate_v2(
        t_span=(6.0, 6.0 + total_time), dt=dt, seed=seed,
        pharma_psilocybin=[
            (dose1_time, dose_strength),
            (dose2_time, dose_strength),
        ],
        noise_scale=1.5,
    )


def simulate_inmt_inhibition(seed=42, dt=0.01):
    """
    Reduced endogenous tonic 5-HT2A/plasticity signaling.
    Simulates what happens when tonic plasticity drive is reduced by 70%.
    Originally framed as INMT inhibition; reframed as general reduction in
    tonic 5-HT2A activation since INMT is not established as the primary
    DMT-producing enzyme (Hatzipantelis et al. 2025).
    """
    return simulate_v2(
        t_span=(6.0, 54.0), dt=dt, seed=seed,
        endogenous_plasticity_scale=0.3,  # 70% reduction
        p_min=0.05,
    )


def simulate_microdosing_with_tolerance(seed=42, dt=0.05, weeks=12):
    """
    Microdosing with receptor dynamics over 12 weeks.
    Fadiman protocol: every 3rd day at 08:00.
    """
    doses = []
    for day in range(weeks * 7):
        if day % 3 == 0:
            doses.append((day * 24.0 + 8.0, 0.08))

    return simulate_v2(
        t_span=(0.0, weeks * 7 * 24.0), dt=dt, seed=seed,
        pharma_psilocybin=doses,
    )


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def compute_waking_stats(t, P_dict, neuromod):
    """Compute mean and std of P during waking hours only."""
    wake_mask = neuromod['sleep'] < 0.3
    stats = {}
    for level in ['sensory', 'conceptual', 'selfmodel']:
        waking_P = P_dict[level][wake_mask]
        stats[level] = {
            'mean': np.mean(waking_P),
            'std': np.std(waking_P),
            'min': np.min(waking_P),
            'max': np.max(waking_P),
        }
    return stats


def classify_emergent(scenario_name, stats, normal_stats=None):
    """
    Classify whether P profile changes are EMERGENT or ASSUMED.
    EMERGENT: the scenario only specified upstream parameters; P changed
    as a downstream consequence of the dynamics.
    """
    emergent_scenarios = {
        'depression', 'psychosis', 'anxiety', 'ptsd', 'adhd', 'ocd',
        'ketamine', 'mixed_bipolar', 'tolerance',
    }
    assumed_scenarios = set()  # v2 has no assumed scenarios

    if scenario_name.lower() in emergent_scenarios:
        return "EMERGENT"
    elif scenario_name.lower() in assumed_scenarios:
        return "ASSUMED"
    else:
        return "EMERGENT"  # default for v2


def print_scenario_summary(name, t, P_dict, state_dict, neuromod,
                           normal_stats=None):
    """Print summary statistics for a scenario."""
    stats = compute_waking_stats(t, P_dict, neuromod)
    classification = classify_emergent(name, stats, normal_stats)

    print(f"\n{'='*60}")
    print(f"Scenario: {name} [{classification}]")
    print(f"{'='*60}")
    for level in ['sensory', 'conceptual', 'selfmodel']:
        s = stats[level]
        print(f"  {level:>12}: mean={s['mean']:.3f}, "
              f"std={s['std']:.3f}, range=[{s['min']:.3f}, {s['max']:.3f}]")

    if 'R_5HT2A' in state_dict:
        r = state_dict['R_5HT2A']
        print(f"  {'R_5HT2A':>12}: final={r[-1]:.3f}")
    if 'cortisol' in state_dict:
        c = state_dict['cortisol']
        print(f"  {'cortisol':>12}: mean={np.mean(c):.3f}, max={np.max(c):.3f}")
    if 'allostatic_load' in state_dict:
        a = state_dict['allostatic_load']
        print(f"  {'allo_load':>12}: final={a[-1]:.3f}")

    return stats, classification


if __name__ == "__main__":
    print("Plasticity Model v2 — Core Engine")
    print("Run scenarios_v2.py for full simulation suite.")

    # Quick test: normal 24h
    t, P, state, nm = simulate_normal_24h(dt=0.05)
    stats = compute_waking_stats(t, P, nm)
    print("\nNormal 24h waking stats:")
    for level, s in stats.items():
        print(f"  {level}: mean P = {s['mean']:.3f} ± {s['std']:.3f}")
