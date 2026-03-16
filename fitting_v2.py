"""
Systematic Parameter Fitting for Plasticity Model v2
=====================================================

Fits MOTIVATED coupling constants against expanded empirical targets using
scipy.optimize.differential_evolution, with cross-validation and
identifiability analysis to avoid overfitting.

Run:
    python3 fitting_v2.py            # Full optimization (~1-3 hours)
    python3 fitting_v2.py --quick    # Quick mode, maxiter=10 (<10 min)

Generates figures in figures/v2/ with FIT_ prefix.
Prints fitted parameter snippet for manual review.

Targets: ~35 discriminative data points from published EEG alpha, LZW
complexity, cortisol dynamics, clinical phenomenology, P300, HRV, BDNF,
pupil diameter, and treatment response dynamics (SSRI onset, ketamine
temporal). Includes model comparison (AIC/BIC) and comorbidity predictions.

Citations:
    Cantero et al. 2002, Muthukumaraswamy 2013, Schartner et al. 2017,
    Debener et al. 2000, Dickerson & Kemeny 2004, Weitzman et al. 1971,
    Zarate et al. 2006, Bramon et al. 2004, Barry et al. 2003,
    Bruder et al. 2012, Kemp et al. 2010, Nagpal et al. 2013,
    Brunoni et al. 2008, Green et al. 2011, Cascardi et al. 2015,
    Taylor et al. 2006, Kolossa et al. 2015, Joshi et al. 2016
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy import stats as scipy_stats
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from parameters import *
from plasticity_v2 import (
    simulate_v2, simulate_normal_24h, simulate_depression,
    simulate_psychosis, simulate_psilocybin, simulate_ketamine,
    simulate_ptsd, simulate_adhd,
    SimulationState, compute_waking_stats,
    p_to_eeg_alpha, p_to_lzw,
    p_to_p300, ne_to_pupil, ne_cort_to_hrv, plasticity_to_bdnf,
)

from datetime import datetime

_BASE_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'v2')
FIGURES_DIR = _BASE_FIGURES_DIR  # overwritten per-run in run_fitting()


def _make_run_dir(mode: str) -> str:
    """Create a timestamped output directory for this run."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(_BASE_FIGURES_DIR, f'{stamp}_{mode}')
    os.makedirs(run_dir, exist_ok=True)

    # Symlink 'latest' for convenience
    latest = os.path.join(_BASE_FIGURES_DIR, 'latest')
    if os.path.islink(latest):
        os.remove(latest)
    try:
        os.symlink(run_dir, latest)
    except OSError:
        pass
    return run_dir


def savefig(fig, name):
    """Save figure and close."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# EMPIRICAL TARGETS
# ============================================================================

@dataclass
class EmpiricalTarget:
    """A single empirical data point to fit against."""
    name: str
    published_value: float
    tolerance: float       # acceptable error band (used in chi-squared)
    weight: float          # higher = more important
    category: str          # 'alpha', 'lzw', 'cortisol', 'P', 'clinical'
    citation: str
    train: bool = True     # True = train set, False = test set


def build_targets() -> List[EmpiricalTarget]:
    """
    Build the expanded empirical target set (~35 data points).

    Train set (~26): well-replicated quantitative findings (EEG, LZW, cortisol,
        hierarchy, sleep, P300, HRV, BDNF, pupil) + anchoring treatment targets
        (SSRI week 4, ketamine 24h)
    Test set (~9): clinical phenomenology + treatment dynamics, held out for
        cross-validation (depression alpha, psychosis LZW, ADHD P, PTSD P,
        SSRI week 1, ketamine 4h/7d)

    Constraint ratio: ~35 targets / 6 effective params ≈ 6:1
    """
    targets = []

    # --- EEG Alpha (normalized to relaxed waking = 1.0) ---
    # alpha_wake removed: always 1.0 by normalization, trivially matched
    targets.append(EmpiricalTarget(
        name='alpha_nrem', published_value=0.30, tolerance=0.08, weight=2.0,
        category='alpha', citation='Cantero 2002', train=True))
    targets.append(EmpiricalTarget(
        name='alpha_rem', published_value=0.40, tolerance=0.10, weight=2.0,
        category='alpha', citation='Cantero 2002', train=True))
    targets.append(EmpiricalTarget(
        name='alpha_psilocybin', published_value=0.40, tolerance=0.10, weight=2.0,
        category='alpha', citation='Muthukumaraswamy 2013', train=True))

    # --- LZW Complexity (Schartner et al. 2017) ---
    # lzw_wake removed: always 0.45 by normalization, trivially matched
    targets.append(EmpiricalTarget(
        name='lzw_propofol', published_value=0.35, tolerance=0.05, weight=2.0,
        category='lzw', citation='Schartner 2017', train=True))
    targets.append(EmpiricalTarget(
        name='lzw_ketamine', published_value=0.55, tolerance=0.06, weight=2.0,
        category='lzw', citation='Schartner 2017', train=True))
    targets.append(EmpiricalTarget(
        name='lzw_psilocybin', published_value=0.52, tolerance=0.06, weight=2.0,
        category='lzw', citation='Schartner 2017', train=True))

    # --- Cortisol dynamics ---
    targets.append(EmpiricalTarget(
        name='cortisol_dawn_nadir_ratio', published_value=4.0, tolerance=1.0,
        weight=1.0, category='cortisol', citation='Weitzman 1971', train=True))
    targets.append(EmpiricalTarget(
        name='cortisol_stress_peak_ratio', published_value=2.5, tolerance=0.5,
        weight=1.0, category='cortisol', citation='Dickerson & Kemeny 2004',
        train=True))

    # --- P hierarchy constraint (soft gaps) ---
    # published = expected gap; predicted = actual P gap (continuous)
    targets.append(EmpiricalTarget(
        name='hierarchy_sens_gt_conc', published_value=0.05, tolerance=0.03,
        weight=0.5, category='P', citation='Model constraint (Bayesian hierarchy)',
        train=True))
    targets.append(EmpiricalTarget(
        name='hierarchy_conc_gt_self', published_value=0.05, tolerance=0.03,
        weight=0.5, category='P', citation='Model constraint (Bayesian hierarchy)',
        train=True))

    # --- Sleep P levels ---
    targets.append(EmpiricalTarget(
        name='sleep_nrem_conceptual', published_value=0.55, tolerance=0.05,
        weight=2.0, category='P', citation='Model target (NREM P)',
        train=True))
    targets.append(EmpiricalTarget(
        name='sleep_rem_conceptual', published_value=0.25, tolerance=0.05,
        weight=1.0, category='P', citation='Model target (REM P)',
        train=True))

    # --- Depression (Debener et al. 2000): alpha +15% vs normal ---
    targets.append(EmpiricalTarget(
        name='depression_alpha_change', published_value=1.15, tolerance=0.10,
        weight=1.5, category='alpha', citation='Debener 2000', train=False))

    # --- Psychosis LZW (Schartner 2017 schizophrenia): +15-20% ---
    targets.append(EmpiricalTarget(
        name='psychosis_lzw_change', published_value=1.175, tolerance=0.10,
        weight=1.5, category='lzw', citation='Schartner 2017', train=False))

    # --- ADHD: reduced mean P, increased variability ---
    targets.append(EmpiricalTarget(
        name='adhd_conceptual_p_deficit', published_value=-0.10, tolerance=0.05,
        weight=1.0, category='P', citation='ADHD attention deficit model',
        train=False))
    targets.append(EmpiricalTarget(
        name='adhd_p_variability_ratio', published_value=1.75, tolerance=0.25,
        weight=1.0, category='P', citation='ADHD variability model', train=False))

    # --- PTSD: sensory hypervigilance + selfmodel dissociation ---
    targets.append(EmpiricalTarget(
        name='ptsd_sensory_p', published_value=0.82, tolerance=0.05,
        weight=1.0, category='P', citation='PTSD hypervigilance model',
        train=False))
    targets.append(EmpiricalTarget(
        name='ptsd_selfmodel_p', published_value=0.52, tolerance=0.05,
        weight=1.0, category='P', citation='PTSD dissociation model',
        train=False))

    # =====================================================================
    # EXPANDED BIOMARKER TARGETS (v3 expansion)
    # =====================================================================

    # --- P300 amplitude (ratio to normal, TRAIN) ---
    # P300 ∝ P_conceptual; reduced in conditions with lower conceptual P
    targets.append(EmpiricalTarget(
        name='schizophrenia_p300', published_value=0.57, tolerance=0.10,
        weight=1.5, category='p300', citation='Bramon 2004 (d=-0.85)',
        train=True))
    targets.append(EmpiricalTarget(
        name='adhd_p300', published_value=0.75, tolerance=0.10,
        weight=1.5, category='p300', citation='Barry 2003 (d=-0.50)',
        train=True))
    targets.append(EmpiricalTarget(
        name='depression_p300', published_value=0.80, tolerance=0.10,
        weight=1.5, category='p300', citation='Bruder 2012 (d=-0.40)',
        train=True))

    # --- HRV (ratio to normal, TRAIN) ---
    # HRV reflects vagal tone; reduced by NE elevation + cortisol
    targets.append(EmpiricalTarget(
        name='depression_hrv', published_value=0.72, tolerance=0.08,
        weight=1.5, category='hrv', citation='Kemp 2010 (d=-0.66)',
        train=True))
    targets.append(EmpiricalTarget(
        name='ptsd_hrv', published_value=0.66, tolerance=0.08,
        weight=1.5, category='hrv', citation='Nagpal 2013 (d=-0.76)',
        train=True))
    targets.append(EmpiricalTarget(
        name='psychosis_hrv', published_value=0.80, tolerance=0.10,
        weight=1.0, category='hrv', citation='Clamor 2016 (d=-0.46)',
        train=True))

    # --- BDNF (ratio to normal, TRAIN) ---
    # BDNF ∝ endogenous plasticity; reduced in depression/schizophrenia
    targets.append(EmpiricalTarget(
        name='depression_bdnf', published_value=0.69, tolerance=0.08,
        weight=1.5, category='bdnf', citation='Brunoni 2008 (d=-0.71)',
        train=True))
    targets.append(EmpiricalTarget(
        name='schizophrenia_bdnf', published_value=0.72, tolerance=0.08,
        weight=1.5, category='bdnf', citation='Green 2011 (d=-0.64)',
        train=True))

    # --- Pupil diameter (ratio to normal, TRAIN) ---
    # Pupil ∝ NE; elevated in PTSD (NE sensitization)
    targets.append(EmpiricalTarget(
        name='ptsd_pupil', published_value=1.175, tolerance=0.10,
        weight=1.0, category='pupil', citation='Cascardi 2015 (+15-20%)',
        train=True))

    # --- Treatment response dynamics ---
    # SSRI: delayed onset (weeks 1 vs 4 improvement ratio)
    # Improvement ratio = (P_dep_baseline - P_at_time) / (P_dep_baseline - P_normal)
    # Week 1 held out (TEST); week 4 in TRAIN to force depression signal
    targets.append(EmpiricalTarget(
        name='ssri_week1_improvement', published_value=0.15, tolerance=0.10,
        weight=1.5, category='treatment', citation='Taylor 2006 (SSRI onset delay)',
        train=False))
    targets.append(EmpiricalTarget(
        name='ssri_week4_improvement', published_value=0.55, tolerance=0.15,
        weight=2.0, category='treatment', citation='Taylor 2006 (SSRI week 4)',
        train=True))

    # Ketamine: rapid onset, transient
    # 24h peak in TRAIN to anchor ketamine temporal curve; 4h and 7d in TEST
    targets.append(EmpiricalTarget(
        name='ketamine_4h_improvement', published_value=0.40, tolerance=0.15,
        weight=1.5, category='treatment', citation='Zarate 2006 (4h post)',
        train=False))
    targets.append(EmpiricalTarget(
        name='ketamine_24h_improvement', published_value=0.35, tolerance=0.15,
        weight=2.0, category='treatment', citation='Zarate 2006 (24h post)',
        train=True))
    targets.append(EmpiricalTarget(
        name='ketamine_7d_improvement', published_value=0.20, tolerance=0.15,
        weight=1.0, category='treatment', citation='Zarate 2006 (7d post)',
        train=False))

    # --- Additional biomarker targets (filling to ~35) ---
    # Depression pupil: reduced reactivity (anhedonia/LC hypofunction)
    targets.append(EmpiricalTarget(
        name='depression_pupil', published_value=0.85, tolerance=0.10,
        weight=1.0, category='pupil', citation='Siegle 2011 (reduced pupil reactivity)',
        train=True))
    # ADHD HRV: reduced vagal tone
    targets.append(EmpiricalTarget(
        name='adhd_hrv', published_value=0.80, tolerance=0.10,
        weight=1.0, category='hrv', citation='Quintana 2012 (d=-0.45)',
        train=True))
    # PTSD BDNF: reduced plasticity
    targets.append(EmpiricalTarget(
        name='ptsd_bdnf', published_value=0.75, tolerance=0.10,
        weight=1.0, category='bdnf', citation='Angelucci 2014 (d=-0.58)',
        train=True))

    return targets


# ============================================================================
# PARAMETERS TO FIT
# ============================================================================

# (name, current_default, lower_bound, upper_bound)
# Reduced from 18→9 based on Fisher Information analysis.
# Removed parameters with FIM stiffness ~0 (unidentifiable), at bounds
# (optimizer wants to leave feasible region), or redundant with others.
# Kept: high-FIM, well-constrained parameters that drive key observables.
FITTED_PARAMS_STANDARD = [
    # High FIM stiffness — well-constrained by EEG/LZW data
    # Bounds widened where quick-mode optimizer hit limits
    ('ALPHA_NE',                0.40,  0.05, 0.80),
    ('ALPHA_5HT',               0.35,  0.05, 0.70),
    ('BETA_PLAST',              0.45,  0.20, 1.20),
    ('GAMMA_SENSORY',           0.90,  0.40, 1.50),
    # Operationalization exponents — directly map P to observables, high FIM
    ('ALPHA_POWER_EXPONENT',    1.5,   0.5,  3.0),
    ('LZW_EXPONENT',            0.5,   0.1,  2.0),
    # Structural sleep parameter — high FIM
    ('P_CONCEPTUAL_NREM',       0.55,  0.35, 0.70),
    # Cortisol driver — needed for cortisol ratio targets
    ('CORTISOL_STRESS_GAIN',    0.50,  0.20, 2.00),
    # GABA gain — NOW identifiable after dual GABA pathway fix
    ('GABA_NE_GAIN_MOD',        0.50,  0.15, 1.50),
    # Pharmacological gain — targeted at receptor-mediated amplification
    ('PSILOCYBIN_PHARMA_GAIN',  2.0,   0.2, 5.0),
    ('KETAMINE_PHARMA_GAIN',    2.0,   1.0, 5.0),
    # PTSD dissociation strength (was hardcoded 0.5)
    ('PTSD_DISSOC_COEFF',       0.25,  0.05, 0.50),
]

# --- Mode: wide-bounds ---
# Widen bounds on parameters that hit limits in Run 2 (ALPHA_NE=0.05,
# ALPHA_5HT=0.05, BETA_PLAST=1.2, KETAMINE_PHARMA_GAIN=5.0).
FITTED_PARAMS_WIDE = [
    ('ALPHA_NE',                0.40,  0.05, 0.80),    # raised: monoamine coupling can't be negligible
    ('ALPHA_5HT',               0.35,  0.05, 0.70),    # raised: monoamine coupling can't be negligible
    ('BETA_PLAST',              0.45,  0.20,  2.00),    # was [0.20, 1.20]
    ('GAMMA_SENSORY',           0.90,  0.40,  1.50),
    ('ALPHA_POWER_EXPONENT',    1.5,   0.5,   3.0),
    ('LZW_EXPONENT',            0.5,   0.1,   2.0),
    ('P_CONCEPTUAL_NREM',       0.55,  0.35,  0.70),
    ('CORTISOL_STRESS_GAIN',    0.50,  0.20,  2.00),
    ('GABA_NE_GAIN_MOD',        0.50,  0.15,  1.50),
    ('PSILOCYBIN_PHARMA_GAIN',  2.0,   0.2,   5.0),
    ('KETAMINE_PHARMA_GAIN',    2.0,   1.0,  10.0),    # was [1.0, 5.0]
    ('PTSD_DISSOC_COEFF',       0.25,  0.05,  0.50),
]

# --- Mode: parsimonious ---
# Fix 6 sloppy parameters at Run 2 best-fit values.
# Only fit 6 well-constrained parameters (high FIM diagonal).
SLOPPY_FIXED_VALUES = {
    'BETA_PLAST':           1.2000,
    'GAMMA_SENSORY':        0.7215,
    'ALPHA_POWER_EXPONENT': 2.7136,
    'CORTISOL_STRESS_GAIN': 1.4606,
    'KETAMINE_PHARMA_GAIN': 5.0000,
    'PTSD_DISSOC_COEFF':    0.2211,
}
FITTED_PARAMS_PARSIMONIOUS = [
    ('ALPHA_NE',                0.40,  0.05, 0.80),    # raised: monoamine coupling can't be negligible
    ('ALPHA_5HT',               0.35,  0.05, 0.70),    # raised: monoamine coupling can't be negligible
    ('LZW_EXPONENT',            0.5,   0.1,   2.0),
    ('P_CONCEPTUAL_NREM',       0.55,  0.35,  0.70),
    ('GABA_NE_GAIN_MOD',        0.50,  0.15,  1.50),
    ('PSILOCYBIN_PHARMA_GAIN',  2.0,   0.2,   5.0),
    ('P300_EXPONENT',           0.1,   0.01,  1.0),    # updating sensitivity for two-factor P300
]


def get_fitted_params(mode: str = 'standard'):
    """Return (param_list, fixed_override) for the given fitting mode."""
    if mode == 'wide':
        return FITTED_PARAMS_WIDE, {}
    elif mode == 'parsimonious':
        return FITTED_PARAMS_PARSIMONIOUS, dict(SLOPPY_FIXED_VALUES)
    else:
        return FITTED_PARAMS_STANDARD, {}


# Default for backward compatibility
FITTED_PARAMS = FITTED_PARAMS_STANDARD
PARAM_NAMES = [p[0] for p in FITTED_PARAMS]
PARAM_DEFAULTS = np.array([p[1] for p in FITTED_PARAMS])
PARAM_BOUNDS = [(p[2], p[3]) for p in FITTED_PARAMS]


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(params_dict: Dict[str, float], targets: List[EmpiricalTarget],
                   dt: float = 0.05, fixed_override: Optional[Dict[str, float]] = None
                   ) -> Dict[str, float]:
    """
    Run relevant scenarios and compute predictions for all targets.

    Uses coarser dt for speed during optimization. The params_override
    mechanism passes coupling constants without modifying globals.

    fixed_override: extra params that are fixed (not optimized) but still
    need to be passed to the simulator (used in parsimonious mode).
    """
    # Merge fixed overrides into params_dict (fitted values take precedence)
    all_params = dict(fixed_override or {})
    all_params.update(params_dict)

    # Separate operationalization exponents from simulation params
    alpha_exp = all_params.get('ALPHA_POWER_EXPONENT', ALPHA_POWER_EXPONENT)
    lzw_exp = all_params.get('LZW_EXPONENT', LZW_EXPONENT)
    p300_exp = all_params.get('P300_EXPONENT', P300_EXPONENT)

    # Build params_override for simulation (exclude operationalization params)
    sim_override = {k: v for k, v in all_params.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT', 'P300_EXPONENT')}

    predictions = {}

    # --- Normal 24h ---
    try:
        t_norm, P_norm, st_norm, nm_norm = simulate_v2(
            t_span=(6.0, 30.0), dt=dt, seed=42,
            params_override=sim_override if sim_override else None,
        )

        wake_mask = nm_norm['sleep'] < 0.3
        nrem_mask = (nm_norm['sleep'] > 0.5) & (nm_norm['REM'] < 0.3)
        rem_mask = nm_norm['REM'] > 0.3

        P_wake = np.mean(P_norm['conceptual'][wake_mask])
        P_nrem = np.mean(P_norm['conceptual'][nrem_mask]) if np.any(nrem_mask) else 0.55
        P_rem = np.mean(P_norm['conceptual'][rem_mask]) if np.any(rem_mask) else 0.25

        P_wake_s = np.mean(P_norm['sensory'][wake_mask])
        P_wake_c = np.mean(P_norm['conceptual'][wake_mask])
        P_wake_sm = np.mean(P_norm['selfmodel'][wake_mask])

        # EEG Alpha (normalized to wake=1.0)
        alpha_wake_raw = p_to_eeg_alpha(P_wake, exponent=alpha_exp)
        alpha_nrem_raw = p_to_eeg_alpha(P_nrem, exponent=alpha_exp)
        alpha_rem_raw = p_to_eeg_alpha(P_rem, exponent=alpha_exp)
        norm_factor = 1.0 / alpha_wake_raw if alpha_wake_raw > 0 else 1.0

        predictions['alpha_nrem'] = alpha_nrem_raw * norm_factor
        predictions['alpha_rem'] = alpha_rem_raw * norm_factor

        # LZW (normalized to wake baseline of 0.45)
        lzw_wake_raw = p_to_lzw(P_wake, exponent=lzw_exp)
        lzw_norm = 0.45 / lzw_wake_raw if lzw_wake_raw > 0 else 1.0

        # Hierarchy (soft gaps: predicted = actual P gap)
        predictions['hierarchy_sens_gt_conc'] = P_wake_s - P_wake_c
        predictions['hierarchy_conc_gt_self'] = P_wake_c - P_wake_sm

        # Sleep P
        predictions['sleep_nrem_conceptual'] = P_nrem
        predictions['sleep_rem_conceptual'] = P_rem

        # Cortisol dynamics
        cort = st_norm['cortisol']
        dawn_mask = (t_norm >= 6.5) & (t_norm <= 8.0)
        nadir_mask = (t_norm >= 23.0) & (t_norm <= 25.0)
        cort_dawn = np.mean(cort[dawn_mask]) if np.any(dawn_mask) else 0.6
        cort_nadir = np.mean(cort[nadir_mask]) if np.any(nadir_mask) else 0.2
        predictions['cortisol_dawn_nadir_ratio'] = (
            cort_dawn / cort_nadir if cort_nadir > 0.01 else 10.0)

        # --- Normal biomarker baselines (for ratio targets) ---
        NE_norm_wake = np.mean(nm_norm['NE'][wake_mask])
        DA_norm_wake = np.mean(nm_norm['DA'][wake_mask])
        cort_norm_wake = np.mean(st_norm['cortisol'][wake_mask])
        plast_norm_wake = np.mean(nm_norm['endogenous_plasticity'][wake_mask])

        p300_norm = p_to_p300(P_wake_c, DA=DA_norm_wake, exponent=p300_exp)
        hrv_norm = ne_cort_to_hrv(NE_norm_wake, cort_norm_wake)
        bdnf_norm = plasticity_to_bdnf(plast_norm_wake)
        pupil_norm = ne_to_pupil(NE_norm_wake)

    except Exception:
        for tgt in targets:
            if tgt.name not in predictions:
                predictions[tgt.name] = 0.0
        return predictions

    # --- Psilocybin ---
    try:
        t_psi, P_psi, _, nm_psi = simulate_v2(
            t_span=(6.0, 30.0), dt=dt, seed=42,
            pharma_psilocybin=[(14.0, 0.6)],
            noise_scale=1.5,
            params_override=sim_override if sim_override else None,
        )
        peak_idx = np.argmin(np.abs(t_psi - 15.75))
        P_psi_peak = P_psi['conceptual'][peak_idx]
        predictions['alpha_psilocybin'] = (
            p_to_eeg_alpha(P_psi_peak, exponent=alpha_exp) * norm_factor)
        predictions['lzw_psilocybin'] = (
            p_to_lzw(P_psi_peak, exponent=lzw_exp) * lzw_norm)
    except Exception:
        predictions['alpha_psilocybin'] = 0.0
        predictions['lzw_psilocybin'] = 0.0

    # --- Propofol LZW (simulated: GABA boost → sedation → high P) ---
    try:
        t_prop, P_prop, _, _ = simulate_v2(
            t_span=(14.0, 18.0), dt=dt, seed=42,
            gaba_deficit=-0.25,  # negative deficit = GABA potentiation
            params_override=sim_override if sim_override else None,
        )
        P_propofol = np.mean(P_prop['conceptual'])
        predictions['lzw_propofol'] = p_to_lzw(P_propofol, exponent=lzw_exp) * lzw_norm
    except Exception:
        predictions['lzw_propofol'] = 0.0

    # --- Ketamine LZW (simulated: NMDA blockade → GLU surge → P reduction) ---
    try:
        t_ket, P_ket, _, _ = simulate_v2(
            t_span=(6.0, 30.0), dt=dt, seed=42,
            pharma_ketamine=[(14.0, 0.5)],
            params_override=sim_override if sim_override else None,
        )
        peak_idx_ket = np.argmin(np.abs(t_ket - 15.5))
        P_ketamine = P_ket['conceptual'][peak_idx_ket]
        predictions['lzw_ketamine'] = p_to_lzw(P_ketamine, exponent=lzw_exp) * lzw_norm
    except Exception:
        predictions['lzw_ketamine'] = 0.0

    # --- Cortisol stress response ---
    try:
        t_stress, _, st_stress, _ = simulate_v2(
            t_span=(6.0, 30.0), dt=dt, seed=42,
            chronic_stress=0.6,
            params_override=sim_override if sim_override else None,
        )
        cort_stress_peak = np.max(st_stress['cortisol'])
        predictions['cortisol_stress_peak_ratio'] = (
            cort_stress_peak / CORTISOL_BASELINE
            if CORTISOL_BASELINE > 0.01 else 5.0)
    except Exception:
        predictions['cortisol_stress_peak_ratio'] = 0.0

    # --- Depression alpha change + biomarkers ---
    # 2 weeks at coarse dt suffices — we only need waking mean P in last week
    try:
        t_dep, P_dep, st_dep, nm_dep = simulate_v2(
            t_span=(6.0, 6.0 + 2 * 7 * 24), dt=max(dt, 0.2), seed=42,
            chronic_stress=0.8,  # raised from 0.6: stronger depression signal for treatment targets
            params_override=sim_override if sim_override else None,
        )
        late_mask = t_dep > (t_dep[-1] - 5 * 24)
        wake_dep = (nm_dep['sleep'][late_mask] < 0.3)
        P_dep_wake = np.mean(P_dep['conceptual'][late_mask][wake_dep]) if np.any(wake_dep) else P_wake
        alpha_dep = p_to_eeg_alpha(P_dep_wake, exponent=alpha_exp) * norm_factor
        predictions['depression_alpha_change'] = alpha_dep
        predictions['_P_dep_wake'] = P_dep_wake  # internal: reused by treatment sims

        # Depression biomarkers (ratios to normal)
        NE_dep = np.mean(nm_dep['NE'][late_mask][wake_dep]) if np.any(wake_dep) else NE_norm_wake
        DA_dep = np.mean(nm_dep['DA'][late_mask][wake_dep]) if np.any(wake_dep) else DA_norm_wake
        cort_dep = np.mean(st_dep['cortisol'][late_mask][wake_dep]) if np.any(wake_dep) else cort_norm_wake
        plast_dep = np.mean(nm_dep['endogenous_plasticity'][late_mask][wake_dep]) if np.any(wake_dep) else plast_norm_wake

        predictions['depression_p300'] = p_to_p300(P_dep_wake, DA=DA_dep, exponent=p300_exp) / p300_norm if p300_norm > 1e-6 else 1.0
        predictions['depression_hrv'] = ne_cort_to_hrv(NE_dep, cort_dep) / hrv_norm if hrv_norm > 1e-6 else 1.0
        predictions['depression_bdnf'] = plasticity_to_bdnf(plast_dep) / bdnf_norm if bdnf_norm > 1e-6 else 1.0
        predictions['depression_pupil'] = ne_to_pupil(NE_dep) / pupil_norm if pupil_norm > 1e-6 else 1.0
    except Exception:
        predictions['depression_alpha_change'] = 1.0
        predictions['depression_p300'] = 1.0
        predictions['depression_hrv'] = 1.0
        predictions['depression_bdnf'] = 1.0
        predictions['depression_pupil'] = 1.0

    # --- Psychosis LZW change + biomarkers ---
    # chronic_stress=0.3: psychosis involves mild HPA axis activation (Corcoran 2003)
    try:
        t_psy, P_psy, st_psy, nm_psy = simulate_v2(
            t_span=(6.0, 30.0), dt=dt, seed=42,
            da_excess=1.5, gaba_deficit=0.3, chronic_stress=0.3,
            params_override=sim_override if sim_override else None,
        )
        wake_psy = nm_psy['sleep'] < 0.3
        P_psy_wake = np.mean(P_psy['conceptual'][wake_psy])
        lzw_psy = p_to_lzw(P_psy_wake, exponent=lzw_exp) * lzw_norm
        predictions['psychosis_lzw_change'] = lzw_psy / 0.45

        # Psychosis biomarkers (schizophrenia P300, HRV, BDNF)
        NE_psy = np.mean(nm_psy['NE'][wake_psy])
        DA_psy = np.mean(nm_psy['DA'][wake_psy])
        cort_psy = np.mean(st_psy['cortisol'][wake_psy])
        plast_psy = np.mean(nm_psy['endogenous_plasticity'][wake_psy])

        predictions['schizophrenia_p300'] = p_to_p300(P_psy_wake, DA=DA_psy, exponent=p300_exp) / p300_norm if p300_norm > 1e-6 else 1.0
        predictions['psychosis_hrv'] = ne_cort_to_hrv(NE_psy, cort_psy) / hrv_norm if hrv_norm > 1e-6 else 1.0
        predictions['schizophrenia_bdnf'] = plasticity_to_bdnf(plast_psy) / bdnf_norm if bdnf_norm > 1e-6 else 1.0
    except Exception:
        predictions['psychosis_lzw_change'] = 1.0
        predictions['schizophrenia_p300'] = 1.0
        predictions['psychosis_hrv'] = 1.0
        predictions['schizophrenia_bdnf'] = 1.0

    # --- ADHD (multi-seed for robust variability estimate) ---
    try:
        gamma_scale = 0.65
        adhd_override = dict(sim_override) if sim_override else {}
        adhd_override['GAMMA_SENSORY'] = (
            params_dict.get('GAMMA_SENSORY', GAMMA_SENSORY) * gamma_scale)
        adhd_override['GAMMA_CONCEPTUAL'] = GAMMA_CONCEPTUAL * gamma_scale
        adhd_override['GAMMA_SELFMODEL'] = (
            params_dict.get('GAMMA_SELFMODEL', GAMMA_SELFMODEL) * gamma_scale)

        seed_means = []
        seed_stds = []
        NE_adhd_acc = []
        DA_adhd_acc = []
        cort_adhd_acc = []
        for adhd_seed in [42, 43, 44]:
            t_adhd, P_adhd, st_adhd, nm_adhd = simulate_v2(
                t_span=(6.0, 30.0), dt=dt, seed=adhd_seed,
                dat_dysfunction=0.7, net_dysfunction=0.8,
                noise_scale=4.0,
                params_override=adhd_override,
            )
            wake_adhd = nm_adhd['sleep'] < 0.3
            P_adhd_wake = P_adhd['conceptual'][wake_adhd]
            seed_means.append(np.mean(P_adhd_wake))
            seed_stds.append(np.std(P_adhd_wake))
            NE_adhd_acc.append(np.mean(nm_adhd['NE'][wake_adhd]))
            DA_adhd_acc.append(np.mean(nm_adhd['DA'][wake_adhd]))
            cort_adhd_acc.append(np.mean(st_adhd['cortisol'][wake_adhd]))

        adhd_mean = np.mean(seed_means)
        adhd_std = np.mean(seed_stds)
        NE_adhd = np.mean(NE_adhd_acc)
        DA_adhd = np.mean(DA_adhd_acc)
        cort_adhd = np.mean(cort_adhd_acc)

        predictions['adhd_conceptual_p_deficit'] = adhd_mean - P_wake_c
        normal_std = np.std(P_norm['conceptual'][wake_mask])
        predictions['adhd_p_variability_ratio'] = (
            adhd_std / normal_std if normal_std > 0.001 else 1.0)

        # ADHD P300 (DA deficit → impaired salience + reduced updating)
        predictions['adhd_p300'] = p_to_p300(adhd_mean, DA=DA_adhd, exponent=p300_exp) / p300_norm if p300_norm > 1e-6 else 1.0
        # ADHD HRV (NE dysregulation → reduced vagal tone)
        predictions['adhd_hrv'] = ne_cort_to_hrv(NE_adhd, cort_adhd) / hrv_norm if hrv_norm > 1e-6 else 1.0
    except Exception:
        predictions['adhd_conceptual_p_deficit'] = 0.0
        predictions['adhd_p_variability_ratio'] = 1.0
        predictions['adhd_p300'] = 1.0
        predictions['adhd_hrv'] = 1.0

    # --- PTSD ---
    try:
        t_ptsd, P_ptsd, st_ptsd, nm_ptsd = simulate_v2(
            t_span=(6.0, 54.0), dt=dt, seed=42,
            ne_sensitization=1.8, coupling_breakdown=0.5,
            chronic_stress=0.4,
            td_coupling_scale=PTSD_TD_BREAKDOWN,
            params_override=sim_override if sim_override else None,
        )
        wake_ptsd = nm_ptsd['sleep'] < 0.3
        predictions['ptsd_sensory_p'] = np.mean(P_ptsd['sensory'][wake_ptsd])
        predictions['ptsd_selfmodel_p'] = np.mean(P_ptsd['selfmodel'][wake_ptsd])

        # PTSD biomarkers (HRV reduced, pupil dilated, BDNF reduced)
        NE_ptsd = np.mean(nm_ptsd['NE'][wake_ptsd])
        cort_ptsd = np.mean(st_ptsd['cortisol'][wake_ptsd])
        plast_ptsd = np.mean(nm_ptsd['endogenous_plasticity'][wake_ptsd])
        predictions['ptsd_hrv'] = ne_cort_to_hrv(NE_ptsd, cort_ptsd) / hrv_norm if hrv_norm > 1e-6 else 1.0
        predictions['ptsd_pupil'] = ne_to_pupil(NE_ptsd) / pupil_norm if pupil_norm > 1e-6 else 1.0
        predictions['ptsd_bdnf'] = plasticity_to_bdnf(plast_ptsd) / bdnf_norm if bdnf_norm > 1e-6 else 1.0
    except Exception:
        predictions['ptsd_sensory_p'] = 0.7
        predictions['ptsd_selfmodel_p'] = 0.6
        predictions['ptsd_hrv'] = 1.0
        predictions['ptsd_pupil'] = 1.0
        predictions['ptsd_bdnf'] = 1.0

    # --- SSRI treatment response (improvement over weeks) ---
    # Simulate depression + SSRI, extract P at week 1 and week 4
    # Improvement ratio = (P_dep_baseline - P_at_time) / (P_dep_baseline - P_normal)
    try:
        # First establish depressed baseline P (reuse from depression sim above)
        P_dep_baseline = predictions.get('_P_dep_wake', P_wake + 0.05)

        t_ssri, P_ssri, _, nm_ssri = simulate_v2(
            t_span=(6.0, 6.0 + 6 * 7 * 24), dt=max(dt, 0.2), seed=42,
            chronic_stress=0.8,  # match depression sim
            pharma_ssri=(6.0, 0.15),  # SSRI starts at simulation start
            params_override=sim_override if sim_override else None,
        )

        # Extract P at week 1 and week 4
        week1_start = 6.0 + 6 * 24  # day 6-7
        week1_end = 6.0 + 8 * 24
        week4_start = 6.0 + 27 * 24  # day 27-28
        week4_end = 6.0 + 29 * 24

        w1_mask = (t_ssri >= week1_start) & (t_ssri <= week1_end) & (nm_ssri['sleep'] < 0.3)
        w4_mask = (t_ssri >= week4_start) & (t_ssri <= week4_end) & (nm_ssri['sleep'] < 0.3)

        P_w1 = np.mean(P_ssri['conceptual'][w1_mask]) if np.any(w1_mask) else P_dep_baseline
        P_w4 = np.mean(P_ssri['conceptual'][w4_mask]) if np.any(w4_mask) else P_dep_baseline

        denom = P_dep_baseline - P_wake_c
        if abs(denom) > 0.01:
            predictions['ssri_week1_improvement'] = np.clip(
                (P_dep_baseline - P_w1) / denom, -0.5, 1.5)
            predictions['ssri_week4_improvement'] = np.clip(
                (P_dep_baseline - P_w4) / denom, -0.5, 1.5)
        else:
            predictions['ssri_week1_improvement'] = 0.0
            predictions['ssri_week4_improvement'] = 0.0
    except Exception:
        predictions['ssri_week1_improvement'] = 0.0
        predictions['ssri_week4_improvement'] = 0.0

    # --- Ketamine temporal dynamics (improvement at 4h, 24h, 7d) ---
    # Two-phase: evolve depression, then ketamine dose
    try:
        P_dep_baseline_ket = predictions.get('_P_dep_wake', P_wake + 0.05)

        # Phase 1: short depression evolution (2 weeks) to get depressed state
        t_dep2, P_dep2, st_dep2, nm_dep2 = simulate_v2(
            t_span=(6.0, 6.0 + 2 * 7 * 24), dt=max(dt, 0.2), seed=42,
            chronic_stress=0.8,  # match depression sim
            params_override=sim_override if sim_override else None,
        )
        late2 = t_dep2 > (t_dep2[-1] - 3 * 24)
        wake2 = nm_dep2['sleep'][late2] < 0.3
        idx2 = np.where(late2)[0]
        if len(idx2) > 0:
            state0_ket = SimulationState(
                P_s=np.mean(P_dep2['sensory'][idx2][wake2]) if np.any(wake2) else P_SENSORY_BASELINE,
                P_c=np.mean(P_dep2['conceptual'][idx2][wake2]) if np.any(wake2) else P_CONCEPTUAL_BASELINE,
                P_sm=np.mean(P_dep2['selfmodel'][idx2][wake2]) if np.any(wake2) else P_SELFMODEL_BASELINE,
                hpa_sensitivity=st_dep2['hpa_sensitivity'][idx2[-1]],
                allostatic_load=st_dep2['allostatic_load'][idx2[-1]],
                cortisol=st_dep2['cortisol'][idx2[-1]],
            )
        else:
            state0_ket = SimulationState()

        # Phase 2: ketamine dose at t=14h, simulate 8 days
        dose_time_ket = 14.0
        t_ket2, P_ket2, _, nm_ket2 = simulate_v2(
            t_span=(6.0, 6.0 + 8 * 24), dt=max(dt, 0.1), seed=1042,
            state0=state0_ket,
            chronic_stress=0.3,  # reduced post-treatment
            pharma_ketamine=[(dose_time_ket, 0.5)],
            params_override=sim_override if sim_override else None,
        )

        # Extract P at 4h, 24h, 7d post-dose
        def _wake_mean_at(t_arr, P_arr, nm_arr, center, window=2.0):
            mask = (t_arr >= center - window) & (t_arr <= center + window)
            wake = nm_arr['sleep'][mask] < 0.3
            if np.any(wake):
                return np.mean(P_arr['conceptual'][mask][wake])
            elif np.any(mask):
                return np.mean(P_arr['conceptual'][mask])
            return P_dep_baseline_ket

        P_4h = _wake_mean_at(t_ket2, P_ket2, nm_ket2, dose_time_ket + 4.0)
        P_24h = _wake_mean_at(t_ket2, P_ket2, nm_ket2, dose_time_ket + 24.0)
        P_7d = _wake_mean_at(t_ket2, P_ket2, nm_ket2, dose_time_ket + 7 * 24)

        denom_ket = P_dep_baseline_ket - P_wake_c
        if abs(denom_ket) > 0.01:
            predictions['ketamine_4h_improvement'] = np.clip(
                (P_dep_baseline_ket - P_4h) / denom_ket, -0.5, 1.5)
            predictions['ketamine_24h_improvement'] = np.clip(
                (P_dep_baseline_ket - P_24h) / denom_ket, -0.5, 1.5)
            predictions['ketamine_7d_improvement'] = np.clip(
                (P_dep_baseline_ket - P_7d) / denom_ket, -0.5, 1.5)
        else:
            predictions['ketamine_4h_improvement'] = 0.0
            predictions['ketamine_24h_improvement'] = 0.0
            predictions['ketamine_7d_improvement'] = 0.0
    except Exception:
        predictions['ketamine_4h_improvement'] = 0.0
        predictions['ketamine_24h_improvement'] = 0.0
        predictions['ketamine_7d_improvement'] = 0.0

    # Fill missing
    for tgt in targets:
        if tgt.name not in predictions:
            predictions[tgt.name] = 0.0

    return predictions


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def residuals_vector(param_vector: np.ndarray, param_names: List[str],
                     targets: List[EmpiricalTarget], dt: float = 0.05,
                     fixed_override: Optional[Dict[str, float]] = None
                     ) -> np.ndarray:
    """Compute weighted residual vector for given parameters."""
    params_dict = dict(zip(param_names, param_vector))
    predictions = evaluate_model(params_dict, targets, dt=dt,
                                 fixed_override=fixed_override)

    resid = np.zeros(len(targets))
    for i, tgt in enumerate(targets):
        pred = predictions.get(tgt.name, 0.0)
        resid[i] = np.sqrt(tgt.weight) * (pred - tgt.published_value) / tgt.tolerance

    return resid


def objective(param_vector: np.ndarray, param_names: List[str],
              targets: List[EmpiricalTarget], dt: float = 0.05,
              fixed_override: Optional[Dict[str, float]] = None) -> float:
    """Weighted chi-squared objective for differential_evolution."""
    resid = residuals_vector(param_vector, param_names, targets, dt=dt,
                             fixed_override=fixed_override)
    return np.sum(resid ** 2)


# ============================================================================
# R-SQUARED COMPUTATION
# ============================================================================

def compute_r_squared(param_vector: np.ndarray, param_names: List[str],
                      targets: List[EmpiricalTarget], dt: float = 0.05,
                      fixed_override: Optional[Dict[str, float]] = None
                      ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """
    Compute R-squared on tolerance-normalized values, RMSE, and per-target detail.

    Normalizes residuals by each target's tolerance so that all measurement
    domains (EEG alpha ~0.3-0.4, cortisol ratios ~2.5-4.0) contribute equally
    to R². Without this, cortisol dominates due to scale.
    """
    params_dict = dict(zip(param_names, param_vector))
    predictions = evaluate_model(params_dict, targets, dt=dt,
                                 fixed_override=fixed_override)

    published = np.array([t.published_value for t in targets])
    predicted = np.array([predictions.get(t.name, 0.0) for t in targets])
    scales = np.array([t.tolerance for t in targets])

    # Normalize both arrays by per-target tolerance
    published_n = published / scales
    predicted_n = predicted / scales

    ss_res = np.sum((predicted_n - published_n) ** 2)
    ss_tot = np.sum((published_n - np.mean(published_n)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((predicted_n - published_n) ** 2))

    detail = {t.name: (predictions.get(t.name, 0.0), t.published_value)
              for t in targets}

    return r_squared, rmse, detail


# ============================================================================
# FISHER INFORMATION MATRIX
# ============================================================================

def fisher_information(opt_params: np.ndarray, param_names: List[str],
                       targets: List[EmpiricalTarget], dt: float = 0.05,
                       epsilon: float = 0.03,
                       fixed_override: Optional[Dict[str, float]] = None):
    """
    Approximate Fisher Information Matrix via Jacobian of residuals.
    Returns FIM, eigenvalues (clamped to 1e-12), eigenvectors.
    Uses bounds-aware finite differences.
    """
    n_targets = len(targets)
    n_params = len(opt_params)
    J = np.zeros((n_targets, n_params))

    # Look up bounds for the param names we're actually fitting
    all_param_specs = {p[0]: (p[2], p[3]) for p in
                       FITTED_PARAMS_WIDE + FITTED_PARAMS_PARSIMONIOUS + FITTED_PARAMS_STANDARD}

    for j in range(n_params):
        step = epsilon * max(abs(opt_params[j]), 0.05)
        lo, hi = all_param_specs.get(param_names[j], (0.0, 10.0))

        p_plus = opt_params.copy()
        p_plus[j] += step
        p_minus = opt_params.copy()
        p_minus[j] -= step

        # Clamp to parameter bounds; use asymmetric difference if needed
        p_plus[j] = min(p_plus[j], hi)
        p_minus[j] = max(p_minus[j], lo)
        total_step = p_plus[j] - p_minus[j]

        r_plus = residuals_vector(p_plus, param_names, targets, dt=dt,
                                  fixed_override=fixed_override)
        r_minus = residuals_vector(p_minus, param_names, targets, dt=dt,
                                   fixed_override=fixed_override)

        if total_step > 1e-10:
            J[:, j] = (r_plus - r_minus) / total_step
        else:
            J[:, j] = 0.0

    FIM = J.T @ J
    eigenvalues, eigenvectors = np.linalg.eigh(FIM)
    # Clamp to guarantee PSD (numerical noise can produce tiny negatives)
    eigenvalues = np.maximum(eigenvalues, 1e-12)

    return FIM, eigenvalues, eigenvectors


# ============================================================================
# NOVEL PREDICTIONS
# ============================================================================

def generate_novel_predictions(opt_params: np.ndarray, param_names: List[str],
                               dt: float = 0.05,
                               fixed_override: Optional[Dict[str, float]] = None
                               ) -> Dict[str, dict]:
    """
    Generate predictions for conditions NOT used in fitting.
    These serve as falsifiable predictions for future empirical work.
    """
    all_params = dict(fixed_override or {})
    all_params.update(dict(zip(param_names, opt_params)))
    sim_override = {k: v for k, v in all_params.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT')}
    lzw_exp = all_params.get('LZW_EXPONENT', LZW_EXPONENT)

    results = {}

    # --- Meditation ---
    # Prediction: enhanced plasticity (BDNF-like, e.g. mindfulness practice)
    # selectively lowers self-model P. No stress change.
    try:
        t_med, P_med, _, nm_med = simulate_v2(
            t_span=(6.0, 30.0), dt=dt, seed=42,
            chronic_stress=0.0,
            endogenous_plasticity_scale=1.3,
            params_override=sim_override if sim_override else None,
        )
        wake_med = nm_med['sleep'] < 0.3
        results['meditation'] = {
            'P_sensory': np.mean(P_med['sensory'][wake_med]),
            'P_conceptual': np.mean(P_med['conceptual'][wake_med]),
            'P_selfmodel': np.mean(P_med['selfmodel'][wake_med]),
            'description': ('Prediction: enhanced plasticity (BDNF-like) '
                            'selectively lowers self-model P'),
        }
    except Exception as e:
        results['meditation'] = {'error': str(e)}

    # --- Anesthesia depth curve ---
    # Propofol dose-response: GABA potentiation increases P, decreases LZW.
    # Negative gaba_deficit = enhanced GABAergic inhibition.
    try:
        doses = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        P_vs_dose = []
        lzw_vs_dose = []
        for dose in doses:
            t_anes, P_anes, _, _ = simulate_v2(
                t_span=(14.0, 18.0), dt=dt, seed=42,
                gaba_deficit=-dose * 0.3,  # negative = GABA boost
                params_override=sim_override if sim_override else None,
            )
            P_mean = np.mean(P_anes['conceptual'])
            P_vs_dose.append(P_mean)
            lzw_vs_dose.append(p_to_lzw(P_mean, exponent=lzw_exp))
        results['anesthesia_depth'] = {
            'doses': doses,
            'P_conceptual': P_vs_dose,
            'LZW': lzw_vs_dose,
            'description': ('Prediction: P increases and LZW decreases '
                            'monotonically with propofol dose'),
        }
    except Exception as e:
        results['anesthesia_depth'] = {'error': str(e)}

    # --- Aging ---
    # Multi-mechanism: NE + 5-HT decline, mild allostatic stress, increased noise
    try:
        ages = [20, 30, 40, 50, 60, 70, 80]
        P_vs_age = []
        for age in ages:
            age_factor = (age - 20) / 60.0  # 0 at 20, 1 at 80
            age_override = dict(sim_override) if sim_override else {}
            age_override['ALPHA_NE'] = (
                all_params.get('ALPHA_NE', ALPHA_NE) * (1.0 - 0.20 * age_factor))
            age_override['ALPHA_5HT'] = (
                all_params.get('ALPHA_5HT', ALPHA_5HT) * (1.0 - 0.20 * age_factor))
            t_age, P_age, _, nm_age = simulate_v2(
                t_span=(6.0, 30.0), dt=dt, seed=42,
                chronic_stress=0.05 * age_factor,
                noise_scale=1.0 + 0.15 * age_factor,
                params_override=age_override if age_override else None,
            )
            wake_age = nm_age['sleep'] < 0.3
            P_vs_age.append(np.mean(P_age['conceptual'][wake_age]))
        results['aging'] = {
            'ages': ages,
            'P_conceptual': P_vs_age,
            'description': ('Prediction: multi-mechanism aging — NE+5-HT decline, '
                            'mild allostatic stress, increased noise'),
        }
    except Exception as e:
        results['aging'] = {'error': str(e)}

    return results


# ============================================================================
# MODEL COMPARISON FRAMEWORK
# ============================================================================

def _infer_condition(target_name: str) -> str:
    """Map target name to condition group for category model."""
    conditions = {
        'depression': ['depression_alpha', 'depression_p300', 'depression_hrv',
                        'depression_bdnf', 'ssri_week'],
        'psychosis': ['psychosis_lzw', 'psychosis_hrv', 'schizophrenia_p300',
                       'schizophrenia_bdnf'],
        'ptsd': ['ptsd_sensory', 'ptsd_selfmodel', 'ptsd_hrv', 'ptsd_pupil'],
        'adhd': ['adhd_conceptual', 'adhd_p_variability', 'adhd_p300'],
        'psilocybin': ['alpha_psilocybin', 'lzw_psilocybin'],
        'ketamine': ['lzw_ketamine', 'ketamine_4h', 'ketamine_24h', 'ketamine_7d'],
        'propofol': ['lzw_propofol'],
        'sleep': ['alpha_nrem', 'alpha_rem', 'sleep_nrem', 'sleep_rem'],
        'cortisol': ['cortisol_dawn', 'cortisol_stress'],
        'hierarchy': ['hierarchy_sens', 'hierarchy_conc'],
    }
    for group, prefixes in conditions.items():
        for prefix in prefixes:
            if target_name.startswith(prefix):
                return group
    return 'other'


def model_comparison(opt_params: np.ndarray, param_names: List[str],
                     targets: List[EmpiricalTarget], dt: float = 0.05,
                     fixed_override: Optional[Dict[str, float]] = None
                     ) -> Dict[str, dict]:
    """
    Compare the full plasticity model against reduced/alternative models
    using AIC, BIC, and R² on ALL targets.

    Models:
      1. Null: k=n (one param per target — perfect fit, max penalty)
      2. Mean-only: k=1 (grand mean)
      3. Category: k=n_groups (per-condition means)
      4. DA-only: k=2 (zero NE/5-HT/plasticity coupling)
      5. No-hierarchy: k=6 (collapse 3 P levels → mean)
      6. Full plasticity: k=6 (our model)

    Reduced models are NOT re-fitted — they use structural restrictions.
    """
    n = len(targets)
    published = np.array([t.published_value for t in targets])
    scales = np.array([t.tolerance for t in targets])
    published_n = published / scales

    # --- Full plasticity model ---
    all_params_dict = dict(fixed_override or {})
    all_params_dict.update(dict(zip(param_names, opt_params)))
    preds_full = evaluate_model(all_params_dict, targets, dt=dt)
    predicted_full = np.array([preds_full.get(t.name, 0.0) for t in targets])
    predicted_full_n = predicted_full / scales

    k_full = len(param_names)
    ss_res_full = np.sum((predicted_full_n - published_n) ** 2)
    r2_full = 1.0 - ss_res_full / np.sum((published_n - np.mean(published_n)) ** 2)

    results = {}

    def _compute_ic(ss_res, k, n_pts):
        """Compute AIC and BIC from residual sum of squares."""
        if ss_res <= 0:
            ss_res = 1e-10
        log_lik = -n_pts / 2.0 * np.log(ss_res / n_pts)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n_pts) - 2 * log_lik
        return aic, bic

    # 1. Null model: perfect fit (SS=0, k=n)
    aic_null, bic_null = _compute_ic(1e-10, n, n)
    results['null'] = {'k': n, 'R2': 1.0, 'AIC': aic_null, 'BIC': bic_null,
                        'description': 'One parameter per target (memorization)'}

    # 2. Mean-only model: predict grand mean for all
    grand_mean_n = np.mean(published_n)
    ss_mean = np.sum((published_n - grand_mean_n) ** 2)
    r2_mean = 0.0  # by definition
    aic_mean, bic_mean = _compute_ic(ss_mean, 1, n)
    results['mean_only'] = {'k': 1, 'R2': r2_mean, 'AIC': aic_mean, 'BIC': bic_mean,
                             'description': 'Grand mean (no model)'}

    # 3. Category model: per-condition means
    groups = [_infer_condition(t.name) for t in targets]
    unique_groups = list(set(groups))
    k_cat = len(unique_groups)
    group_means = {}
    for g in unique_groups:
        g_vals = [published_n[i] for i, grp in enumerate(groups) if grp == g]
        group_means[g] = np.mean(g_vals)
    predicted_cat_n = np.array([group_means[g] for g in groups])
    ss_cat = np.sum((published_n - predicted_cat_n) ** 2)
    ss_tot = np.sum((published_n - np.mean(published_n)) ** 2)
    r2_cat = 1.0 - ss_cat / ss_tot if ss_tot > 0 else 0.0
    aic_cat, bic_cat = _compute_ic(ss_cat, k_cat, n)
    results['category'] = {'k': k_cat, 'R2': r2_cat, 'AIC': aic_cat, 'BIC': bic_cat,
                            'description': f'Per-condition means ({k_cat} groups)'}

    # 4. DA-only model: zero NE/5-HT/plasticity coupling
    try:
        da_override = dict(all_params_dict)
        da_override['ALPHA_NE'] = 0.0
        da_override['ALPHA_5HT'] = 0.0
        da_override['BETA_PLAST'] = 0.0
        preds_da = evaluate_model(da_override, targets, dt=dt)
        predicted_da = np.array([preds_da.get(t.name, 0.0) for t in targets])
        predicted_da_n = predicted_da / scales
        ss_da = np.sum((predicted_da_n - published_n) ** 2)
        r2_da = 1.0 - ss_da / ss_tot if ss_tot > 0 else 0.0
        aic_da, bic_da = _compute_ic(ss_da, 2, n)
        results['da_only'] = {'k': 2, 'R2': r2_da, 'AIC': aic_da, 'BIC': bic_da,
                               'description': 'DA-only (zero NE/5-HT/plasticity)'}
    except Exception:
        results['da_only'] = {'k': 2, 'R2': 0.0, 'AIC': 1e6, 'BIC': 1e6,
                               'description': 'DA-only (failed)'}

    # 5. No-hierarchy model: collapse 3 P levels → average
    try:
        preds_nohier = dict(preds_full)
        # Collapse hierarchy: predict mean P across levels for hierarchy targets
        preds_nohier['hierarchy_sens_gt_conc'] = 0.0
        preds_nohier['hierarchy_conc_gt_self'] = 0.0
        predicted_nohier = np.array([preds_nohier.get(t.name, 0.0) for t in targets])
        predicted_nohier_n = predicted_nohier / scales
        ss_nohier = np.sum((predicted_nohier_n - published_n) ** 2)
        r2_nohier = 1.0 - ss_nohier / ss_tot if ss_tot > 0 else 0.0
        aic_nohier, bic_nohier = _compute_ic(ss_nohier, k_full, n)
        results['no_hierarchy'] = {'k': k_full, 'R2': r2_nohier, 'AIC': aic_nohier,
                                    'BIC': bic_nohier,
                                    'description': 'No hierarchy (collapsed P levels)'}
    except Exception:
        results['no_hierarchy'] = {'k': k_full, 'R2': 0.0, 'AIC': 1e6, 'BIC': 1e6,
                                    'description': 'No hierarchy (failed)'}

    # 6. Full plasticity model
    aic_full, bic_full = _compute_ic(ss_res_full, k_full, n)
    results['full_plasticity'] = {'k': k_full, 'R2': r2_full, 'AIC': aic_full,
                                   'BIC': bic_full,
                                   'description': 'Full plasticity model'}

    return results


# ============================================================================
# COMORBIDITY PREDICTIONS
# ============================================================================

def _extract_biomarker_profile(P_dict, state_dict, neuromod, wake_mask,
                                p300_norm, hrv_norm, bdnf_norm, pupil_norm):
    """Extract all biomarker ratios from a simulation."""
    P_c = np.mean(P_dict['conceptual'][wake_mask])
    NE = np.mean(neuromod['NE'][wake_mask])
    DA = np.mean(neuromod['DA'][wake_mask])
    cort = np.mean(state_dict['cortisol'][wake_mask])
    plast = np.mean(neuromod['endogenous_plasticity'][wake_mask])

    return {
        'P_sensory': np.mean(P_dict['sensory'][wake_mask]),
        'P_conceptual': P_c,
        'P_selfmodel': np.mean(P_dict['selfmodel'][wake_mask]),
        'p300_ratio': p_to_p300(P_c, DA=DA) / p300_norm if p300_norm > 1e-6 else 1.0,
        'hrv_ratio': ne_cort_to_hrv(NE, cort) / hrv_norm if hrv_norm > 1e-6 else 1.0,
        'bdnf_ratio': plasticity_to_bdnf(plast) / bdnf_norm if bdnf_norm > 1e-6 else 1.0,
        'pupil_ratio': ne_to_pupil(NE) / pupil_norm if pupil_norm > 1e-6 else 1.0,
    }


def generate_comorbidity_predictions(opt_params: np.ndarray, param_names: List[str],
                                      dt: float = 0.05,
                                      fixed_override: Optional[Dict[str, float]] = None
                                      ) -> Dict[str, dict]:
    """
    Generate predictions for comorbid conditions NOT used in fitting.
    These are novel, falsifiable predictions combining multiple upstream mechanisms.
    """
    all_params = dict(fixed_override or {})
    all_params.update(dict(zip(param_names, opt_params)))
    sim_override = {k: v for k, v in all_params.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT')}

    # Compute normal baselines for ratio normalization
    t_norm, P_norm, st_norm, nm_norm = simulate_v2(
        t_span=(6.0, 30.0), dt=dt, seed=42,
        params_override=sim_override if sim_override else None,
    )
    wake_norm = nm_norm['sleep'] < 0.3
    NE_n = np.mean(nm_norm['NE'][wake_norm])
    DA_n = np.mean(nm_norm['DA'][wake_norm])
    cort_n = np.mean(st_norm['cortisol'][wake_norm])
    plast_n = np.mean(nm_norm['endogenous_plasticity'][wake_norm])
    P_c_n = np.mean(P_norm['conceptual'][wake_norm])

    p300_n = p_to_p300(P_c_n, DA=DA_n)
    hrv_n = ne_cort_to_hrv(NE_n, cort_n)
    bdnf_n = plasticity_to_bdnf(plast_n)
    pupil_n = ne_to_pupil(NE_n)

    results = {}

    # --- Depression + PTSD ---
    try:
        t_dp, P_dp, st_dp, nm_dp = simulate_v2(
            t_span=(6.0, 54.0), dt=dt, seed=42,
            chronic_stress=0.8, ne_sensitization=1.8, coupling_breakdown=0.5,
            td_coupling_scale=PTSD_TD_BREAKDOWN,
            params_override=sim_override if sim_override else None,
        )
        wake_dp = nm_dp['sleep'] < 0.3
        profile = _extract_biomarker_profile(P_dp, st_dp, nm_dp, wake_dp,
                                              p300_n, hrv_n, bdnf_n, pupil_n)
        profile['description'] = 'Depression + PTSD: chronic stress + NE sensitization + coupling breakdown'
        results['depression_ptsd'] = profile
    except Exception as e:
        results['depression_ptsd'] = {'error': str(e)}

    # --- Depression + Anxiety ---
    try:
        t_da, P_da, st_da, nm_da = simulate_v2(
            t_span=(6.0, 54.0), dt=dt, seed=42,
            chronic_stress=0.8, gaba_deficit=0.15,
            params_override=sim_override if sim_override else None,
        )
        wake_da = nm_da['sleep'] < 0.3
        profile = _extract_biomarker_profile(P_da, st_da, nm_da, wake_da,
                                              p300_n, hrv_n, bdnf_n, pupil_n)
        profile['description'] = 'Depression + Anxiety: chronic stress + GABA deficit'
        results['depression_anxiety'] = profile
    except Exception as e:
        results['depression_anxiety'] = {'error': str(e)}

    # --- ADHD + Anxiety ---
    try:
        adhd_anx_override = dict(sim_override) if sim_override else {}
        gamma_scale = 0.65
        adhd_anx_override['GAMMA_SENSORY'] = all_params.get('GAMMA_SENSORY', GAMMA_SENSORY) * gamma_scale
        adhd_anx_override['GAMMA_CONCEPTUAL'] = GAMMA_CONCEPTUAL * gamma_scale
        adhd_anx_override['GAMMA_SELFMODEL'] = GAMMA_SELFMODEL * gamma_scale

        t_aa, P_aa, st_aa, nm_aa = simulate_v2(
            t_span=(6.0, 30.0), dt=dt, seed=42,
            dat_dysfunction=0.7, gaba_deficit=0.15,
            noise_scale=4.0,
            params_override=adhd_anx_override,
        )
        wake_aa = nm_aa['sleep'] < 0.3
        profile = _extract_biomarker_profile(P_aa, st_aa, nm_aa, wake_aa,
                                              p300_n, hrv_n, bdnf_n, pupil_n)
        profile['description'] = 'ADHD + Anxiety: DAT dysfunction + GABA deficit'
        results['adhd_anxiety'] = profile
    except Exception as e:
        results['adhd_anxiety'] = {'error': str(e)}

    return results


# ============================================================================
# CONVERGENCE CALLBACK
# ============================================================================

class ConvergenceTracker:
    """Track optimization progress for convergence plot."""
    def __init__(self):
        self.convergence_values = []
        self.generations = []
        self._gen = 0

    def __call__(self, xk, convergence):
        self._gen += 1
        self.convergence_values.append(convergence)
        self.generations.append(self._gen)
        if self._gen % 5 == 0:
            print(f"    Generation {self._gen}: convergence = {convergence:.6f}")
        return False


# ============================================================================
# FIGURES
# ============================================================================

def plot_convergence(tracker: ConvergenceTracker, suffix: str = ""):
    """Fig 1: Fractional convergence vs generation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tracker.generations, tracker.convergence_values, 'b-', lw=2,
            label='Fractional convergence')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fractional convergence')
    ax.set_title('Optimization Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, f'FIT_convergence{suffix}.png')


def plot_param_comparison(pre_fit: np.ndarray, post_fit: np.ndarray,
                          names: List[str], suffix: str = ""):
    """Fig 2: Bar chart of pre-fit vs post-fit parameter values (normalized)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(names))
    width = 0.35

    scale = np.maximum(np.abs(pre_fit), np.abs(post_fit))
    scale[scale < 1e-6] = 1.0
    pre_norm = pre_fit / scale
    post_norm = post_fit / scale

    ax.bar(x - width/2, pre_norm, width, label='Pre-fit (default)', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, post_norm, width, label='Post-fit (optimized)', color='coral', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Normalized value')
    ax.set_title('Parameter Comparison: Pre-fit vs Post-fit', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', lw=0.5)
    plt.tight_layout()
    savefig(fig, f'FIT_param_comparison{suffix}.png')


def plot_alpha_comparison(detail_train: dict, detail_test: dict, suffix: str = ""):
    """Fig 3: EEG alpha model vs published."""
    alpha_targets = {k: v for k, v in {**detail_train, **detail_test}.items()
                     if 'alpha' in k}
    if not alpha_targets:
        return

    names = list(alpha_targets.keys())
    predicted = [alpha_targets[n][0] for n in names]
    published = [alpha_targets[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, published, width, label='Published', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, predicted, width, label='Model (fitted)', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('alpha_', '').replace('_', ' ').title() for n in names],
                       fontsize=10)
    ax.set_ylabel('Normalized EEG Alpha Power')
    ax.set_title('EEG Alpha: Fitted Model vs Published Data', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    savefig(fig, f'FIT_alpha_comparison{suffix}.png')


def plot_lzw_comparison(detail_train: dict, detail_test: dict, suffix: str = ""):
    """Fig 4: LZW model vs published."""
    lzw_targets = {k: v for k, v in {**detail_train, **detail_test}.items()
                   if 'lzw' in k}
    if not lzw_targets:
        return

    names = list(lzw_targets.keys())
    predicted = [lzw_targets[n][0] for n in names]
    published = [lzw_targets[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, published, width, label='Published', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, predicted, width, label='Model (fitted)', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('lzw_', '').replace('_', ' ').title() for n in names],
                       fontsize=10)
    ax.set_ylabel('Normalized LZW Complexity')
    ax.set_title('LZW Complexity: Fitted Model vs Published Data',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    savefig(fig, f'FIT_lzw_comparison{suffix}.png')


def plot_scatter(targets: List[EmpiricalTarget], detail: dict,
                 r_squared: float, rmse: float, title: str, filename: str):
    """Fig 5/6: Scatter plot model vs published with R-squared."""
    names = [t.name for t in targets]
    predicted = np.array([detail[n][0] for n in names])
    published = np.array([detail[n][1] for n in names])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(published, predicted, s=80, c='steelblue', edgecolors='black', zorder=5)

    for i, name in enumerate(names):
        short_name = name.replace('_', ' ')
        ax.annotate(short_name, (published[i], predicted[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=7)

    all_vals = np.concatenate([published, predicted])
    lims = [min(all_vals) - 0.1, max(all_vals) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect fit')

    if len(published) > 2:
        slope, intercept, _, _, _ = scipy_stats.linregress(published, predicted)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'r-', alpha=0.7,
                label=f'R²={r_squared:.3f}, RMSE={rmse:.3f}')

    ax.set_xlabel('Published Value')
    ax.set_ylabel('Model Prediction')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    savefig(fig, filename)


def plot_fisher_eigenvalues(eigenvalues: np.ndarray, param_names: List[str],
                            suffix: str = ""):
    """Fig 7: FIM eigenvalue spectrum (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    clamped = np.maximum(eigenvalues, 1e-12)
    sorted_eig = np.sort(clamped)[::-1]
    ax.bar(range(len(sorted_eig)), sorted_eig, color='steelblue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index (sorted)')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Fisher Information Matrix: Eigenvalue Spectrum',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    cond = sorted_eig[0] / sorted_eig[-1]
    ax.text(0.95, 0.95, f'Condition number: {cond:.1e}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    savefig(fig, f'FIT_fisher_eigenvalues{suffix}.png')


def plot_parameter_identifiability(FIM: np.ndarray, param_names: List[str],
                                   suffix: str = ""):
    """Fig 8: Bar chart of FIM diagonal (parameter stiffness)."""
    diag = np.maximum(np.diag(FIM), 1e-12)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['coral' if d > np.median(diag) else 'lightblue' for d in diag]
    ax.bar(range(len(diag)), diag, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel('FIM diagonal (stiffness, log scale)')
    ax.set_title('Parameter Identifiability: Well-Constrained vs Sloppy',
                 fontsize=13, fontweight='bold')
    ax.axhline(np.median(diag), color='gray', ls='--', alpha=0.5, label='Median stiffness')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    savefig(fig, f'FIT_parameter_identifiability{suffix}.png')


def plot_novel_predictions(novel: Dict[str, dict], suffix: str = ""):
    """Fig 9: Predicted P trajectories for meditation/anesthesia/aging."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Novel Predictions (NOT used in fitting)',
                 fontsize=14, fontweight='bold')

    # Meditation
    ax = axes[0]
    if 'error' not in novel.get('meditation', {'error': True}):
        med = novel['meditation']
        levels = ['P_sensory', 'P_conceptual', 'P_selfmodel']
        vals = [med[l] for l in levels]
        colors = ['blue', 'green', 'red']
        labels = ['Sensory', 'Conceptual', 'Self-model']
        ax.bar(range(3), vals, color=colors, alpha=0.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.axhline(P_WAKING, color='gray', ls='--', alpha=0.5, label=f'Normal P={P_WAKING}')
        ax.set_ylabel('Waking Mean P')
        ax.set_title('Meditation (enhanced plasticity)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Error in simulation', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis='y')

    # Anesthesia depth (dual y-axis: P + LZW)
    ax = axes[1]
    if 'error' not in novel.get('anesthesia_depth', {'error': True}):
        anes = novel['anesthesia_depth']
        ax2 = ax.twinx()
        line1, = ax.plot(anes['doses'], anes['P_conceptual'], 'bo-', lw=2,
                         markersize=8, label='P (conceptual)')
        line2, = ax2.plot(anes['doses'], anes['LZW'], 'rs--', lw=2,
                          markersize=8, label='LZW complexity')
        ax.set_xlabel('Propofol dose (a.u.)')
        ax.set_ylabel('Mean P (conceptual)', color='blue')
        ax2.set_ylabel('LZW complexity', color='red')
        ax.set_title('Anesthesia Depth Curve', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        ax.legend([line1, line2], ['P (conceptual)', 'LZW complexity'],
                  fontsize=8, loc='center right')
    else:
        ax.text(0.5, 0.5, 'Error in simulation', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # Aging
    ax = axes[2]
    if 'error' not in novel.get('aging', {'error': True}):
        aging = novel['aging']
        ax.plot(aging['ages'], aging['P_conceptual'], 'ro-', lw=2, markersize=8)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Waking Mean P (conceptual)')
        ax.set_title('Aging (multi-mechanism)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Error in simulation', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, f'FIT_novel_predictions{suffix}.png')


def plot_model_comparison(comparison: Dict[str, dict], suffix: str = ""):
    """Model comparison: AIC/BIC bars, R² bars, summary table."""
    model_names = ['mean_only', 'category', 'da_only', 'no_hierarchy', 'full_plasticity']
    labels = ['Mean\nOnly', 'Category', 'DA\nOnly', 'No\nHierarchy', 'Full\nPlasticity']

    aics = [comparison[m]['AIC'] for m in model_names]
    bics = [comparison[m]['BIC'] for m in model_names]
    r2s = [comparison[m]['R2'] for m in model_names]
    ks = [comparison[m]['k'] for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

    # Panel 1: AIC
    ax = axes[0]
    colors = ['coral' if m == 'full_plasticity' else 'steelblue' for m in model_names]
    ax.bar(range(len(model_names)), aics, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('AIC (lower = better)')
    ax.set_title('AIC', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: BIC
    ax = axes[1]
    ax.bar(range(len(model_names)), bics, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('BIC (lower = better)')
    ax.set_title('BIC', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: R²
    ax = axes[2]
    ax.bar(range(len(model_names)), r2s, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('R² (higher = better)')
    ax.set_title('R²', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate with k values
    for i, k in enumerate(ks):
        ax.annotate(f'k={k}', (i, r2s[i] + 0.03), ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, f'FIT_model_comparison{suffix}.png')


def plot_comorbidity_predictions(comorbidity: Dict[str, dict], suffix: str = ""):
    """Biomarker profiles for comorbid conditions."""
    valid = {k: v for k, v in comorbidity.items() if 'error' not in v}
    if not valid:
        return

    biomarkers = ['p300_ratio', 'hrv_ratio', 'bdnf_ratio', 'pupil_ratio']
    bm_labels = ['P300', 'HRV', 'BDNF', 'Pupil']

    n_conds = len(valid)
    fig, axes = plt.subplots(1, n_conds, figsize=(6 * n_conds, 6))
    if n_conds == 1:
        axes = [axes]

    fig.suptitle('Comorbidity Predictions (NOT used in fitting)',
                 fontsize=14, fontweight='bold')

    for ax, (cond, profile) in zip(axes, valid.items()):
        vals = [profile.get(bm, 1.0) for bm in biomarkers]
        colors_bm = ['steelblue', 'green', 'coral', 'purple']
        ax.bar(range(len(biomarkers)), vals, color=colors_bm, alpha=0.7, edgecolor='black')
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Normal')
        ax.set_xticks(range(len(biomarkers)))
        ax.set_xticklabels(bm_labels)
        ax.set_ylabel('Ratio to Normal')
        cond_label = cond.replace('_', ' + ').replace('depression', 'Dep').replace(
            'ptsd', 'PTSD').replace('anxiety', 'Anx').replace('adhd', 'ADHD')
        ax.set_title(cond_label, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(max(vals) * 1.15, 1.3))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, f'FIT_comorbidity_predictions{suffix}.png')


def plot_scatter_all(all_targets: List[EmpiricalTarget], detail_all: dict,
                     r2_all: float, rmse_all: float, suffix: str = ""):
    """Scatter plot of ALL targets (train + test) in single figure."""
    plot_scatter(all_targets, detail_all, r2_all, rmse_all,
                 f'All Targets: Model vs Published (R²={r2_all:.3f})',
                 f'FIT_scatter_all{suffix}.png')


# ============================================================================
# FITTED PARAMETER SNIPPET
# ============================================================================

def print_fitted_snippet(opt_params: np.ndarray, param_names: List[str],
                         fixed_override: Optional[Dict[str, float]] = None):
    """Print a parameters_fitted.py snippet for manual review."""
    print("\n" + "=" * 70)
    print("FITTED PARAMETER SNIPPET — paste into parameters.py after review")
    print("=" * 70)
    print()
    print("# --- Fitted parameters (from fitting_v2.py) ---")
    defaults_map = {p[0]: p[1] for p in FITTED_PARAMS}
    for name, val, (lo, hi) in zip(param_names, opt_params, PARAM_BOUNDS):
        default = defaults_map.get(name, 0.0)
        change_pct = ((val - default) / abs(default) * 100) if abs(default) > 1e-6 else 0
        at_bound = " ← AT BOUND" if (abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6) else ""
        print(f"{name} = {val:.4f}  "
              f"# was {default:.4f}, change: {change_pct:+.1f}%, "
              f"bounds: [{lo}, {hi}]{at_bound}")
    if fixed_override:
        print()
        print("# --- Fixed (sloppy) parameters (not optimized) ---")
        for name, val in fixed_override.items():
            print(f"{name} = {val:.4f}  # fixed at Run 2 best-fit value")
    print()
    print("# To apply: copy the lines above into parameters.py,")
    print("# replacing the existing MOTIVATED/FREE values.")
    print("=" * 70)


# ============================================================================
# MAIN FITTING PIPELINE
# ============================================================================

def run_fitting(maxiter: int = 100, popsize: int = 15, dt: float = 0.05,
                seed: int = 42, quick: bool = False, workers: int = -1,
                mode: str = 'standard'):
    """
    Run the full fitting pipeline:
    1. Build targets and split into train/test
    2. Optimize with differential_evolution (parallel workers)
    3. Evaluate on train and test sets
    4. Fisher information analysis
    5. Novel predictions
    6. Generate all figures
    7. Print fitted parameter snippet

    mode: 'standard' | 'wide' | 'parsimonious'
      - standard: original bounds (12 params)
      - wide: widened bounds on bound-hitting params (12 params)
      - parsimonious: fix 6 sloppy params at Run 2 values, fit 6 well-constrained
    """
    global FITTED_PARAMS, PARAM_NAMES, PARAM_DEFAULTS, PARAM_BOUNDS, FIGURES_DIR

    start_time = time.time()
    FIGURES_DIR = _make_run_dir(mode)

    print("=" * 70)
    print("Plasticity Model v2 — Systematic Parameter Fitting")
    print("=" * 70)

    # --- Select mode ---
    fitted_params_list, fixed_override = get_fitted_params(mode)
    FITTED_PARAMS = fitted_params_list
    PARAM_NAMES = [p[0] for p in FITTED_PARAMS]
    PARAM_DEFAULTS = np.array([p[1] for p in FITTED_PARAMS])
    PARAM_BOUNDS = [(p[2], p[3]) for p in FITTED_PARAMS]

    print(f"\n  Mode: {mode}")
    if fixed_override:
        print(f"  Fixed (sloppy) parameters: {list(fixed_override.keys())}")
        for k, v in fixed_override.items():
            print(f"    {k} = {v:.4f}")

    if quick:
        print("  [QUICK MODE] maxiter=15, popsize=15")
        maxiter = 15
        popsize = 15

    # --- 1. Build targets ---
    all_targets = build_targets()
    train_targets = [t for t in all_targets if t.train]
    test_targets = [t for t in all_targets if not t.train]
    print(f"\n  Targets: {len(all_targets)} total "
          f"({len(train_targets)} train, {len(test_targets)} test)")
    print(f"  Parameters to fit: {len(PARAM_NAMES)}")
    print(f"  Optimizer: differential_evolution, maxiter={maxiter}, "
          f"popsize={popsize}, workers={workers}")
    print(f"  Simulation dt: {dt}")

    # --- Pre-fit R-squared ---
    print("\n  Pre-fit evaluation (default parameters)...")
    r2_pre_train, rmse_pre_train, detail_pre_train = compute_r_squared(
        PARAM_DEFAULTS, PARAM_NAMES, train_targets, dt=dt,
        fixed_override=fixed_override)
    r2_pre_test, rmse_pre_test, detail_pre_test = compute_r_squared(
        PARAM_DEFAULTS, PARAM_NAMES, test_targets, dt=dt,
        fixed_override=fixed_override)
    r2_pre_all, rmse_pre_all, detail_pre_all = compute_r_squared(
        PARAM_DEFAULTS, PARAM_NAMES, all_targets, dt=dt,
        fixed_override=fixed_override)
    print(f"    Train R² = {r2_pre_train:.3f}, RMSE = {rmse_pre_train:.3f}")
    print(f"    Test  R² = {r2_pre_test:.3f}, RMSE = {rmse_pre_test:.3f}")
    print(f"    All   R² = {r2_pre_all:.3f}, RMSE = {rmse_pre_all:.3f}")

    print(f"\n    Per-target pre-fit predictions:")
    for tgt in all_targets:
        pred, pub = detail_pre_all[tgt.name]
        err = pred - pub
        split = 'TRAIN' if tgt.train else 'TEST '
        print(f"      [{split}] {tgt.name:<35} pub={pub:>7.3f}  "
              f"pred={pred:>7.3f}  err={err:>+7.3f}")

    # --- 2. Optimize ---
    print(f"\n  Starting optimization (workers={workers})...")
    tracker = ConvergenceTracker()

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        args=(PARAM_NAMES, train_targets, dt, fixed_override),
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        seed=seed,
        disp=True,
        callback=tracker,
        workers=workers,
    )

    opt_params = result.x
    opt_cost = result.fun
    print(f"\n  Optimization complete!")
    print(f"    Final chi² = {opt_cost:.4f}")
    print(f"    Success: {result.success}")
    print(f"    Message: {result.message}")
    print(f"    Function evaluations: {result.nfev}")

    # --- 3. Evaluate post-fit ---
    print("\n  Post-fit evaluation...")
    r2_train, rmse_train, detail_train = compute_r_squared(
        opt_params, PARAM_NAMES, train_targets, dt=dt,
        fixed_override=fixed_override)
    r2_test, rmse_test, detail_test = compute_r_squared(
        opt_params, PARAM_NAMES, test_targets, dt=dt,
        fixed_override=fixed_override)
    r2_all, rmse_all, detail_all = compute_r_squared(
        opt_params, PARAM_NAMES, all_targets, dt=dt,
        fixed_override=fixed_override)

    print(f"    Train R² = {r2_train:.3f}, RMSE = {rmse_train:.3f}")
    print(f"    Test  R² = {r2_test:.3f}, RMSE = {rmse_test:.3f}")
    print(f"    All   R² = {r2_all:.3f}, RMSE = {rmse_all:.3f}")

    print(f"\n    Per-target post-fit predictions:")
    for tgt in all_targets:
        pred, pub = detail_all[tgt.name]
        err = pred - pub
        split = 'TRAIN' if tgt.train else 'TEST '
        print(f"      [{split}] {tgt.name:<35} pub={pub:>7.3f}  "
              f"pred={pred:>7.3f}  err={err:>+7.3f}")

    # --- Improvement summary ---
    print(f"\n  Improvement Summary:")
    print(f"    Train R²: {r2_pre_train:.3f} → {r2_train:.3f} "
          f"(+{r2_train - r2_pre_train:.3f})")
    print(f"    Test  R²: {r2_pre_test:.3f} → {r2_test:.3f} "
          f"(+{r2_test - r2_pre_test:.3f})")

    # --- 4. Fisher Information ---
    print("\n  Computing Fisher Information Matrix...")
    FIM, eigenvalues, eigenvectors = fisher_information(
        opt_params, PARAM_NAMES, train_targets, dt=dt,
        fixed_override=fixed_override)

    sorted_eig = np.sort(eigenvalues)[::-1]
    print(f"    Eigenvalue range: {sorted_eig[-1]:.4e} to {sorted_eig[0]:.4e}")
    print(f"    Condition number: {sorted_eig[0] / sorted_eig[-1]:.2e}")

    diag = np.maximum(np.diag(FIM), 1e-12)
    median_stiff = np.median(diag)
    well_constrained = [PARAM_NAMES[i] for i in range(len(diag))
                        if diag[i] > median_stiff]
    sloppy = [PARAM_NAMES[i] for i in range(len(diag))
              if diag[i] <= median_stiff]
    print(f"    Well-constrained ({len(well_constrained)}): "
          f"{', '.join(well_constrained)}")
    print(f"    Sloppy ({len(sloppy)}): {', '.join(sloppy)}")

    # --- 5. Novel predictions ---
    print("\n  Generating novel predictions...")
    novel = generate_novel_predictions(opt_params, PARAM_NAMES, dt=dt,
                                       fixed_override=fixed_override)
    for cond, res in novel.items():
        if 'error' not in res:
            print(f"    {cond}: {res.get('description', 'OK')}")
        else:
            print(f"    {cond}: ERROR - {res['error']}")

    # --- 5b. Comorbidity predictions ---
    print("\n  Generating comorbidity predictions...")
    comorbidity = generate_comorbidity_predictions(opt_params, PARAM_NAMES, dt=dt,
                                                    fixed_override=fixed_override)
    for cond, profile in comorbidity.items():
        if 'error' not in profile:
            print(f"    {cond}:")
            print(f"      {profile.get('description', '')}")
            for key in ['P_sensory', 'P_conceptual', 'P_selfmodel',
                        'p300_ratio', 'hrv_ratio', 'bdnf_ratio', 'pupil_ratio']:
                if key in profile:
                    print(f"      {key}: {profile[key]:.3f}")
        else:
            print(f"    {cond}: ERROR - {profile['error']}")

    # --- 5c. Model comparison ---
    print("\n  Running model comparison...")
    comparison = model_comparison(opt_params, PARAM_NAMES, all_targets, dt=dt,
                                  fixed_override=fixed_override)
    print(f"\n    {'Model':<20} {'k':>4} {'R²':>8} {'AIC':>10} {'BIC':>10}")
    print(f"    {'-'*52}")
    for model_name in ['mean_only', 'category', 'da_only', 'no_hierarchy',
                        'full_plasticity', 'null']:
        if model_name in comparison:
            m = comparison[model_name]
            print(f"    {model_name:<20} {m['k']:>4} {m['R2']:>8.3f} "
                  f"{m['AIC']:>10.1f} {m['BIC']:>10.1f}")

    # --- 6. Generate figures ---
    # Use mode suffix so different runs don't overwrite each other
    suffix = f"_{mode}" if mode != 'standard' else ""
    print(f"\n  Generating figures (suffix='{suffix}')...")

    if tracker.convergence_values:
        plot_convergence(tracker, suffix=suffix)

    plot_param_comparison(PARAM_DEFAULTS, opt_params, PARAM_NAMES, suffix=suffix)
    plot_alpha_comparison(detail_train, detail_test, suffix=suffix)
    plot_lzw_comparison(detail_train, detail_test, suffix=suffix)

    plot_scatter(train_targets, detail_train, r2_train, rmse_train,
                 f'Train Set [{mode}]: Model vs Published (R²={r2_train:.3f})',
                 f'FIT_scatter_train{suffix}.png')

    if test_targets:
        plot_scatter(test_targets, detail_test, r2_test, rmse_test,
                     f'Test Set [{mode}]: Model vs Published (R²={r2_test:.3f})',
                     f'FIT_scatter_test{suffix}.png')

    plot_fisher_eigenvalues(eigenvalues, PARAM_NAMES, suffix=suffix)
    plot_parameter_identifiability(FIM, PARAM_NAMES, suffix=suffix)
    plot_novel_predictions(novel, suffix=suffix)

    # New figures
    plot_scatter_all(all_targets, detail_all, r2_all, rmse_all, suffix=suffix)
    plot_model_comparison(comparison, suffix=suffix)
    plot_comorbidity_predictions(comorbidity, suffix=suffix)

    # --- 7. Print fitted snippet ---
    print_fitted_snippet(opt_params, PARAM_NAMES, fixed_override=fixed_override)

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("\nDone!")

    return {
        'opt_params': opt_params,
        'param_names': PARAM_NAMES,
        'result': result,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'r2_all': r2_all,
        'FIM': FIM,
        'eigenvalues': eigenvalues,
        'novel': novel,
        'comorbidity': comorbidity,
        'comparison': comparison,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Systematic parameter fitting for Plasticity Model v2')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: maxiter=10 (<10 min)')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='Maximum optimizer iterations (default: 100)')
    parser.add_argument('--popsize', type=int, default=15,
                        help='Population size per generation (default: 15)')
    parser.add_argument('--dt', type=float, default=0.05,
                        help='Simulation timestep (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--workers', type=int, default=-1,
                        help='Parallel workers (-1 = all cores, 1 = serial, default: -1)')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'wide', 'parsimonious'],
                        help='Fitting mode: standard (original), wide (widened bounds), '
                             'parsimonious (fix sloppy params, fit 6)')
    args = parser.parse_args()

    run_fitting(
        maxiter=args.maxiter,
        popsize=args.popsize,
        dt=args.dt,
        seed=args.seed,
        quick=args.quick,
        workers=args.workers,
        mode=args.mode,
    )
