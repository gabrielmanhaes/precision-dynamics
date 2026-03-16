"""
Sensory Deprivation & Neuroplasticity Simulation
==================================================

Models float tank / REST (Restricted Environmental Stimulation Technique)
effects on precision weighting and neuroplasticity.

Sensory deprivation in the model:
  - Dramatically reduced sensory noise (no external stimuli)
  - Reduced NE gain (LC responds to novelty; no novelty → less arousal)
  - Enhanced endogenous plasticity (theta-state, parasympathetic shift)
  - Mild GABA enhancement (relaxation response)

Key predictions:
  1. Sensory P drops rapidly (no predictions to maintain)
  2. Conceptual P drops with delay (reduced top-down constraint)
  3. Self-model P may increase OR decrease depending on duration
  4. Plasticity increases — sensory deprivation as non-pharmacological psychedelic analog
  5. Extended deprivation (>6h) reverses: becomes stressful, NE rises

Simulations:
  A. Acute float session (90 min) — P trajectory + aftereffect
  B. Repeated weekly floats (8 weeks) — cumulative plasticity
  C. Float + microdose psilocybin — synergy exploration
  D. Float for depression vs PTSD — therapeutic application
  E. Duration-response: 1h, 3h, 6h, 12h — when does benefit reverse?

Run:
    python3 sim_sensory_deprivation.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from parameters import *
from model import (
    simulate_v2, SimulationState,
    p_to_eeg_alpha, p_to_lzw, p_to_p300,
    ne_to_pupil, ne_cort_to_hrv, plasticity_to_bdnf,
)


# ============================================================================
# OUTPUT
# ============================================================================
def _make_output_dir():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'figures', 'sensory_deprivation')
    run_dir = os.path.join(base, stamp)
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(base, 'latest')
    if os.path.islink(latest):
        os.remove(latest)
    try:
        os.symlink(run_dir, latest)
    except OSError:
        pass
    return run_dir


# ============================================================================
# SENSORY DEPRIVATION PARAMETERS
# ============================================================================
# Float tank conditions mapped to model parameters:
#   - noise_scale_per_level: [sensory, conceptual, selfmodel]
#     Sensory noise nearly eliminated; conceptual/self-model partially reduced
#   - ne_sensitization: <1.0 = NE desensitization (reduced arousal)
#   - endogenous_plasticity_scale: >1.0 = enhanced plasticity (theta rhythm)
#   - gaba_deficit: <0 = GABA potentiation (relaxation)

FLOAT_PARAMS = {
    'noise_scale_per_level': [0.05, 0.3, 0.7],  # sensory input almost zero
    'ne_sensitization': 0.5,                      # LC deactivated (no novelty)
    'endogenous_plasticity_scale': 1.4,           # theta state → enhanced plasticity
    'gaba_deficit': -0.08,                        # mild relaxation (parasympathetic)
}

# Extended deprivation becomes stressful after ~6h
EXTENDED_STRESS_ONSET = 6.0  # hours until deprivation becomes stressful


def _extract_snapshot(t, P, st, nm, t_start, t_end):
    """Extract waking biomarkers from a time window."""
    mask = (t >= t_start) & (t <= t_end)
    wake = nm['sleep'][mask] < 0.3
    if not np.any(wake):
        wake = np.ones(np.sum(mask), dtype=bool)
    return {
        'P_s': np.mean(P['sensory'][mask][wake]),
        'P_c': np.mean(P['conceptual'][mask][wake]),
        'P_sm': np.mean(P['selfmodel'][mask][wake]),
        'NE': np.mean(nm['NE'][mask][wake]),
        'DA': np.mean(nm['DA'][mask][wake]),
        'plasticity': np.mean(nm['endogenous_plasticity'][mask][wake]),
        'cortisol': np.mean(st['cortisol'][mask][wake]),
        'alpha': p_to_eeg_alpha(np.mean(P['conceptual'][mask][wake])),
        'lzw': p_to_lzw(np.mean(P['conceptual'][mask][wake])),
        'bdnf': plasticity_to_bdnf(np.mean(nm['endogenous_plasticity'][mask][wake])),
    }


def _depressed_state():
    """Evolve 4-week depressed baseline."""
    t, P, st, nm = simulate_v2(
        t_span=(6.0, 6.0 + 4 * 7 * 24), dt=0.2, seed=42,
        chronic_stress=0.8,
    )
    late = t > (t[-1] - 3 * 24)
    wake = nm['sleep'][late] < 0.3
    idx = np.where(late)[0]
    state = SimulationState(
        P_s=np.mean(P['sensory'][idx][wake]) if np.any(wake) else P_SENSORY_BASELINE,
        P_c=np.mean(P['conceptual'][idx][wake]) if np.any(wake) else P_CONCEPTUAL_BASELINE,
        P_sm=np.mean(P['selfmodel'][idx][wake]) if np.any(wake) else P_SELFMODEL_BASELINE,
        hpa_sensitivity=st['hpa_sensitivity'][idx[-1]],
        allostatic_load=st['allostatic_load'][idx[-1]],
        cortisol=st['cortisol'][idx[-1]],
    )
    snap = _extract_snapshot(t, P, st, nm, t[-1] - 3*24, t[-1])
    return state, snap


def _ptsd_state():
    """Evolve 4-week PTSD baseline."""
    t, P, st, nm = simulate_v2(
        t_span=(6.0, 6.0 + 4 * 7 * 24), dt=0.2, seed=42,
        ne_sensitization=1.8, coupling_breakdown=0.5,
        chronic_stress=0.4, td_coupling_scale=PTSD_TD_BREAKDOWN,
    )
    late = t > (t[-1] - 3 * 24)
    wake = nm['sleep'][late] < 0.3
    idx = np.where(late)[0]
    state = SimulationState(
        P_s=np.mean(P['sensory'][idx][wake]) if np.any(wake) else 0.85,
        P_c=np.mean(P['conceptual'][idx][wake]) if np.any(wake) else 0.70,
        P_sm=np.mean(P['selfmodel'][idx][wake]) if np.any(wake) else 0.45,
        hpa_sensitivity=st['hpa_sensitivity'][idx[-1]],
        allostatic_load=st['allostatic_load'][idx[-1]],
        cortisol=st['cortisol'][idx[-1]],
    )
    snap = _extract_snapshot(t, P, st, nm, t[-1] - 3*24, t[-1])
    return state, snap


def _normal_snapshot():
    t, P, st, nm = simulate_v2(t_span=(6.0, 30.0), dt=0.05, seed=42)
    return _extract_snapshot(t, P, st, nm, 8.0, 20.0)


# ============================================================================
# A. ACUTE FLOAT SESSION (90 min)
# ============================================================================
def sim_acute_float(output_dir):
    """
    90-minute float session starting at 2 PM.
    Three phases: pre-float (morning) → float → post-float (24h recovery)
    """
    print("\n" + "="*70)
    print("A. ACUTE FLOAT SESSION (90 min)")
    print("="*70)

    normal_snap = _normal_snapshot()

    # Phase 1: Normal morning until float starts (6 AM to 2 PM)
    t1, P1, st1, nm1 = simulate_v2(
        t_span=(6.0, 14.0), dt=0.02, seed=42,
    )

    # Phase 2: Float session (2 PM to 3:30 PM = 1.5h)
    state_float = SimulationState(
        P_s=P1['sensory'][-1], P_c=P1['conceptual'][-1],
        P_sm=P1['selfmodel'][-1],
        hpa_sensitivity=st1['hpa_sensitivity'][-1],
        allostatic_load=st1['allostatic_load'][-1],
        cortisol=st1['cortisol'][-1],
    )
    t2, P2, st2, nm2 = simulate_v2(
        t_span=(14.0, 15.5), dt=0.02, seed=42,
        state0=state_float,
        **FLOAT_PARAMS,
    )

    # Phase 3: Post-float recovery (3:30 PM to next day 6 PM = ~26.5h)
    state_post = SimulationState(
        P_s=P2['sensory'][-1], P_c=P2['conceptual'][-1],
        P_sm=P2['selfmodel'][-1],
        hpa_sensitivity=st2['hpa_sensitivity'][-1],
        allostatic_load=st2['allostatic_load'][-1],
        cortisol=st2['cortisol'][-1],
    )
    t3, P3, st3, nm3 = simulate_v2(
        t_span=(15.5, 42.0), dt=0.02, seed=42,
        state0=state_post,
    )

    # Control: no float, just normal 36h
    t_ctrl, P_ctrl, st_ctrl, nm_ctrl = simulate_v2(
        t_span=(6.0, 42.0), dt=0.02, seed=42,
    )

    # Combine float phases
    t_float = np.concatenate([t1, t2, t3])
    P_float = {k: np.concatenate([P1[k], P2[k], P3[k]]) for k in P1}
    nm_float = {k: np.concatenate([nm1[k], nm2[k], nm3[k]]) for k in nm1}
    st_float = {k: np.concatenate([st1[k], st2[k], st3[k]]) for k in st1}

    # Print key timepoints
    timepoints = [
        (13.0, 14.0, 'Pre-float (1 PM)'),
        (14.5, 15.0, 'Mid-float'),
        (15.3, 15.5, 'End-float'),
        (16.0, 17.0, 'Post-float +1h'),
        (18.0, 20.0, 'Post-float +4h'),
        (30.0, 34.0, 'Next morning'),
    ]
    print(f"\n  {'Timepoint':<25} {'P_sens':>8} {'P_conc':>8} {'P_self':>8} "
          f"{'NE':>8} {'Plast':>8} {'Alpha':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for t_s, t_e, label in timepoints:
        snap = _extract_snapshot(t_float, P_float, st_float, nm_float, t_s, t_e)
        print(f"  {label:<25} {snap['P_s']:8.3f} {snap['P_c']:8.3f} {snap['P_sm']:8.3f} "
              f"{snap['NE']:8.3f} {snap['plasticity']:8.3f} {snap['alpha']:8.4f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    levels = ['sensory', 'conceptual', 'selfmodel']
    level_labels = ['P Sensory', 'P Conceptual', 'P Self-Model']

    for i, (level, lbl) in enumerate(zip(levels, level_labels)):
        ax = axes[0, i]
        ax.plot(t_float - 6.0, P_float[level], 'b-', linewidth=1, alpha=0.7,
                label='Float session')
        ax.plot(t_ctrl - 6.0, P_ctrl[level], 'gray', linewidth=0.8, alpha=0.4,
                label='Normal day')
        ax.axvspan(14.0 - 6.0, 15.5 - 6.0, alpha=0.15, color='cyan', label='Float')
        ax.set_ylabel(lbl)
        ax.set_xlabel('Hours from 6 AM')
        ax.legend(fontsize=6)

    # Bottom left: NE trajectory
    ax = axes[1, 0]
    ax.plot(t_float - 6.0, nm_float['NE'], 'b-', linewidth=1, alpha=0.7, label='Float')
    ax.plot(t_ctrl - 6.0, nm_ctrl['NE'], 'gray', linewidth=0.8, alpha=0.4, label='Normal')
    ax.axvspan(14.0 - 6.0, 15.5 - 6.0, alpha=0.15, color='cyan')
    ax.set_ylabel('Norepinephrine')
    ax.set_xlabel('Hours from 6 AM')
    ax.legend(fontsize=6)

    # Bottom middle: Endogenous plasticity
    ax = axes[1, 1]
    ax.plot(t_float - 6.0, nm_float['endogenous_plasticity'], 'b-', linewidth=1,
            alpha=0.7, label='Float')
    ax.plot(t_ctrl - 6.0, nm_ctrl['endogenous_plasticity'], 'gray', linewidth=0.8,
            alpha=0.4, label='Normal')
    ax.axvspan(14.0 - 6.0, 15.5 - 6.0, alpha=0.15, color='cyan')
    ax.set_ylabel('Endogenous Plasticity')
    ax.set_xlabel('Hours from 6 AM')
    ax.legend(fontsize=6)

    # Bottom right: EEG alpha
    ax = axes[1, 2]
    alpha_float = np.array([p_to_eeg_alpha(p) for p in P_float['conceptual']])
    alpha_ctrl = np.array([p_to_eeg_alpha(p) for p in P_ctrl['conceptual']])
    ax.plot(t_float - 6.0, alpha_float, 'b-', linewidth=1, alpha=0.7, label='Float')
    ax.plot(t_ctrl - 6.0, alpha_ctrl, 'gray', linewidth=0.8, alpha=0.4, label='Normal')
    ax.axvspan(14.0 - 6.0, 15.5 - 6.0, alpha=0.15, color='cyan')
    ax.set_ylabel('EEG Alpha Power')
    ax.set_xlabel('Hours from 6 AM')
    ax.legend(fontsize=6)

    fig.suptitle('Acute Float Session (90 min) — Precision & Plasticity',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'SD_acute_float.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: SD_acute_float.png")


# ============================================================================
# B. REPEATED WEEKLY FLOATS (8 weeks)
# ============================================================================
def sim_repeated_floats(output_dir):
    """
    Weekly 90-min float sessions for 8 weeks.
    Track cumulative effects on plasticity, P levels, and biomarkers.
    """
    print("\n" + "="*70)
    print("B. REPEATED WEEKLY FLOATS (8 weeks)")
    print("="*70)

    WEEKS = 8
    FLOAT_DURATION = 1.5  # hours
    FLOAT_TIME = 14.0     # 2 PM each session day

    # Weekly metrics
    weekly_data = {'P_s': [], 'P_c': [], 'P_sm': [], 'plasticity': [],
                   'bdnf': [], 'NE': [], 'cortisol': []}
    weekly_data_ctrl = {'P_s': [], 'P_c': [], 'P_sm': [], 'plasticity': [],
                        'bdnf': [], 'NE': [], 'cortisol': []}

    # --- Float protocol ---
    state = None  # will use defaults
    for week in range(WEEKS):
        week_start = 6.0 + week * 7 * 24

        # Day 1: float day
        # Pre-float
        t_pre, P_pre, st_pre, nm_pre = simulate_v2(
            t_span=(week_start, week_start + 8.0), dt=0.1, seed=42 + week,
            state0=state,
        )
        # Float
        state_f = SimulationState(
            P_s=P_pre['sensory'][-1], P_c=P_pre['conceptual'][-1],
            P_sm=P_pre['selfmodel'][-1],
            hpa_sensitivity=st_pre['hpa_sensitivity'][-1],
            allostatic_load=st_pre['allostatic_load'][-1],
            cortisol=st_pre['cortisol'][-1],
        )
        t_f, P_f, st_f, nm_f = simulate_v2(
            t_span=(week_start + 8.0, week_start + 8.0 + FLOAT_DURATION),
            dt=0.05, seed=42 + week,
            state0=state_f,
            **FLOAT_PARAMS,
        )
        # Post-float: rest of the week
        state_post = SimulationState(
            P_s=P_f['sensory'][-1], P_c=P_f['conceptual'][-1],
            P_sm=P_f['selfmodel'][-1],
            hpa_sensitivity=st_f['hpa_sensitivity'][-1],
            allostatic_load=st_f['allostatic_load'][-1],
            cortisol=st_f['cortisol'][-1],
        )
        t_rest, P_rest, st_rest, nm_rest = simulate_v2(
            t_span=(week_start + 8.0 + FLOAT_DURATION, week_start + 7 * 24),
            dt=0.2, seed=42 + week,
            state0=state_post,
        )

        # Update state for next week
        state = SimulationState(
            P_s=P_rest['sensory'][-1], P_c=P_rest['conceptual'][-1],
            P_sm=P_rest['selfmodel'][-1],
            hpa_sensitivity=st_rest['hpa_sensitivity'][-1],
            allostatic_load=st_rest['allostatic_load'][-1],
            cortisol=st_rest['cortisol'][-1],
        )

        # Extract end-of-week snapshot (last 2 days, waking)
        snap = _extract_snapshot(t_rest, P_rest, st_rest, nm_rest,
                                 t_rest[-1] - 2*24, t_rest[-1])
        for k in weekly_data:
            weekly_data[k].append(snap[k])

    # --- Control (no floats) ---
    state_ctrl = None
    for week in range(WEEKS):
        week_start = 6.0 + week * 7 * 24
        t_c, P_c, st_c, nm_c = simulate_v2(
            t_span=(week_start, week_start + 7 * 24), dt=0.2, seed=42 + week,
            state0=state_ctrl,
        )
        state_ctrl = SimulationState(
            P_s=P_c['sensory'][-1], P_c=P_c['conceptual'][-1],
            P_sm=P_c['selfmodel'][-1],
            hpa_sensitivity=st_c['hpa_sensitivity'][-1],
            allostatic_load=st_c['allostatic_load'][-1],
            cortisol=st_c['cortisol'][-1],
        )
        snap = _extract_snapshot(t_c, P_c, st_c, nm_c, t_c[-1] - 2*24, t_c[-1])
        for k in weekly_data_ctrl:
            weekly_data_ctrl[k].append(snap[k])

    weeks = list(range(1, WEEKS + 1))
    print(f"\n  {'Week':>6} {'P_c (float)':>12} {'P_c (ctrl)':>12} {'Plast':>10} "
          f"{'BDNF':>8} {'NE':>8}")
    for w in range(WEEKS):
        print(f"  {w+1:>6} {weekly_data['P_c'][w]:12.4f} {weekly_data_ctrl['P_c'][w]:12.4f} "
              f"{weekly_data['plasticity'][w]:10.4f} {weekly_data['bdnf'][w]:8.4f} "
              f"{weekly_data['NE'][w]:8.4f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(weeks, weekly_data['P_c'], 'b-o', markersize=4, label='Float protocol')
    ax.plot(weeks, weekly_data_ctrl['P_c'], 'gray', linestyle='--', label='Control')
    ax.set_xlabel('Week')
    ax.set_ylabel('P conceptual')
    ax.set_title('Precision (end of week)')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(weeks, weekly_data['plasticity'], 'b-o', markersize=4, label='Float')
    ax.plot(weeks, weekly_data_ctrl['plasticity'], 'gray', linestyle='--', label='Control')
    ax.set_xlabel('Week')
    ax.set_ylabel('Endogenous plasticity')
    ax.set_title('Plasticity (cumulative effect)')
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(weeks, weekly_data['bdnf'], 'b-o', markersize=4, label='Float')
    ax.plot(weeks, weekly_data_ctrl['bdnf'], 'gray', linestyle='--', label='Control')
    ax.set_xlabel('Week')
    ax.set_ylabel('BDNF')
    ax.set_title('BDNF (plasticity biomarker)')
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(weeks, weekly_data['cortisol'], 'b-o', markersize=4, label='Float')
    ax.plot(weeks, weekly_data_ctrl['cortisol'], 'gray', linestyle='--', label='Control')
    ax.set_xlabel('Week')
    ax.set_ylabel('Cortisol')
    ax.set_title('Stress (cortisol)')
    ax.legend(fontsize=7)

    fig.suptitle('Repeated Weekly Float Sessions (8 weeks)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'SD_repeated_floats.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: SD_repeated_floats.png")

    return weekly_data


# ============================================================================
# C. FLOAT + MICRODOSE PSILOCYBIN SYNERGY
# ============================================================================
def sim_float_plus_microdose(output_dir):
    """
    Does combining a float session with a microdose amplify the plasticity effect?

    Three conditions on a depressed baseline:
    1. Float only (90 min)
    2. Microdose only (0.10 dose at 8 AM)
    3. Microdose at 8 AM + float at 2 PM (combined)
    4. No treatment (control)
    """
    print("\n" + "="*70)
    print("C. FLOAT + MICRODOSE PSILOCYBIN SYNERGY")
    print("="*70)

    dep_state, dep_snap = _depressed_state()
    normal_snap = _normal_snapshot()
    print(f"  Depressed baseline: P_c={dep_snap['P_c']:.3f}")
    print(f"  Normal baseline:    P_c={normal_snap['P_c']:.3f}")

    conditions = {}

    # 1. Float only
    # Pre-float
    t1a, P1a, st1a, nm1a = simulate_v2(
        t_span=(6.0, 14.0), dt=0.05, seed=42,
        state0=dep_state, chronic_stress=0.5,
    )
    sf1 = SimulationState(P_s=P1a['sensory'][-1], P_c=P1a['conceptual'][-1],
                          P_sm=P1a['selfmodel'][-1],
                          hpa_sensitivity=st1a['hpa_sensitivity'][-1],
                          allostatic_load=st1a['allostatic_load'][-1],
                          cortisol=st1a['cortisol'][-1])
    t1b, P1b, st1b, nm1b = simulate_v2(
        t_span=(14.0, 15.5), dt=0.02, seed=42,
        state0=sf1, chronic_stress=0.5, **FLOAT_PARAMS,
    )
    sp1 = SimulationState(P_s=P1b['sensory'][-1], P_c=P1b['conceptual'][-1],
                          P_sm=P1b['selfmodel'][-1],
                          hpa_sensitivity=st1b['hpa_sensitivity'][-1],
                          allostatic_load=st1b['allostatic_load'][-1],
                          cortisol=st1b['cortisol'][-1])
    t1c, P1c, st1c, nm1c = simulate_v2(
        t_span=(15.5, 30.0), dt=0.05, seed=42,
        state0=sp1, chronic_stress=0.5,
    )
    t_f = np.concatenate([t1a, t1b, t1c])
    P_f = {k: np.concatenate([P1a[k], P1b[k], P1c[k]]) for k in P1a}
    nm_f = {k: np.concatenate([nm1a[k], nm1b[k], nm1c[k]]) for k in nm1a}
    st_f = {k: np.concatenate([st1a[k], st1b[k], st1c[k]]) for k in st1a}
    conditions['Float only'] = (t_f, P_f, st_f, nm_f)

    # 2. Microdose only
    t2, P2, st2, nm2 = simulate_v2(
        t_span=(6.0, 30.0), dt=0.05, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(8.0, 0.10)],  # microdose at 8 AM
    )
    conditions['Microdose only'] = (t2, P2, st2, nm2)

    # 3. Microdose + Float combined
    t3a, P3a, st3a, nm3a = simulate_v2(
        t_span=(6.0, 14.0), dt=0.05, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(8.0, 0.10)],
    )
    sf3 = SimulationState(P_s=P3a['sensory'][-1], P_c=P3a['conceptual'][-1],
                          P_sm=P3a['selfmodel'][-1],
                          hpa_sensitivity=st3a['hpa_sensitivity'][-1],
                          allostatic_load=st3a['allostatic_load'][-1],
                          cortisol=st3a['cortisol'][-1])
    t3b, P3b, st3b, nm3b = simulate_v2(
        t_span=(14.0, 15.5), dt=0.02, seed=42,
        state0=sf3, chronic_stress=0.5,
        pharma_psilocybin=[(8.0, 0.10)],  # still active
        **FLOAT_PARAMS,
    )
    sp3 = SimulationState(P_s=P3b['sensory'][-1], P_c=P3b['conceptual'][-1],
                          P_sm=P3b['selfmodel'][-1],
                          hpa_sensitivity=st3b['hpa_sensitivity'][-1],
                          allostatic_load=st3b['allostatic_load'][-1],
                          cortisol=st3b['cortisol'][-1])
    t3c, P3c, st3c, nm3c = simulate_v2(
        t_span=(15.5, 30.0), dt=0.05, seed=42,
        state0=sp3, chronic_stress=0.5,
    )
    t_combo = np.concatenate([t3a, t3b, t3c])
    P_combo = {k: np.concatenate([P3a[k], P3b[k], P3c[k]]) for k in P3a}
    nm_combo = {k: np.concatenate([nm3a[k], nm3b[k], nm3c[k]]) for k in nm3a}
    st_combo = {k: np.concatenate([st3a[k], st3b[k], st3c[k]]) for k in st3a}
    conditions['Microdose + Float'] = (t_combo, P_combo, st_combo, nm_combo)

    # 4. No treatment
    t4, P4, st4, nm4 = simulate_v2(
        t_span=(6.0, 30.0), dt=0.05, seed=42,
        state0=dep_state, chronic_stress=0.5,
    )
    conditions['No treatment'] = (t4, P4, st4, nm4)

    # Print comparison at evening (6 PM, post-float window)
    print(f"\n  Biomarkers at 6 PM (post-float window):")
    print(f"  {'Condition':<25} {'P_c':>8} {'P_sm':>8} {'Plast':>8} {'BDNF':>8} {'LZW':>8}")
    for name, (t, P, st, nm) in conditions.items():
        snap = _extract_snapshot(t, P, st, nm, 18.0, 20.0)
        print(f"  {name:<25} {snap['P_c']:8.4f} {snap['P_sm']:8.4f} "
              f"{snap['plasticity']:8.4f} {snap['bdnf']:8.4f} {snap['lzw']:8.4f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Float only': 'steelblue', 'Microdose only': 'darkorange',
              'Microdose + Float': 'firebrick', 'No treatment': 'gray'}

    # P_conceptual trajectories
    ax = axes[0]
    for name, (t, P, st, nm) in conditions.items():
        ax.plot(t - 6.0, P['conceptual'], color=colors[name], alpha=0.6,
                linewidth=1.2, label=name)
    ax.axvspan(14.0 - 6.0, 15.5 - 6.0, alpha=0.12, color='cyan', label='Float window')
    ax.axhline(normal_snap['P_c'], color='green', ls='--', alpha=0.3, label='Normal')
    ax.set_xlabel('Hours from 6 AM')
    ax.set_ylabel('P conceptual')
    ax.set_title('Precision Trajectory')
    ax.legend(fontsize=6)

    # Plasticity trajectories
    ax = axes[1]
    for name, (t, P, st, nm) in conditions.items():
        ax.plot(t - 6.0, nm['endogenous_plasticity'], color=colors[name],
                alpha=0.6, linewidth=1.2, label=name)
    ax.axvspan(14.0 - 6.0, 15.5 - 6.0, alpha=0.12, color='cyan')
    ax.set_xlabel('Hours from 6 AM')
    ax.set_ylabel('Endogenous Plasticity')
    ax.set_title('Plasticity Drive')
    ax.legend(fontsize=6)

    # Bar chart: improvement over no treatment
    ax = axes[2]
    notx_snap = _extract_snapshot(t4, P4, st4, nm4, 18.0, 20.0)
    cond_names = ['Float only', 'Microdose only', 'Microdose + Float']
    metrics = ['P_c reduction', 'Plasticity boost', 'BDNF boost']
    improvements = []
    for name in cond_names:
        t, P, st, nm = conditions[name]
        snap = _extract_snapshot(t, P, st, nm, 18.0, 20.0)
        improvements.append([
            notx_snap['P_c'] - snap['P_c'],          # P reduction (positive = better)
            snap['plasticity'] - notx_snap['plasticity'],  # plasticity increase
            snap['bdnf'] - notx_snap['bdnf'],               # BDNF increase
        ])
    x = np.arange(len(metrics))
    width = 0.25
    for i, (name, impr) in enumerate(zip(cond_names, improvements)):
        ax.bar(x + i * width, impr, width, color=colors[name], alpha=0.8, label=name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel('Improvement over no treatment')
    ax.set_title('Synergy Check')
    ax.legend(fontsize=6)
    ax.axhline(0, color='black', linewidth=0.5)

    fig.suptitle('Float + Microdose Psilocybin: Synergy on Depressed Baseline',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'SD_float_microdose_synergy.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: SD_float_microdose_synergy.png")


# ============================================================================
# D. FLOAT FOR DEPRESSION vs PTSD
# ============================================================================
def sim_float_therapeutic(output_dir):
    """
    Float session on depressed vs PTSD baseline.

    MODEL PREDICTION:
    - Depression (high P): Float reduces P → therapeutic ✓
    - PTSD (high P_sensory, low P_selfmodel): Float reduces sensory input →
      P_sensory drops (good for hyperarousal), but self-model in isolation
      could trigger dissociative symptoms in some patients
    """
    print("\n" + "="*70)
    print("D. FLOAT FOR DEPRESSION vs PTSD")
    print("="*70)

    dep_state, dep_snap = _depressed_state()
    ptsd_state, ptsd_snap = _ptsd_state()
    normal_snap = _normal_snapshot()

    print(f"  Normal:     P_s={normal_snap['P_s']:.3f}, P_c={normal_snap['P_c']:.3f}, "
          f"P_sm={normal_snap['P_sm']:.3f}")
    print(f"  Depressed:  P_s={dep_snap['P_s']:.3f}, P_c={dep_snap['P_c']:.3f}, "
          f"P_sm={dep_snap['P_sm']:.3f}")
    print(f"  PTSD:       P_s={ptsd_snap['P_s']:.3f}, P_c={ptsd_snap['P_c']:.3f}, "
          f"P_sm={ptsd_snap['P_sm']:.3f}")

    results = {}

    for cond_name, state0, extra_kw in [
        ('Depression', dep_state, {'chronic_stress': 0.5}),
        ('PTSD', ptsd_state, {'ne_sensitization': 1.8, 'coupling_breakdown': 0.5,
                               'chronic_stress': 0.3, 'td_coupling_scale': PTSD_TD_BREAKDOWN}),
        ('Normal', None, {}),
    ]:
        # Pre-float
        t1, P1, st1, nm1 = simulate_v2(
            t_span=(6.0, 14.0), dt=0.05, seed=42,
            state0=state0, **extra_kw,
        )
        sf = SimulationState(
            P_s=P1['sensory'][-1], P_c=P1['conceptual'][-1],
            P_sm=P1['selfmodel'][-1],
            hpa_sensitivity=st1['hpa_sensitivity'][-1],
            allostatic_load=st1['allostatic_load'][-1],
            cortisol=st1['cortisol'][-1],
        )
        # Float
        float_kw = dict(FLOAT_PARAMS)
        float_kw.update(extra_kw)
        t2, P2, st2, nm2 = simulate_v2(
            t_span=(14.0, 15.5), dt=0.02, seed=42,
            state0=sf, **float_kw,
        )
        sp = SimulationState(
            P_s=P2['sensory'][-1], P_c=P2['conceptual'][-1],
            P_sm=P2['selfmodel'][-1],
            hpa_sensitivity=st2['hpa_sensitivity'][-1],
            allostatic_load=st2['allostatic_load'][-1],
            cortisol=st2['cortisol'][-1],
        )
        # Post-float
        t3, P3, st3, nm3 = simulate_v2(
            t_span=(15.5, 30.0), dt=0.05, seed=42,
            state0=sp, **extra_kw,
        )

        t_all = np.concatenate([t1, t2, t3])
        P_all = {k: np.concatenate([P1[k], P2[k], P3[k]]) for k in P1}
        st_all = {k: np.concatenate([st1[k], st2[k], st3[k]]) for k in st1}
        nm_all = {k: np.concatenate([nm1[k], nm2[k], nm3[k]]) for k in nm1}
        results[cond_name] = (t_all, P_all, st_all, nm_all)

    # Print pre/during/post-float for each condition
    phases = [('Pre-float', 13.0, 14.0), ('Mid-float', 14.5, 15.0),
              ('Post +1h', 16.0, 17.0), ('Post +4h', 18.0, 20.0)]
    for cond_name in ['Depression', 'PTSD', 'Normal']:
        t, P, st, nm = results[cond_name]
        print(f"\n  {cond_name}:")
        print(f"    {'Phase':<15} {'dP_s':>8} {'dP_c':>8} {'dP_sm':>8} {'Plasticity':>10}")
        baseline_snap = _extract_snapshot(t, P, st, nm, 10.0, 13.0)
        for phase_name, ts, te in phases:
            snap = _extract_snapshot(t, P, st, nm, ts, te)
            dPs = snap['P_s'] - baseline_snap['P_s']
            dPc = snap['P_c'] - baseline_snap['P_c']
            dPsm = snap['P_sm'] - baseline_snap['P_sm']
            print(f"    {phase_name:<15} {dPs:+8.4f} {dPc:+8.4f} {dPsm:+8.4f} "
                  f"{snap['plasticity']:10.4f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    cond_colors = {'Depression': 'steelblue', 'PTSD': 'firebrick', 'Normal': 'seagreen'}
    levels = ['sensory', 'conceptual', 'selfmodel']
    level_labels = ['P Sensory', 'P Conceptual', 'P Self-Model']

    for row, cond_name in enumerate(['Normal', 'Depression', 'PTSD']):
        t, P, st, nm = results[cond_name]
        for col, (level, lbl) in enumerate(zip(levels, level_labels)):
            ax = axes[row, col]
            ax.plot(t - 6.0, P[level], color=cond_colors[cond_name],
                    linewidth=1.2, alpha=0.7)
            ax.axvspan(14.0 - 6.0, 15.5 - 6.0, alpha=0.12, color='cyan')
            if row == 0:
                ax.set_title(lbl, fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{cond_name}\nP value', fontsize=9)
            if row == 2:
                ax.set_xlabel('Hours from 6 AM')

    fig.suptitle('Float Session: Normal vs Depression vs PTSD',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'SD_float_therapeutic.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: SD_float_therapeutic.png")


# ============================================================================
# E. DURATION-RESPONSE: When does deprivation become harmful?
# ============================================================================
def sim_duration_response(output_dir):
    """
    Sensory deprivation at increasing durations: 1h, 3h, 6h, 12h, 24h.

    MODEL PREDICTION:
    - Short (1-3h): Therapeutic — reduced NE, enhanced plasticity, P drops
    - Medium (6h): Peak benefit — maximum plasticity window
    - Extended (12-24h): Becomes stressful — the brain interprets prolonged
      absence of input as threat, NE rises, cortisol rises, P increases
      This mirrors real float research: extended sessions can cause anxiety

    Implemented: After EXTENDED_STRESS_ONSET hours, chronic_stress ramps up
    """
    print("\n" + "="*70)
    print("E. DURATION-RESPONSE (1h to 24h)")
    print("="*70)

    durations = [1.0, 2.0, 3.0, 6.0, 12.0, 24.0]  # hours
    duration_results = {}

    for dur in durations:
        # Pre-deprivation (6 AM to 2 PM)
        t1, P1, st1, nm1 = simulate_v2(
            t_span=(6.0, 14.0), dt=0.05, seed=42,
        )
        sf = SimulationState(
            P_s=P1['sensory'][-1], P_c=P1['conceptual'][-1],
            P_sm=P1['selfmodel'][-1],
            hpa_sensitivity=st1['hpa_sensitivity'][-1],
            allostatic_load=st1['allostatic_load'][-1],
            cortisol=st1['cortisol'][-1],
        )

        # Deprivation phase
        # After EXTENDED_STRESS_ONSET hours, add escalating stress
        deprivation_stress = max(0, (dur - EXTENDED_STRESS_ONSET) * 0.15)
        float_kw = dict(FLOAT_PARAMS)
        float_kw['chronic_stress'] = deprivation_stress

        t2, P2, st2, nm2 = simulate_v2(
            t_span=(14.0, 14.0 + dur), dt=0.02, seed=42,
            state0=sf, **float_kw,
        )

        # Post-deprivation recovery (24h)
        sp = SimulationState(
            P_s=P2['sensory'][-1], P_c=P2['conceptual'][-1],
            P_sm=P2['selfmodel'][-1],
            hpa_sensitivity=st2['hpa_sensitivity'][-1],
            allostatic_load=st2['allostatic_load'][-1],
            cortisol=st2['cortisol'][-1],
        )
        t3, P3, st3, nm3 = simulate_v2(
            t_span=(14.0 + dur, 14.0 + dur + 24), dt=0.05, seed=42,
            state0=sp,
        )

        # Snapshots
        pre_snap = _extract_snapshot(
            np.concatenate([t1]), {k: P1[k] for k in P1},
            {k: st1[k] for k in st1}, {k: nm1[k] for k in nm1},
            10.0, 14.0)
        during_snap = _extract_snapshot(
            t2, P2, st2, nm2, 14.0, 14.0 + dur)
        # Post-recovery: 4h after session ends
        post_snap = _extract_snapshot(
            t3, P3, st3, nm3,
            14.0 + dur + 2, 14.0 + dur + 6)

        duration_results[dur] = {
            'pre': pre_snap,
            'during': during_snap,
            'post': post_snap,
            'stress_added': deprivation_stress,
        }

    # Print summary
    print(f"\n  {'Duration':>10} {'Stress':>8} {'dP_c (during)':>14} {'dP_c (post)':>12} "
          f"{'Plast (dur)':>12} {'Plast (post)':>12}")
    for dur in durations:
        d = duration_results[dur]
        dPc_dur = d['during']['P_c'] - d['pre']['P_c']
        dPc_post = d['post']['P_c'] - d['pre']['P_c']
        print(f"  {dur:>8.0f}h {d['stress_added']:>8.2f} {dPc_dur:>+14.4f} {dPc_post:>+12.4f} "
              f"{d['during']['plasticity']:>12.4f} {d['post']['plasticity']:>12.4f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    dur_vals = durations
    dPc_during = [duration_results[d]['during']['P_c'] - duration_results[d]['pre']['P_c']
                  for d in dur_vals]
    dPc_post = [duration_results[d]['post']['P_c'] - duration_results[d]['pre']['P_c']
                for d in dur_vals]
    plast_during = [duration_results[d]['during']['plasticity'] for d in dur_vals]
    plast_post = [duration_results[d]['post']['plasticity'] for d in dur_vals]
    cort_during = [duration_results[d]['during']['cortisol'] for d in dur_vals]

    # Panel A: P_c change vs duration
    ax = axes[0]
    ax.plot(dur_vals, dPc_during, 'b-o', label='During session', markersize=5)
    ax.plot(dur_vals, dPc_post, 'r-s', label='Post-session (+4h)', markersize=5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(EXTENDED_STRESS_ONSET, color='orange', linestyle=':', alpha=0.5,
               label=f'Stress onset ({EXTENDED_STRESS_ONSET}h)')
    ax.set_xlabel('Deprivation Duration (hours)')
    ax.set_ylabel('P_c change from baseline')
    ax.set_title('Precision: Therapeutic Window')
    ax.legend(fontsize=7)

    # Panel B: Plasticity vs duration
    ax = axes[1]
    ax.plot(dur_vals, plast_during, 'b-o', label='During', markersize=5)
    ax.plot(dur_vals, plast_post, 'r-s', label='Post (+4h)', markersize=5)
    ax.axvline(EXTENDED_STRESS_ONSET, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Deprivation Duration (hours)')
    ax.set_ylabel('Endogenous Plasticity')
    ax.set_title('Plasticity: Peak and Rebound')
    ax.legend(fontsize=7)

    # Panel C: Cortisol vs duration
    ax = axes[2]
    ax.plot(dur_vals, cort_during, 'b-o', markersize=5)
    ax.axvline(EXTENDED_STRESS_ONSET, color='orange', linestyle=':', alpha=0.5,
               label=f'Stress onset')
    ax.set_xlabel('Deprivation Duration (hours)')
    ax.set_ylabel('Mean Cortisol')
    ax.set_title('Stress Response')
    ax.legend(fontsize=7)

    fig.suptitle('Sensory Deprivation Duration-Response Curve',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'SD_duration_response.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: SD_duration_response.png")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("="*70)
    print("Sensory Deprivation & Neuroplasticity Simulation")
    print("="*70)
    print("""
    KEY PREDICTIONS:
    1. Float tanks produce a transient "precision holiday" — sensory P drops
       rapidly, enabling plasticity-driven reorganization similar to dreaming
    2. Weekly floats produce cumulative plasticity gains (BDNF measurable)
    3. Float + microdose psilocybin shows SYNERGY — the sensory isolation
       amplifies psilocybin's plasticity drive by removing competing signals
    4. For PTSD: float helps hyperarousal (P_sensory drops) but self-model
       changes unpredictable (could trigger dissociation in vulnerable patients)
    5. Optimal duration: 1-3 hours. Beyond 6h, deprivation becomes stressful
       and the NE/cortisol rebound REVERSES the therapeutic effect
    """)

    output_dir = _make_output_dir()
    print(f"Output: {output_dir}\n")

    sim_acute_float(output_dir)
    sim_repeated_floats(output_dir)
    sim_float_plus_microdose(output_dir)
    sim_float_therapeutic(output_dir)
    sim_duration_response(output_dir)

    print("\n" + "="*70)
    print("ALL SENSORY DEPRIVATION SIMULATIONS COMPLETE")
    print(f"Figures saved to: {output_dir}")
    print("="*70)
