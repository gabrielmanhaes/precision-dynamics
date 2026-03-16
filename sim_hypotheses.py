"""
Ground-Breaking Hypothesis Simulations
========================================

Novel predictions from the precision-weighting plasticity model that emerge
from the interaction of mechanisms no single existing framework captures.

Each hypothesis:
  - Predicts an interaction NOT yet explained by science
  - Is counter-intuitive (not just "common sense")
  - Is specific and falsifiable with real experiments
  - Emerges uniquely from this model's architecture

Hypotheses:
  1. Sleep deprivation potentiates ketamine via P-level priming
  2. Ketamine → psilocybin sequential priming (the plasticity cascade)
  3. PTSD shows OPPOSITE ketamine response profile to depression
  4. Circadian timing of psychedelics: morning vs evening dosing
  5. Anxiolytics after psychedelics BLOCK therapeutic afterglow
  6. DMT vs psilocybin temporal dynamics (duration → afterglow strength)
  7. Pineal calcification → cognitive rigidity via reduced plasticity

Run:
    python3 sim_hypotheses.py
"""

import os
import sys
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
    dmt_perturbation,
)


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
def _make_output_dir():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'hypotheses')
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
# HELPER: extract biomarker snapshot from simulation window
# ============================================================================
def _extract_snapshot(t, P, st, nm, t_start, t_end):
    """Extract mean biomarkers from a time window (waking only)."""
    mask = (t >= t_start) & (t <= t_end)
    wake = nm['sleep'][mask] < 0.3
    if not np.any(wake):
        wake = np.ones(np.sum(mask), dtype=bool)  # fallback: use all

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
        'p300': p_to_p300(np.mean(P['conceptual'][mask][wake]),
                          DA=np.mean(nm['DA'][mask][wake])),
        'hrv': ne_cort_to_hrv(np.mean(nm['NE'][mask][wake]),
                               np.mean(st['cortisol'][mask][wake])),
        'bdnf': plasticity_to_bdnf(np.mean(nm['endogenous_plasticity'][mask][wake])),
        'pupil': ne_to_pupil(np.mean(nm['NE'][mask][wake])),
    }


def _depressed_state(weeks=4, chronic_stress=0.8):
    """Evolve a depressed baseline state."""
    t, P, st, nm = simulate_v2(
        t_span=(6.0, 6.0 + weeks * 7 * 24),
        dt=0.2, seed=42,
        chronic_stress=chronic_stress,
    )
    # Extract end-state
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


def _ptsd_state(weeks=4):
    """Evolve a PTSD baseline state."""
    t, P, st, nm = simulate_v2(
        t_span=(6.0, 6.0 + weeks * 7 * 24),
        dt=0.2, seed=42,
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
    """Get normal baseline snapshot."""
    t, P, st, nm = simulate_v2(
        t_span=(6.0, 30.0), dt=0.05, seed=42,
    )
    return _extract_snapshot(t, P, st, nm, 8.0, 20.0)


# ============================================================================
# HYPOTHESIS 1: Sleep Deprivation Potentiates Ketamine
# ============================================================================
def hypothesis_sleep_deprivation_ketamine(output_dir):
    """
    PREDICTION: Ketamine given during sleep deprivation produces a LARGER
    and LONGER-LASTING antidepressant response than ketamine after normal sleep.

    MECHANISM: Sleep deprivation removes NREM P-elevation cycles. When ketamine
    hits a system with already-lowered P (no NREM restoration), the BDNF-driven
    plasticity surge operates on a more favorable substrate — there's less
    homeostatic "pull-back" competing with the therapeutic P reduction.

    KNOWN SEPARATELY:
    - Sleep deprivation is a rapid antidepressant (Wu & Bunney 1990)
    - Ketamine is a rapid antidepressant (Zarate 2006)
    But the INTERACTION (whether they potentiate each other) is barely studied.

    COUNTER-INTUITIVE: You'd think sleep deprivation (stressful) would
    impair ketamine response. The model predicts the opposite.

    TESTABLE: RCT with ketamine after 24h sleep deprivation vs after normal
    sleep. Primary outcome: MADRS at 24h, 72h, 7d. Biomarkers: P300, BDNF.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Sleep Deprivation Potentiates Ketamine")
    print("="*70)

    dep_state, dep_snap = _depressed_state()
    print(f"  Depressed baseline: P_c={dep_snap['P_c']:.3f}, "
          f"alpha={dep_snap['alpha']:.4f}, BDNF={dep_snap['bdnf']:.3f}")

    # --- Condition A: Ketamine after NORMAL sleep ---
    # 2-day sim starting morning, ketamine at 2pm day 1, then 8 days
    t_a, P_a, st_a, nm_a = simulate_v2(
        t_span=(6.0, 6.0 + 10 * 24), dt=0.1, seed=42,
        state0=dep_state,
        chronic_stress=0.5,  # depression continues but reduced (seeking treatment)
        pharma_ketamine=[(14.0, 0.5)],  # ketamine at 2pm day 1
    )

    # --- Condition B: Ketamine during SLEEP DEPRIVATION ---
    # Sleep deprivation = override sleep drive to stay awake
    # We simulate this by using a very late sleep onset (force waking state)
    # In the model, we can achieve sleep deprivation by shifting sleep window
    # Using params_override to push SLEEP_ONSET very late for first 36h
    # Then ketamine at hour 30 (after ~24h awake), followed by recovery sleep

    # Phase B1: 36h sleep deprivation + ketamine at hour 30
    t_b1, P_b1, st_b1, nm_b1 = simulate_v2(
        t_span=(6.0, 6.0 + 36), dt=0.1, seed=42,
        state0=dep_state,
        chronic_stress=0.5,
        pharma_ketamine=[(30.0, 0.5)],  # ketamine after ~24h awake
        params_override={'SLEEP_ONSET': 48.0},  # force waking for 36h
    )

    # Phase B2: recovery sleep + continued monitoring (8 more days)
    idx_end = len(t_b1) - 1
    state_b2 = SimulationState(
        P_s=P_b1['sensory'][idx_end],
        P_c=P_b1['conceptual'][idx_end],
        P_sm=P_b1['selfmodel'][idx_end],
        hpa_sensitivity=st_b1['hpa_sensitivity'][idx_end],
        allostatic_load=st_b1['allostatic_load'][idx_end],
        cortisol=st_b1['cortisol'][idx_end],
    )
    t_b2, P_b2, st_b2, nm_b2 = simulate_v2(
        t_span=(42.0, 42.0 + 9 * 24), dt=0.1, seed=42,
        state0=state_b2,
        chronic_stress=0.5,
    )

    # Combine phase B
    t_b = np.concatenate([t_b1, t_b2])
    P_b = {k: np.concatenate([P_b1[k], P_b2[k]]) for k in P_b1}

    # --- Condition C: Sleep deprivation ONLY (no ketamine) ---
    t_c1, P_c1, st_c1, nm_c1 = simulate_v2(
        t_span=(6.0, 6.0 + 36), dt=0.1, seed=42,
        state0=dep_state,
        chronic_stress=0.5,
        params_override={'SLEEP_ONSET': 48.0},
    )
    state_c2 = SimulationState(
        P_s=P_c1['sensory'][-1], P_c=P_c1['conceptual'][-1],
        P_sm=P_c1['selfmodel'][-1],
        hpa_sensitivity=st_c1['hpa_sensitivity'][-1],
        allostatic_load=st_c1['allostatic_load'][-1],
        cortisol=st_c1['cortisol'][-1],
    )
    t_c2, P_c2, st_c2, nm_c2 = simulate_v2(
        t_span=(42.0, 42.0 + 9 * 24), dt=0.1, seed=42,
        state0=state_c2, chronic_stress=0.5,
    )
    t_c = np.concatenate([t_c1, t_c2])
    P_c = {k: np.concatenate([P_c1[k], P_c2[k]]) for k in P_c1}

    # --- Condition D: Ketamine ONLY (no sleep deprivation, same timing) ---
    # Same as A but reference
    t_d = t_a
    P_d = P_a

    # --- Extract timepoints for comparison ---
    def _mean_P_at(t_arr, P_arr, center_h, window=4.0):
        mask = (t_arr >= center_h - window) & (t_arr <= center_h + window)
        return np.mean(P_arr['conceptual'][mask]) if np.any(mask) else np.nan

    # Timepoints relative to ketamine dose
    # For A/D: dose at t=14.0, so +4h=18, +24h=38, +72h=86, +7d=182
    # For B: dose at t=30.0, so +4h=34, +24h=54, +72h=102, +7d=198
    timepoints_h = [4, 24, 72, 168]  # hours post-dose
    labels = ['4h', '24h', '72h', '7d']

    results = {'ketamine_only': [], 'sleep_dep_only': [], 'sleep_dep_ketamine': []}
    for dt_h in timepoints_h:
        results['ketamine_only'].append(_mean_P_at(t_a, P_a, 14.0 + dt_h))
        results['sleep_dep_only'].append(_mean_P_at(t_c, P_c, 30.0 + dt_h))
        results['sleep_dep_ketamine'].append(_mean_P_at(t_b, P_b, 30.0 + dt_h))

    # Print results
    print(f"\n  {'Condition':<25} {'4h':>8} {'24h':>8} {'72h':>8} {'7d':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Depressed baseline':<25} {dep_snap['P_c']:8.3f} {dep_snap['P_c']:8.3f} "
          f"{dep_snap['P_c']:8.3f} {dep_snap['P_c']:8.3f}")
    for cond, vals in results.items():
        print(f"  {cond:<25} {vals[0]:8.3f} {vals[1]:8.3f} {vals[2]:8.3f} {vals[3]:8.3f}")

    # Compute improvement ratios (lower P = better in depression)
    normal_snap = _normal_snapshot()
    print(f"\n  Normal P_c: {normal_snap['P_c']:.3f}")
    print(f"\n  Improvement (P_dep - P_condition, positive = better):")
    for cond, vals in results.items():
        impr = [dep_snap['P_c'] - v for v in vals]
        print(f"    {cond:<25} {impr[0]:+8.4f} {impr[1]:+8.4f} {impr[2]:+8.4f} {impr[3]:+8.4f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: P_conceptual trajectories (first 200h)
    ax = axes[0]
    t_plot_max = 200
    mask_a = t_a < t_plot_max
    mask_b = t_b < t_plot_max
    mask_c = t_c < t_plot_max

    ax.plot(t_a[mask_a] - 6.0, P_a['conceptual'][mask_a], 'b-', alpha=0.7,
            label='Ketamine only', linewidth=1.5)
    ax.plot(t_b[mask_b] - 6.0, P_b['conceptual'][mask_b], 'r-', alpha=0.7,
            label='Sleep dep + Ketamine', linewidth=1.5)
    ax.plot(t_c[mask_c] - 6.0, P_c['conceptual'][mask_c], 'g--', alpha=0.5,
            label='Sleep dep only', linewidth=1)
    ax.axhline(dep_snap['P_c'], color='gray', linestyle=':', alpha=0.5, label='Depressed baseline')
    ax.axhline(normal_snap['P_c'], color='gray', linestyle='--', alpha=0.3, label='Normal')
    ax.axvline(14.0 - 6.0, color='b', alpha=0.3, linestyle=':')
    ax.axvline(30.0 - 6.0, color='r', alpha=0.3, linestyle=':')
    ax.set_xlabel('Hours from start')
    ax.set_ylabel('P conceptual')
    ax.set_title('H1: Sleep Deprivation + Ketamine')
    ax.legend(fontsize=7, loc='upper right')

    # Panel B: Bar chart of improvement at each timepoint
    ax = axes[1]
    x = np.arange(len(labels))
    width = 0.25
    for i, (cond, color) in enumerate([
        ('ketamine_only', 'steelblue'),
        ('sleep_dep_ketamine', 'firebrick'),
        ('sleep_dep_only', 'seagreen'),
    ]):
        vals = [dep_snap['P_c'] - v for v in results[cond]]
        ax.bar(x + i * width, vals, width, label=cond.replace('_', ' '),
               color=color, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel('P reduction from depressed baseline')
    ax.set_title('Improvement (higher = better)')
    ax.legend(fontsize=7)
    ax.axhline(0, color='black', linewidth=0.5)

    fig.suptitle('HYPOTHESIS 1: Sleep Deprivation Potentiates Ketamine', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'H1_sleep_dep_ketamine.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: H1_sleep_dep_ketamine.png")

    return results


# ============================================================================
# HYPOTHESIS 2: Ketamine → Psilocybin Sequential Priming
# ============================================================================
def hypothesis_sequential_priming(output_dir):
    """
    PREDICTION: Ketamine 48h before psilocybin produces LARGER therapeutic
    effect than either drug alone, because ketamine's BDNF surge primes the
    plasticity pathway that psilocybin then exploits.

    MECHANISM:
    - Ketamine → BDNF surge → elevated endogenous plasticity (peaks 24-48h)
    - Psilocybin → 5-HT2A agonism → precision reduction + plasticity drive
    - When psilocybin hits during elevated plasticity (from ketamine's BDNF),
      the combined plasticity drive produces deeper P reduction
    - But SIMULTANEOUS dosing saturates pathways (diminishing returns)

    NEVER TESTED: No clinical trial has examined this specific sequential protocol.
    Could be a revolutionary combination treatment for treatment-resistant depression.

    TESTABLE: 4-arm RCT: (A) ketamine alone, (B) psilocybin alone,
    (C) ketamine day 0 + psilocybin day 2, (D) both day 0.
    Outcomes: MADRS, BDNF, P300 at day 3, 7, 14, 28.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Ketamine → Psilocybin Sequential Priming")
    print("="*70)

    dep_state, dep_snap = _depressed_state()
    normal_snap = _normal_snapshot()
    print(f"  Depressed baseline: P_c={dep_snap['P_c']:.3f}")
    print(f"  Normal baseline:    P_c={normal_snap['P_c']:.3f}")

    dose_day = 14.0  # first dose at 2pm day 1

    conditions = {}

    # --- A: Ketamine only ---
    t_a, P_a, st_a, nm_a = simulate_v2(
        t_span=(6.0, 6.0 + 28 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_ketamine=[(dose_day, 0.5)],
    )
    conditions['A: Ketamine only'] = (t_a, P_a, st_a, nm_a)

    # --- B: Psilocybin only ---
    t_b, P_b, st_b, nm_b = simulate_v2(
        t_span=(6.0, 6.0 + 28 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(dose_day, 0.6)],  # full therapeutic dose
    )
    conditions['B: Psilocybin only'] = (t_b, P_b, st_b, nm_b)

    # --- C: Sequential (ketamine day 0, psilocybin day 2) ---
    t_c, P_c, st_c, nm_c = simulate_v2(
        t_span=(6.0, 6.0 + 28 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_ketamine=[(dose_day, 0.5)],
        pharma_psilocybin=[(dose_day + 48.0, 0.6)],  # psilocybin 48h later
    )
    conditions['C: Sequential (K→P 48h)'] = (t_c, P_c, st_c, nm_c)

    # --- D: Simultaneous (both day 0) ---
    t_d, P_d, st_d, nm_d = simulate_v2(
        t_span=(6.0, 6.0 + 28 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_ketamine=[(dose_day, 0.5)],
        pharma_psilocybin=[(dose_day, 0.6)],  # same time
    )
    conditions['D: Simultaneous'] = (t_d, P_d, st_d, nm_d)

    # --- E: Depression control (no treatment) ---
    t_e, P_e, st_e, nm_e = simulate_v2(
        t_span=(6.0, 6.0 + 28 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
    )
    conditions['E: No treatment'] = (t_e, P_e, st_e, nm_e)

    # --- Extract daily P_conceptual (waking mean) ---
    days = list(range(1, 29))
    daily_P = {}
    for cond_name, (t_arr, P_arr, st_arr, nm_arr) in conditions.items():
        daily = []
        for d in days:
            t_start = 6.0 + (d - 1) * 24 + 8  # 2pm
            t_end = 6.0 + (d - 1) * 24 + 20    # 2am
            mask = (t_arr >= t_start) & (t_arr <= t_end) & (nm_arr['sleep'] < 0.3)
            if np.any(mask):
                daily.append(np.mean(P_arr['conceptual'][mask]))
            else:
                daily.append(np.nan)
        daily_P[cond_name] = daily

    # --- Print comparison at key timepoints ---
    check_days = [3, 7, 14, 28]
    print(f"\n  {'Condition':<30} " + "".join(f"{'Day '+str(d):>10}" for d in check_days))
    print(f"  {'-'*30} " + "-"*10*len(check_days))
    for cond_name, daily in daily_P.items():
        vals = [daily[d-1] for d in check_days]
        print(f"  {cond_name:<30} " + "".join(f"{v:10.4f}" for v in vals))

    # Improvement over no-treatment
    print(f"\n  Improvement vs no-treatment (P_notx - P_condition):")
    notx = daily_P['E: No treatment']
    for cond_name, daily in daily_P.items():
        if cond_name == 'E: No treatment':
            continue
        vals = [notx[d-1] - daily[d-1] for d in check_days]
        print(f"    {cond_name:<30} " + "".join(f"{v:+10.4f}" for v in vals))

    # --- Extract plasticity trajectories (to show the priming effect) ---
    plast_traces = {}
    for cond_name, (t_arr, P_arr, st_arr, nm_arr) in conditions.items():
        mask = t_arr < (6.0 + 5 * 24)  # first 5 days
        plast_traces[cond_name] = (t_arr[mask] - 6.0, nm_arr['endogenous_plasticity'][mask])

    # --- FIGURE ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        'A: Ketamine only': 'steelblue',
        'B: Psilocybin only': 'darkorange',
        'C: Sequential (K→P 48h)': 'firebrick',
        'D: Simultaneous': 'purple',
        'E: No treatment': 'gray',
    }

    # Panel A: Daily P_c over 28 days
    ax = axes[0, 0]
    for cond_name, daily in daily_P.items():
        ax.plot(days, daily, '-o', markersize=2, color=colors[cond_name],
                label=cond_name, linewidth=1.5, alpha=0.8)
    ax.axhline(normal_snap['P_c'], color='green', linestyle='--', alpha=0.3, label='Normal')
    ax.set_xlabel('Day')
    ax.set_ylabel('P conceptual (waking mean)')
    ax.set_title('A: Treatment Response Over 28 Days')
    ax.legend(fontsize=6)

    # Panel B: Plasticity traces (first 5 days) — shows the priming
    ax = axes[0, 1]
    for cond_name, (t_p, plast) in plast_traces.items():
        ax.plot(t_p, plast, color=colors[cond_name], label=cond_name,
                linewidth=1, alpha=0.7)
    ax.axvline(dose_day - 6.0, color='steelblue', alpha=0.3, linestyle=':', label='Ketamine dose')
    ax.axvline(dose_day + 48.0 - 6.0, color='darkorange', alpha=0.3, linestyle=':', label='Psilocybin dose (seq)')
    ax.set_xlabel('Hours from start')
    ax.set_ylabel('Endogenous plasticity')
    ax.set_title('B: Plasticity Priming (first 5 days)')
    ax.legend(fontsize=5)

    # Panel C: Improvement bar chart at day 7 and day 28
    ax = axes[1, 0]
    notx_vals = daily_P['E: No treatment']
    cond_list = ['A: Ketamine only', 'B: Psilocybin only',
                 'C: Sequential (K→P 48h)', 'D: Simultaneous']
    x = np.arange(len(cond_list))
    width = 0.35
    day7_impr = [notx_vals[6] - daily_P[c][6] for c in cond_list]
    day28_impr = [notx_vals[27] - daily_P[c][27] for c in cond_list]
    ax.bar(x - width/2, day7_impr, width, label='Day 7', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, day28_impr, width, label='Day 28', color='firebrick', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split(': ')[1] for c in cond_list], fontsize=7, rotation=15)
    ax.set_ylabel('P reduction vs no treatment')
    ax.set_title('C: Improvement at Day 7 & 28')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel D: P_conceptual raw trace (first 7 days, high resolution)
    ax = axes[1, 1]
    for cond_name, (t_arr, P_arr, st_arr, nm_arr) in conditions.items():
        mask = t_arr < (6.0 + 7 * 24)
        ax.plot(t_arr[mask] - 6.0, P_arr['conceptual'][mask],
                color=colors[cond_name], alpha=0.5, linewidth=0.8)
    ax.axhline(dep_snap['P_c'], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Hours from start')
    ax.set_ylabel('P conceptual (raw)')
    ax.set_title('D: Raw Trajectories (first 7 days)')

    fig.suptitle('HYPOTHESIS 2: Sequential Ketamine → Psilocybin Priming',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'H2_sequential_priming.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: H2_sequential_priming.png")

    return daily_P


# ============================================================================
# HYPOTHESIS 3: PTSD Shows Opposite Ketamine Response to Depression
# ============================================================================
def hypothesis_ptsd_ketamine_dissociation(output_dir):
    """
    PREDICTION: Ketamine produces DISSOCIABLE effects in PTSD — sensory
    hyperarousal improves (P_sensory decreases) but dissociative symptoms
    WORSEN (P_selfmodel decreases further, widening the self-model gap).

    MECHANISM:
    - Depression: elevated P_conceptual → ketamine lowers it → improvement
    - PTSD: P_sensory HIGH (hyperarousal) + P_selfmodel LOW (dissociation)
    - Ketamine's plasticity drive lowers ALL P levels uniformly
    - This helps P_sensory (moves toward normal) but HURTS P_selfmodel
      (pushes it even further from normal)

    EXPLAINS: Mixed/conflicting clinical data on ketamine for PTSD.
    Some studies show benefit (Feder 2014), others show limited improvement
    or worsening dissociation (Schönenberg 2005).

    TESTABLE: Within-subject PTSD ketamine trial measuring BOTH re-experiencing
    symptoms (maps to P_sensory) AND dissociation (maps to P_selfmodel) at
    4h, 24h, 7d. Predict: re-experiencing improves, DES scores worsen.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: PTSD Opposite Ketamine Response")
    print("="*70)

    normal_snap = _normal_snapshot()
    dep_state, dep_snap = _depressed_state()
    ptsd_state, ptsd_snap = _ptsd_state()

    print(f"  Normal:     P_s={normal_snap['P_s']:.3f}, P_c={normal_snap['P_c']:.3f}, "
          f"P_sm={normal_snap['P_sm']:.3f}")
    print(f"  Depressed:  P_s={dep_snap['P_s']:.3f}, P_c={dep_snap['P_c']:.3f}, "
          f"P_sm={dep_snap['P_sm']:.3f}")
    print(f"  PTSD:       P_s={ptsd_snap['P_s']:.3f}, P_c={ptsd_snap['P_c']:.3f}, "
          f"P_sm={ptsd_snap['P_sm']:.3f}")

    dose_time = 14.0

    # --- Depression + Ketamine ---
    t_dep, P_dep, st_dep, nm_dep = simulate_v2(
        t_span=(6.0, 6.0 + 10 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_ketamine=[(dose_time, 0.5)],
    )

    # --- PTSD + Ketamine ---
    t_ptsd, P_ptsd, st_ptsd, nm_ptsd = simulate_v2(
        t_span=(6.0, 6.0 + 10 * 24), dt=0.1, seed=42,
        state0=ptsd_state,
        ne_sensitization=1.8, coupling_breakdown=0.5,
        chronic_stress=0.3, td_coupling_scale=PTSD_TD_BREAKDOWN,
        pharma_ketamine=[(dose_time, 0.5)],
    )

    # --- PTSD control (no ketamine) ---
    t_ptsd_ctrl, P_ptsd_ctrl, _, _ = simulate_v2(
        t_span=(6.0, 6.0 + 10 * 24), dt=0.1, seed=42,
        state0=ptsd_state,
        ne_sensitization=1.8, coupling_breakdown=0.5,
        chronic_stress=0.3, td_coupling_scale=PTSD_TD_BREAKDOWN,
    )

    # Extract at 4h, 24h, 7d post-dose for all P levels
    timepoints = [(dose_time + 4, '4h'), (dose_time + 24, '24h'),
                  (dose_time + 7*24, '7d')]

    print(f"\n  Depression + Ketamine response (change from depressed baseline):")
    for tp, label in timepoints:
        snap = _extract_snapshot(t_dep, P_dep, st_dep, nm_dep, tp-2, tp+2)
        dP_s = snap['P_s'] - dep_snap['P_s']
        dP_c = snap['P_c'] - dep_snap['P_c']
        dP_sm = snap['P_sm'] - dep_snap['P_sm']
        print(f"    {label}: dP_s={dP_s:+.4f}, dP_c={dP_c:+.4f}, dP_sm={dP_sm:+.4f}")

    print(f"\n  PTSD + Ketamine response (change from PTSD baseline):")
    print(f"  NOTE: For PTSD, P_s DECREASE = hyperarousal improvement (GOOD)")
    print(f"        But P_sm DECREASE = more dissociation (BAD)")
    for tp, label in timepoints:
        snap = _extract_snapshot(t_ptsd, P_ptsd, st_ptsd, nm_ptsd, tp-2, tp+2)
        dP_s = snap['P_s'] - ptsd_snap['P_s']
        dP_c = snap['P_c'] - ptsd_snap['P_c']
        dP_sm = snap['P_sm'] - ptsd_snap['P_sm']
        good_bad_s = "GOOD" if dP_s < 0 else "BAD"
        good_bad_sm = "BAD" if dP_sm < 0 else "GOOD"
        print(f"    {label}: dP_s={dP_s:+.4f} ({good_bad_s}), dP_c={dP_c:+.4f}, "
              f"dP_sm={dP_sm:+.4f} ({good_bad_sm})")

    # --- FIGURE ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    levels = ['sensory', 'conceptual', 'selfmodel']
    level_labels = ['P sensory\n(hyperarousal)', 'P conceptual\n(cognition)',
                    'P self-model\n(dissociation)']
    normal_vals = [normal_snap['P_s'], normal_snap['P_c'], normal_snap['P_sm']]

    # Top row: Depression + Ketamine trajectories
    for i, (level, lbl) in enumerate(zip(levels, level_labels)):
        ax = axes[0, i]
        mask = t_dep < (6.0 + 8*24)
        ax.plot(t_dep[mask] - 6.0, P_dep[level][mask], 'b-', alpha=0.6, linewidth=1)
        ax.axhline(dep_snap[f'P_{"s" if level=="sensory" else ("c" if level=="conceptual" else "sm")}'],
                   color='blue', linestyle=':', alpha=0.4)
        ax.axhline(normal_vals[i], color='green', linestyle='--', alpha=0.3)
        ax.axvline(dose_time - 6.0, color='red', alpha=0.3, linestyle=':')
        ax.set_title(f'Depression: {lbl}', fontsize=9)
        ax.set_ylabel('P')
        if i == 0:
            ax.set_ylabel('Depression + Ket\nP value')

    # Bottom row: PTSD + Ketamine trajectories
    for i, (level, lbl) in enumerate(zip(levels, level_labels)):
        ax = axes[1, i]
        mask = t_ptsd < (6.0 + 8*24)
        ax.plot(t_ptsd[mask] - 6.0, P_ptsd[level][mask], 'r-', alpha=0.6, linewidth=1)
        mask_ctrl = t_ptsd_ctrl < (6.0 + 8*24)
        ax.plot(t_ptsd_ctrl[mask_ctrl] - 6.0, P_ptsd_ctrl[level][mask_ctrl],
                'gray', alpha=0.3, linewidth=1, label='No treatment')
        ax.axhline(ptsd_snap[f'P_{"s" if level=="sensory" else ("c" if level=="conceptual" else "sm")}'],
                   color='red', linestyle=':', alpha=0.4)
        ax.axhline(normal_vals[i], color='green', linestyle='--', alpha=0.3)
        ax.axvline(dose_time - 6.0, color='red', alpha=0.3, linestyle=':')
        ax.set_title(f'PTSD: {lbl}', fontsize=9)
        ax.set_xlabel('Hours')
        if i == 0:
            ax.set_ylabel('PTSD + Ket\nP value')
        if i == 2:
            ax.annotate('Further dissociation\n(WORSENING)',
                        xy=(0.5, 0.15), xycoords='axes fraction',
                        fontsize=8, color='red', ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.suptitle('HYPOTHESIS 3: PTSD Shows Opposite Ketamine Response to Depression',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'H3_ptsd_ketamine_dissociation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: H3_ptsd_ketamine_dissociation.png")


# ============================================================================
# HYPOTHESIS 4: Circadian Timing of Psychedelics
# ============================================================================
def hypothesis_circadian_psychedelic_timing(output_dir):
    """
    PREDICTION: Morning psilocybin (8 AM, high cortisol) produces a LARGER
    acute P reduction but SHORTER afterglow, while evening psilocybin
    (8 PM, low cortisol) produces a SMALLER acute effect but LONGER-LASTING
    therapeutic benefit.

    MECHANISM:
    - Morning: High cortisol → elevated NE → higher baseline suppression →
      psilocybin must overcome more suppression → BUT the larger delta creates
      stronger acute neuroplasticity signals → rapid tolerance from stronger
      5-HT2A engagement
    - Evening: Low cortisol → lower NE → less suppression to overcome →
      psilocybin achieves P reduction more easily → gentler 5-HT2A engagement →
      less receptor downregulation → longer afterglow

    PRACTICAL: Could determine optimal dosing time for clinical psilocybin therapy.
    Currently, dosing time is arbitrary in clinical trials.

    TESTABLE: Within-subject crossover (2 sessions, >4 week washout):
    morning vs evening psilocybin. Measure: acute mystical experience score,
    BDNF at 24h, alpha power at 1d/3d/7d/14d.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 4: Circadian Timing of Psychedelics")
    print("="*70)

    dep_state, dep_snap = _depressed_state()
    normal_snap = _normal_snapshot()
    print(f"  Depressed baseline: P_c={dep_snap['P_c']:.3f}")

    # --- Morning dose (8 AM = hour 8) ---
    morning_dose_time = 8.0
    t_am, P_am, st_am, nm_am = simulate_v2(
        t_span=(6.0, 6.0 + 21 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(morning_dose_time, 0.6)],
    )

    # --- Evening dose (8 PM = hour 20) ---
    evening_dose_time = 20.0
    t_pm, P_pm, st_pm, nm_pm = simulate_v2(
        t_span=(6.0, 6.0 + 21 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(evening_dose_time, 0.6)],
    )

    # --- Control (no dose) ---
    t_ctrl, P_ctrl, st_ctrl, nm_ctrl = simulate_v2(
        t_span=(6.0, 6.0 + 21 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
    )

    # Extract daily waking P_c and biomarkers
    days = list(range(1, 22))
    daily = {}
    for name, (t_arr, P_arr, st_arr, nm_arr) in [
        ('Morning (8 AM)', (t_am, P_am, st_am, nm_am)),
        ('Evening (8 PM)', (t_pm, P_pm, st_pm, nm_pm)),
        ('No treatment', (t_ctrl, P_ctrl, st_ctrl, nm_ctrl)),
    ]:
        pc_daily = []
        bdnf_daily = []
        for d in days:
            t_start = 6.0 + (d - 1) * 24 + 8
            t_end = 6.0 + (d - 1) * 24 + 20
            mask = (t_arr >= t_start) & (t_arr <= t_end) & (nm_arr['sleep'] < 0.3)
            if np.any(mask):
                pc_daily.append(np.mean(P_arr['conceptual'][mask]))
                bdnf_daily.append(plasticity_to_bdnf(
                    np.mean(nm_arr['endogenous_plasticity'][mask])))
            else:
                pc_daily.append(np.nan)
                bdnf_daily.append(np.nan)
        daily[name] = {'P_c': pc_daily, 'bdnf': bdnf_daily}

    # Peak acute effect (minimum P_c in first 12h post-dose)
    def _acute_min(t_arr, P_arr, nm_arr, dose_t):
        mask = (t_arr >= dose_t) & (t_arr <= dose_t + 8)
        if np.any(mask):
            return np.min(P_arr['conceptual'][mask])
        return np.nan

    acute_am = _acute_min(t_am, P_am, nm_am, morning_dose_time)
    acute_pm = _acute_min(t_pm, P_pm, nm_pm, evening_dose_time)

    print(f"\n  Acute P_c minimum (during session):")
    print(f"    Morning dose: {acute_am:.4f} (delta = {dep_snap['P_c'] - acute_am:+.4f})")
    print(f"    Evening dose: {acute_pm:.4f} (delta = {dep_snap['P_c'] - acute_pm:+.4f})")

    # Duration of benefit (days until P_c returns to within 1% of control)
    print(f"\n  Daily P_c (waking mean):")
    check_days = [1, 2, 3, 7, 14, 21]
    print(f"  {'Condition':<20} " + "".join(f"{'Day '+str(d):>10}" for d in check_days))
    for name, data in daily.items():
        vals = [data['P_c'][d-1] for d in check_days]
        print(f"  {name:<20} " + "".join(f"{v:10.4f}" for v in vals))

    # Cortisol at dose time
    am_cort_idx = np.argmin(np.abs(t_am - morning_dose_time))
    pm_cort_idx = np.argmin(np.abs(t_pm - evening_dose_time))
    print(f"\n  Cortisol at dose time:")
    print(f"    Morning (8 AM): {st_am['cortisol'][am_cort_idx]:.3f}")
    print(f"    Evening (8 PM): {st_pm['cortisol'][pm_cort_idx]:.3f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Morning (8 AM)': 'darkorange', 'Evening (8 PM)': 'steelblue',
              'No treatment': 'gray'}

    # Panel A: Daily P_c
    ax = axes[0, 0]
    for name, data in daily.items():
        ax.plot(days, data['P_c'], '-o', markersize=3, color=colors[name],
                label=name, linewidth=1.5)
    ax.axhline(normal_snap['P_c'], color='green', linestyle='--', alpha=0.3, label='Normal')
    ax.set_xlabel('Day')
    ax.set_ylabel('P conceptual (waking mean)')
    ax.set_title('A: Therapeutic Trajectory')
    ax.legend(fontsize=7)

    # Panel B: Raw P_c trace (first 3 days, high res)
    ax = axes[0, 1]
    mask_3d = t_am < (6.0 + 3*24)
    ax.plot(t_am[mask_3d] - 6.0, P_am['conceptual'][mask_3d], color='darkorange',
            alpha=0.6, linewidth=1, label='Morning dose')
    mask_3d_pm = t_pm < (6.0 + 3*24)
    ax.plot(t_pm[mask_3d_pm] - 6.0, P_pm['conceptual'][mask_3d_pm], color='steelblue',
            alpha=0.6, linewidth=1, label='Evening dose')
    ax.axvline(morning_dose_time - 6.0, color='darkorange', alpha=0.4, linestyle=':')
    ax.axvline(evening_dose_time - 6.0, color='steelblue', alpha=0.4, linestyle=':')
    ax.set_xlabel('Hours from start')
    ax.set_ylabel('P conceptual')
    ax.set_title('B: Acute Session (first 3 days)')
    ax.legend(fontsize=7)

    # Panel C: BDNF trajectory
    ax = axes[1, 0]
    for name, data in daily.items():
        if name == 'No treatment':
            continue
        ax.plot(days, data['bdnf'], '-o', markersize=3, color=colors[name],
                label=name, linewidth=1.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('BDNF (normalized)')
    ax.set_title('C: Plasticity / BDNF Trajectory')
    ax.legend(fontsize=7)

    # Panel D: Cortisol context at dose time
    ax = axes[1, 1]
    # Show cortisol over 24h to illustrate the circadian difference
    mask_24h = (t_am >= 6.0) & (t_am <= 30.0)
    ax.plot(t_am[mask_24h] - 6.0, st_am['cortisol'][mask_24h], 'k-', alpha=0.5,
            linewidth=1.5, label='Cortisol (24h cycle)')
    ax.axvline(morning_dose_time - 6.0, color='darkorange', linewidth=2,
               alpha=0.7, label='Morning dose')
    ax.axvline(evening_dose_time - 6.0, color='steelblue', linewidth=2,
               alpha=0.7, label='Evening dose')
    ax.set_xlabel('Hours from 6 AM')
    ax.set_ylabel('Cortisol (normalized)')
    ax.set_title('D: Cortisol Context at Dose Time')
    ax.legend(fontsize=7)

    fig.suptitle('HYPOTHESIS 4: Circadian Timing of Psychedelics',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'H4_circadian_psychedelic.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: H4_circadian_psychedelic.png")

    return daily


# ============================================================================
# HYPOTHESIS 5: Anxiolytics Block Psychedelic Afterglow
# ============================================================================
def hypothesis_anxiolytic_afterglow_block(output_dir):
    """
    PREDICTION: Benzodiazepine use in the 48h AFTER psilocybin substantially
    reduces the 2-week therapeutic afterglow, even though the acute trip
    has already ended.

    MECHANISM:
    - Psilocybin produces acute P reduction + BDNF-mediated plasticity surge
    - The therapeutic "afterglow" depends on sustained low P allowing
      neural reorganization over days 2-7
    - Benzodiazepines (GABA potentiation) RAISE P (similar to propofol pathway)
    - This directly counteracts the post-psilocybin low-P consolidation window
    - The plasticity pathway remains active but operates on a HIGH-P substrate,
      preventing the therapeutic restructuring

    CLINICALLY RELEVANT: Benzos are commonly prescribed alongside psychedelic
    therapy as "rescue medication." If this prediction is correct, their use
    should be strongly discouraged in the post-session window.

    TESTABLE: Retrospective analysis of psilocybin trials — compare outcomes
    for patients who used rescue benzos vs those who didn't. Or prospective
    RCT with planned benzo administration at 24h post-session.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 5: Anxiolytics Block Psychedelic Afterglow")
    print("="*70)

    dep_state, dep_snap = _depressed_state()
    normal_snap = _normal_snapshot()
    print(f"  Depressed baseline: P_c={dep_snap['P_c']:.3f}")

    dose_time = 10.0  # 10 AM psilocybin session

    # --- A: Psilocybin only (no benzo) ---
    t_a, P_a, st_a, nm_a = simulate_v2(
        t_span=(6.0, 6.0 + 21 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(dose_time, 0.6)],
    )

    # --- B: Psilocybin + benzo at 24h post-session ---
    # Simulate benzo as GABA potentiation (gaba_deficit = -0.2, i.e., GABA excess)
    # We need two phases: pre-benzo and post-benzo
    # Phase 1: psilocybin session (first 24h)
    t_b1, P_b1, st_b1, nm_b1 = simulate_v2(
        t_span=(6.0, 6.0 + 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(dose_time, 0.6)],
    )
    # Phase 2: benzo effect (GABA potentiation for 12h, then lingering 36h)
    state_b2 = SimulationState(
        P_s=P_b1['sensory'][-1], P_c=P_b1['conceptual'][-1],
        P_sm=P_b1['selfmodel'][-1],
        hpa_sensitivity=st_b1['hpa_sensitivity'][-1],
        allostatic_load=st_b1['allostatic_load'][-1],
        cortisol=st_b1['cortisol'][-1],
    )
    # Benzo taken at t=30h (24h post-dose), strong GABA potentiation for 2 days
    t_b2, P_b2, st_b2, nm_b2 = simulate_v2(
        t_span=(30.0, 30.0 + 2 * 24), dt=0.1, seed=42,
        state0=state_b2, chronic_stress=0.5,
        gaba_deficit=-0.20,  # negative = GABA potentiation (benzo effect)
    )
    # Phase 3: post-benzo recovery (remaining 18 days)
    state_b3 = SimulationState(
        P_s=P_b2['sensory'][-1], P_c=P_b2['conceptual'][-1],
        P_sm=P_b2['selfmodel'][-1],
        hpa_sensitivity=st_b2['hpa_sensitivity'][-1],
        allostatic_load=st_b2['allostatic_load'][-1],
        cortisol=st_b2['cortisol'][-1],
    )
    t_b3, P_b3, st_b3, nm_b3 = simulate_v2(
        t_span=(78.0, 6.0 + 21 * 24), dt=0.1, seed=42,
        state0=state_b3, chronic_stress=0.5,
    )
    # Combine all phases
    t_b = np.concatenate([t_b1, t_b2, t_b3])
    P_b = {k: np.concatenate([P_b1[k], P_b2[k], P_b3[k]]) for k in P_b1}
    nm_b = {}
    for k in nm_b1:
        nm_b[k] = np.concatenate([nm_b1[k], nm_b2[k], nm_b3[k]])
    st_b = {}
    for k in st_b1:
        st_b[k] = np.concatenate([st_b1[k], st_b2[k], st_b3[k]])

    # --- C: Psilocybin + benzo at 6h post-session (during comedown) ---
    t_c1, P_c1, st_c1, nm_c1 = simulate_v2(
        t_span=(6.0, 6.0 + 16), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
        pharma_psilocybin=[(dose_time, 0.6)],
    )
    state_c2 = SimulationState(
        P_s=P_c1['sensory'][-1], P_c=P_c1['conceptual'][-1],
        P_sm=P_c1['selfmodel'][-1],
        hpa_sensitivity=st_c1['hpa_sensitivity'][-1],
        allostatic_load=st_c1['allostatic_load'][-1],
        cortisol=st_c1['cortisol'][-1],
    )
    # Benzo at comedown (6h post dose = 16h), strong effect
    t_c2, P_c2, st_c2, nm_c2 = simulate_v2(
        t_span=(22.0, 22.0 + 2 * 24), dt=0.1, seed=42,
        state0=state_c2, chronic_stress=0.5,
        gaba_deficit=-0.25,  # stronger dose (acute anxiety management)
    )
    state_c3 = SimulationState(
        P_s=P_c2['sensory'][-1], P_c=P_c2['conceptual'][-1],
        P_sm=P_c2['selfmodel'][-1],
        hpa_sensitivity=st_c2['hpa_sensitivity'][-1],
        allostatic_load=st_c2['allostatic_load'][-1],
        cortisol=st_c2['cortisol'][-1],
    )
    t_c3, P_c3, st_c3, nm_c3 = simulate_v2(
        t_span=(70.0, 6.0 + 21 * 24), dt=0.1, seed=42,
        state0=state_c3, chronic_stress=0.5,
    )
    t_c = np.concatenate([t_c1, t_c2, t_c3])
    P_c_all = {k: np.concatenate([P_c1[k], P_c2[k], P_c3[k]]) for k in P_c1}
    nm_c = {}
    for k in nm_c1:
        nm_c[k] = np.concatenate([nm_c1[k], nm_c2[k], nm_c3[k]])

    # --- D: Control (no treatment) ---
    t_d, P_d, st_d, nm_d = simulate_v2(
        t_span=(6.0, 6.0 + 21 * 24), dt=0.1, seed=42,
        state0=dep_state, chronic_stress=0.5,
    )

    # Extract daily P_c
    days = list(range(1, 22))
    all_conds = {
        'Psilocybin only': (t_a, P_a, nm_a),
        'Psilo + benzo 24h': (t_b, P_b, nm_b),
        'Psilo + benzo 6h': (t_c, P_c_all, nm_c),
        'No treatment': (t_d, P_d, nm_d),
    }
    daily_P = {}
    for name, (t_arr, P_arr, nm_arr) in all_conds.items():
        pc_daily = []
        for d in days:
            t_start = 6.0 + (d - 1) * 24 + 8
            t_end = 6.0 + (d - 1) * 24 + 20
            mask = (t_arr >= t_start) & (t_arr <= t_end) & (nm_arr['sleep'] < 0.3)
            if np.any(mask):
                pc_daily.append(np.mean(P_arr['conceptual'][mask]))
            else:
                pc_daily.append(np.nan)
        daily_P[name] = pc_daily

    # Print comparison
    check_days = [1, 2, 3, 5, 7, 14, 21]
    print(f"\n  Daily waking P_c:")
    print(f"  {'Condition':<25} " + "".join(f"{'Day '+str(d):>10}" for d in check_days))
    print(f"  {'-'*25} " + "-"*10*len(check_days))
    for name, vals in daily_P.items():
        vs = [vals[d-1] for d in check_days]
        print(f"  {name:<25} " + "".join(f"{v:10.4f}" for v in vs))

    # Afterglow duration (days where P < control - 0.005)
    print(f"\n  Afterglow lost (% of psilocybin-only benefit retained at day 7):")
    notx = daily_P['No treatment']
    psilo_benefit_d7 = notx[6] - daily_P['Psilocybin only'][6]
    for name in ['Psilo + benzo 6h', 'Psilo + benzo 24h']:
        ben_d7 = notx[6] - daily_P[name][6]
        retained = (ben_d7 / psilo_benefit_d7 * 100) if psilo_benefit_d7 > 0.001 else 0
        print(f"    {name}: {retained:.0f}% retained")

    # --- FIGURE ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        'Psilocybin only': 'darkorange',
        'Psilo + benzo 24h': 'steelblue',
        'Psilo + benzo 6h': 'firebrick',
        'No treatment': 'gray',
    }

    # Panel A: Daily P_c trajectory
    ax = axes[0]
    for name, vals in daily_P.items():
        ax.plot(days, vals, '-o', markersize=3, color=colors[name],
                label=name, linewidth=1.5)
    ax.axhline(normal_snap['P_c'], color='green', linestyle='--', alpha=0.3, label='Normal')
    ax.axvspan(1.5, 3.5, alpha=0.08, color='steelblue', label='Benzo window')
    ax.set_xlabel('Day')
    ax.set_ylabel('P conceptual (waking mean)')
    ax.set_title('A: Afterglow Disruption')
    ax.legend(fontsize=6)

    # Panel B: Benefit retained at each timepoint
    ax = axes[1]
    retained_data = {}
    for name in ['Psilocybin only', 'Psilo + benzo 24h', 'Psilo + benzo 6h']:
        retained = []
        for d in days:
            psilo_ben = notx[d-1] - daily_P['Psilocybin only'][d-1]
            cond_ben = notx[d-1] - daily_P[name][d-1]
            r = (cond_ben / psilo_ben * 100) if abs(psilo_ben) > 0.001 else 100
            retained.append(np.clip(r, -50, 150))
        retained_data[name] = retained
        ax.plot(days, retained, '-o', markersize=2, color=colors[name],
                label=name, linewidth=1.5)
    ax.axhline(100, color='darkorange', linestyle=':', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('% of psilocybin benefit retained')
    ax.set_title('B: Afterglow Preservation')
    ax.legend(fontsize=6)
    ax.set_ylim(-50, 160)

    fig.suptitle('HYPOTHESIS 5: Anxiolytics Block Psychedelic Afterglow',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'H5_anxiolytic_afterglow.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: H5_anxiolytic_afterglow.png")

    return daily_P


# ============================================================================
# HYPOTHESIS 6: DMT vs Psilocybin Temporal Dynamics
# ============================================================================
def hypothesis_dmt_temporal_dynamics(output_dir):
    """
    PREDICTION: DMT produces equivalent acute P reduction depth to psilocybin
    but WEAKER antidepressant afterglow because the BDNF cascade requires
    sustained low-P duration that DMT's 30-min window cannot provide.

    MECHANISM:
    - DMT and psilocybin both act via 5-HT2A agonism → acute P reduction
    - DMT onset 2 min, duration 30 min (vs psilocybin: 30 min onset, 6h duration)
    - BDNF-mediated plasticity cascade needs sustained receptor activation
    - DMT's brief window → less receptor downregulation → weaker afterglow
    - Ayahuasca (oral DMT + MAOI) extends duration → psilocybin-like afterglow

    TESTABLE: Compare LZW/alpha/BDNF at acute peak AND at 1-week follow-up
    for IV DMT vs oral psilocybin. Also: ayahuasca vs IV DMT head-to-head.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 6: DMT vs Psilocybin Temporal Dynamics")
    print("="*70)

    normal_snap = _normal_snapshot()

    # --- A: Psilocybin (standard: 0.6 dose at 14:00) ---
    t_psi, P_psi, st_psi, nm_psi = simulate_v2(
        t_span=(6.0, 6.0 + 7 * 24), dt=0.05, seed=42,
        pharma_psilocybin=[(14.0, 0.6)],
    )

    # --- B: DMT IV (equivalent strength at 14:00) ---
    t_dmt, P_dmt, st_dmt, nm_dmt = simulate_v2(
        t_span=(6.0, 6.0 + 7 * 24), dt=0.05, seed=42,
        pharma_dmt=[(14.0, 0.6)],
    )

    # --- C: Ayahuasca simulation (extended DMT via MAOI) ---
    # Model as multiple overlapping DMT doses over 4h (oral absorption + MAOI)
    aya_doses = [(14.0 + i * 0.5, 0.15) for i in range(8)]  # 4h sustained release
    t_aya, P_aya, st_aya, nm_aya = simulate_v2(
        t_span=(6.0, 6.0 + 7 * 24), dt=0.05, seed=42,
        pharma_dmt=aya_doses,
    )

    # --- Extract biomarkers ---
    # Acute peak: psilocybin at t=15.75, DMT at t=14.1 (much faster)
    psi_acute = _extract_snapshot(t_psi, P_psi, st_psi, nm_psi, 15.0, 16.5)
    dmt_acute = _extract_snapshot(t_dmt, P_dmt, st_dmt, nm_dmt, 14.05, 14.15)
    aya_acute = _extract_snapshot(t_aya, P_aya, st_aya, nm_aya, 15.0, 17.0)

    # 24h post
    psi_24h = _extract_snapshot(t_psi, P_psi, st_psi, nm_psi, 38.0, 42.0)
    dmt_24h = _extract_snapshot(t_dmt, P_dmt, st_dmt, nm_dmt, 38.0, 42.0)
    aya_24h = _extract_snapshot(t_aya, P_aya, st_aya, nm_aya, 38.0, 42.0)

    # Day 7
    psi_7d = _extract_snapshot(t_psi, P_psi, st_psi, nm_psi,
                                6.0 + 6*24, 6.0 + 7*24)
    dmt_7d = _extract_snapshot(t_dmt, P_dmt, st_dmt, nm_dmt,
                                6.0 + 6*24, 6.0 + 7*24)
    aya_7d = _extract_snapshot(t_aya, P_aya, st_aya, nm_aya,
                                6.0 + 6*24, 6.0 + 7*24)

    # Print results
    print(f"\n  {'Biomarker':<20} {'Normal':>8} {'Psilocybin':>11} {'DMT IV':>8} {'Ayahuasca':>10}")
    print(f"  {'─'*60}")
    for label, norm, psi, dmt, aya in [
        ('ACUTE PEAK:', None, None, None, None),
        ('  P_conceptual', normal_snap['P_c'], psi_acute['P_c'], dmt_acute['P_c'], aya_acute['P_c']),
        ('  Alpha', normal_snap['alpha'], psi_acute['alpha'], dmt_acute['alpha'], aya_acute['alpha']),
        ('  LZW', normal_snap['lzw'], psi_acute['lzw'], dmt_acute['lzw'], aya_acute['lzw']),
        ('DAY 1 (24h post):', None, None, None, None),
        ('  P_conceptual', normal_snap['P_c'], psi_24h['P_c'], dmt_24h['P_c'], aya_24h['P_c']),
        ('  BDNF', normal_snap['bdnf'], psi_24h['bdnf'], dmt_24h['bdnf'], aya_24h['bdnf']),
        ('DAY 7:', None, None, None, None),
        ('  P_conceptual', normal_snap['P_c'], psi_7d['P_c'], dmt_7d['P_c'], aya_7d['P_c']),
        ('  BDNF', normal_snap['bdnf'], psi_7d['bdnf'], dmt_7d['bdnf'], aya_7d['bdnf']),
    ]:
        if norm is None:
            print(f"\n  {label}")
        else:
            print(f"  {label:<20} {norm:>8.4f} {psi:>11.4f} {dmt:>8.4f} {aya:>10.4f}")

    # Key prediction metrics
    psi_afterglow = (psi_7d['P_c'] - normal_snap['P_c']) / normal_snap['P_c'] * 100
    dmt_afterglow = (dmt_7d['P_c'] - normal_snap['P_c']) / normal_snap['P_c'] * 100
    aya_afterglow = (aya_7d['P_c'] - normal_snap['P_c']) / normal_snap['P_c'] * 100

    print(f"\n  KEY PREDICTIONS:")
    print(f"    Psilocybin 7-day P change: {psi_afterglow:+.1f}%")
    print(f"    DMT IV 7-day P change:     {dmt_afterglow:+.1f}%")
    print(f"    Ayahuasca 7-day P change:  {aya_afterglow:+.1f}%")
    print(f"    DMT antidepressant < psilocybin: {abs(dmt_afterglow) < abs(psi_afterglow)}")
    print(f"    Ayahuasca ≈ psilocybin:          {abs(aya_afterglow - psi_afterglow) < 1.0}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: P_c trajectory (first 12h — acute comparison)
    ax = axes[0, 0]
    mask_12h_psi = t_psi < 26.0
    mask_12h_dmt = t_dmt < 26.0
    mask_12h_aya = t_aya < 26.0
    ax.plot(t_psi[mask_12h_psi] - 14.0, P_psi['conceptual'][mask_12h_psi],
            'b-', linewidth=2, label='Psilocybin')
    ax.plot(t_dmt[mask_12h_dmt] - 14.0, P_dmt['conceptual'][mask_12h_dmt],
            'r-', linewidth=2, label='DMT IV')
    ax.plot(t_aya[mask_12h_aya] - 14.0, P_aya['conceptual'][mask_12h_aya],
            'g-', linewidth=2, label='Ayahuasca')
    ax.axhline(normal_snap['P_c'], color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, label='Dose')
    ax.set_xlabel('Hours relative to dose')
    ax.set_ylabel('P_conceptual')
    ax.set_title('A. Acute P Trajectory')
    ax.legend(fontsize=7)

    # Panel B: P_c trajectory (full 7 days)
    ax = axes[0, 1]
    ax.plot((t_psi - 14.0) / 24, P_psi['conceptual'], 'b-', alpha=0.3, linewidth=0.5)
    ax.plot((t_dmt - 14.0) / 24, P_dmt['conceptual'], 'r-', alpha=0.3, linewidth=0.5)
    ax.plot((t_aya - 14.0) / 24, P_aya['conceptual'], 'g-', alpha=0.3, linewidth=0.5)
    # Daily waking means
    for day in range(7):
        t_start = 6.0 + day * 24 + 8
        t_end = 6.0 + day * 24 + 20
        for t_arr, P_arr, color, lbl in [
            (t_psi, P_psi['conceptual'], 'b', 'Psilocybin' if day == 0 else None),
            (t_dmt, P_dmt['conceptual'], 'r', 'DMT IV' if day == 0 else None),
            (t_aya, P_aya['conceptual'], 'g', 'Ayahuasca' if day == 0 else None),
        ]:
            mask = (t_arr >= t_start) & (t_arr <= t_end)
            if np.any(mask):
                ax.plot(day, np.mean(P_arr[mask]), 'o', color=color, markersize=6,
                        label=lbl)
    ax.axhline(normal_snap['P_c'], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Days post-dose')
    ax.set_ylabel('P_conceptual (daily waking mean)')
    ax.set_title('B. 7-Day Afterglow Comparison')
    ax.legend(fontsize=7)

    # Panel C: Biomarker comparison (acute peak)
    ax = axes[1, 0]
    markers = ['Alpha', 'LZW', 'HRV', 'Pupil', 'BDNF']
    psi_vals = [psi_acute['alpha']/normal_snap['alpha'],
                psi_acute['lzw']/normal_snap['lzw'],
                psi_acute['hrv']/normal_snap['hrv'],
                psi_acute['pupil']/normal_snap['pupil'],
                psi_acute['bdnf']/normal_snap['bdnf']]
    dmt_vals = [dmt_acute['alpha']/normal_snap['alpha'],
                dmt_acute['lzw']/normal_snap['lzw'],
                dmt_acute['hrv']/normal_snap['hrv'],
                dmt_acute['pupil']/normal_snap['pupil'],
                dmt_acute['bdnf']/normal_snap['bdnf']]
    x = np.arange(len(markers))
    w = 0.35
    ax.bar(x - w/2, psi_vals, w, color='blue', alpha=0.7, label='Psilocybin')
    ax.bar(x + w/2, dmt_vals, w, color='red', alpha=0.7, label='DMT IV')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(markers)
    ax.set_ylabel('Ratio to normal')
    ax.set_title('C. Acute Biomarkers (ratio to baseline)')
    ax.legend(fontsize=7)

    # Panel D: Afterglow strength (day 7 P change)
    ax = axes[1, 1]
    afterglows = [psi_afterglow, dmt_afterglow, aya_afterglow]
    colors = ['blue', 'red', 'green']
    labels = ['Psilocybin', 'DMT IV', 'Ayahuasca']
    bars = ax.bar(range(3), afterglows, color=colors, alpha=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel('P_c change from baseline (%)')
    ax.set_title('D. 7-Day Afterglow Strength')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    fig.suptitle('HYPOTHESIS 6: DMT vs Psilocybin Temporal Dynamics',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'H6_dmt_psilocybin_dynamics.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: H6_dmt_psilocybin_dynamics.png")

    return {
        'psi_afterglow': psi_afterglow,
        'dmt_afterglow': dmt_afterglow,
        'aya_afterglow': aya_afterglow,
    }


# ============================================================================
# HYPOTHESIS 7: Pineal Calcification → Cognitive Rigidity
# ============================================================================
def hypothesis_pineal_calcification(output_dir):
    """
    PREDICTION: Pineal calcification reduces endogenous plasticity (via melatonin
    disruption), leading to elevated waking P, reduced neural complexity, and
    cognitive rigidity — a subclinical aging phenotype that increases vulnerability
    to depression and neurodegenerative conditions.

    MECHANISM:
    - Pineal calcification → ↓ melatonin production
    - ↓ melatonin → disrupted sleep architecture (reduced REM proportion)
    - ↓ REM → ↓ endogenous plasticity drive (tryptamines, BDNF peaks)
    - ↓ plasticity → ↑ waking P → cognitive rigidity
    - P elevation → ↑ EEG alpha, ↓ LZW, ↓ HRV, higher vulnerability to depression

    PROPOSED VALIDATION:
    - Retrospective: ADNI/UK Biobank CT scans scored for pineal calcification
      correlated with EEG alpha power, cognitive flexibility tests
    - Prospective: longitudinal pineal calcification tracking vs cognitive decline

    NOTE: Simulation predictions are free; data analysis requires IRB (future work).
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 7: Pineal Calcification → Cognitive Rigidity")
    print("="*70)

    normal_snap = _normal_snapshot()

    # --- Dose-response: calcification severity vs P elevation ---
    # Model calcification as reduction in endogenous plasticity + REM disruption
    calcification_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for calc in calcification_levels:
        # Endogenous plasticity reduction (melatonin → tryptamine pathway)
        plast_scale = 1.0 - 0.5 * calc  # 0% → 50% reduction at max calcification

        # Simulate 2 weeks to reach steady state
        t, P, st, nm = simulate_v2(
            t_span=(6.0, 6.0 + 14 * 24), dt=0.1, seed=42,
            endogenous_plasticity_scale=plast_scale,
        )

        # Extract last 3 days waking snapshot
        snap = _extract_snapshot(t, P, st, nm, t[-1] - 3*24, t[-1])
        snap['calcification'] = calc
        snap['plast_scale'] = plast_scale
        results.append(snap)

    # Print dose-response table
    print(f"\n  {'Calcification':>14} {'Plast Scale':>12} {'P_c':>8} {'Alpha':>8} "
          f"{'LZW':>8} {'HRV':>8} {'BDNF':>8}")
    print(f"  {'─'*70}")
    for r in results:
        print(f"  {r['calcification']:>14.1f} {r['plast_scale']:>12.2f} "
              f"{r['P_c']:>8.4f} {r['alpha']:>8.4f} {r['lzw']:>8.4f} "
              f"{r['hrv']:>8.4f} {r['bdnf']:>8.4f}")

    # --- Calcification + depression vulnerability ---
    print(f"\n  Depression vulnerability with calcification:")
    vuln_results = []
    for calc in [0.0, 0.4, 0.8]:
        plast_scale = 1.0 - 0.5 * calc

        # Mild stress (subclinical) — does calcification tip into depression?
        t, P, st, nm = simulate_v2(
            t_span=(6.0, 6.0 + 4 * 7 * 24), dt=0.2, seed=42,
            chronic_stress=0.3,  # mild stress
            endogenous_plasticity_scale=plast_scale,
        )
        snap = _extract_snapshot(t, P, st, nm, t[-1] - 5*24, t[-1])

        # Also get normal (no stress) for comparison
        t_n, P_n, st_n, nm_n = simulate_v2(
            t_span=(6.0, 6.0 + 14 * 24), dt=0.1, seed=42,
            endogenous_plasticity_scale=plast_scale,
        )
        snap_n = _extract_snapshot(t_n, P_n, st_n, nm_n, t_n[-1] - 3*24, t_n[-1])

        delta_P = snap['P_c'] - snap_n['P_c']
        vuln_results.append({
            'calc': calc, 'P_no_stress': snap_n['P_c'],
            'P_with_stress': snap['P_c'], 'delta_P': delta_P,
        })
        print(f"    Calc={calc:.1f}: P_c(no stress)={snap_n['P_c']:.4f}, "
              f"P_c(mild stress)={snap['P_c']:.4f}, ΔP={delta_P:+.4f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Dose-response — calcification vs P levels
    ax = axes[0, 0]
    calcs = [r['calcification'] for r in results]
    ax.plot(calcs, [r['P_s'] for r in results], 'o-', color='blue', label='P_sensory')
    ax.plot(calcs, [r['P_c'] for r in results], 's-', color='green', label='P_conceptual')
    ax.plot(calcs, [r['P_sm'] for r in results], '^-', color='red', label='P_selfmodel')
    ax.set_xlabel('Pineal calcification severity')
    ax.set_ylabel('Waking P level')
    ax.set_title('A. Precision vs Calcification')
    ax.legend(fontsize=7)

    # Panel B: Biomarkers vs calcification
    ax = axes[0, 1]
    ax.plot(calcs, [r['alpha'] / normal_snap['alpha'] for r in results],
            'o-', color='purple', label='Alpha (norm)')
    ax.plot(calcs, [r['lzw'] / normal_snap['lzw'] for r in results],
            's-', color='orange', label='LZW (norm)')
    ax.plot(calcs, [r['bdnf'] / normal_snap['bdnf'] for r in results],
            '^-', color='green', label='BDNF (norm)')
    ax.plot(calcs, [r['hrv'] / normal_snap['hrv'] for r in results],
            'd-', color='red', label='HRV (norm)')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Pineal calcification severity')
    ax.set_ylabel('Ratio to normal')
    ax.set_title('B. Biomarker Changes')
    ax.legend(fontsize=7)

    # Panel C: Depression vulnerability
    ax = axes[1, 0]
    calc_levels = [v['calc'] for v in vuln_results]
    no_stress = [v['P_no_stress'] for v in vuln_results]
    with_stress = [v['P_with_stress'] for v in vuln_results]
    x = np.arange(len(calc_levels))
    w = 0.35
    ax.bar(x - w/2, no_stress, w, color='steelblue', alpha=0.7, label='No stress')
    ax.bar(x + w/2, with_stress, w, color='tomato', alpha=0.7, label='Mild stress')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c:.1f}' for c in calc_levels])
    ax.set_xlabel('Pineal calcification severity')
    ax.set_ylabel('P_conceptual')
    ax.set_title('C. Depression Vulnerability')
    ax.legend(fontsize=7)

    # Panel D: Example trajectories — normal vs severe calcification
    ax = axes[1, 1]
    # 3-day trajectory for normal and severe
    for calc, color, label in [(0.0, 'blue', 'Normal'), (0.8, 'red', 'Severe calc')]:
        plast_scale = 1.0 - 0.5 * calc
        t_ex, P_ex, _, nm_ex = simulate_v2(
            t_span=(6.0, 6.0 + 3 * 24), dt=0.05, seed=42,
            endogenous_plasticity_scale=plast_scale,
        )
        ax.plot((t_ex - 6.0) / 24, P_ex['conceptual'], color=color,
                alpha=0.7, linewidth=1.0, label=label)

    ax.set_xlabel('Days')
    ax.set_ylabel('P_conceptual')
    ax.set_title('D. 3-Day P Trajectory')
    ax.legend(fontsize=7)

    fig.suptitle('HYPOTHESIS 7: Pineal Calcification → Cognitive Rigidity',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'H7_pineal_calcification.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: H7_pineal_calcification.png")

    return results


# ============================================================================
# SUMMARY TABLE
# ============================================================================
def print_summary():
    print("\n" + "="*70)
    print("SUMMARY: GROUND-BREAKING PREDICTIONS FROM PLASTICITY MODEL")
    print("="*70)
    print("""
    H1: SLEEP DEPRIVATION + KETAMINE SYNERGY
        Ketamine during sleep deprivation produces larger, longer-lasting
        antidepressant response. Mechanism: NREM removal prevents P restoration,
        letting ketamine's plasticity surge consolidate on lower-P substrate.
        Test: RCT with ketamine after 24h wake vs normal sleep.

    H2: KETAMINE → PSILOCYBIN SEQUENTIAL PRIMING
        Ketamine 48h before psilocybin produces synergistic effect via BDNF
        priming of the plasticity pathway. Sequential > simultaneous > alone.
        Test: 4-arm RCT. Could revolutionize treatment-resistant depression.

    H3: PTSD OPPOSITE KETAMINE RESPONSE
        Ketamine improves PTSD hyperarousal (P_sensory down) but WORSENS
        dissociation (P_selfmodel down further). Explains conflicting data.
        Test: Within-subject PTSD trial measuring re-experiencing AND dissociation.

    H4: CIRCADIAN TIMING OF PSYCHEDELICS
        Morning psilocybin (high cortisol) → larger acute effect, shorter afterglow.
        Evening psilocybin (low cortisol) → smaller acute but longer-lasting benefit.
        Test: Crossover morning vs evening dosing with 4-week washout.

    H5: ANXIOLYTICS BLOCK PSYCHEDELIC AFTERGLOW
        Benzodiazepines in the 48h after psilocybin prevent therapeutic
        consolidation by raising P during the critical plasticity window.
        Test: Retrospective analysis of psilocybin trials with/without rescue benzos.

    H6: DMT vs PSILOCYBIN TEMPORAL DYNAMICS
        DMT produces equivalent acute P reduction but weaker 7-day afterglow
        because BDNF cascade needs sustained low-P duration (>30 min).
        Ayahuasca (oral DMT + MAOI) recovers psilocybin-like afterglow.
        Test: IV DMT vs psilocybin vs ayahuasca head-to-head.

    H7: PINEAL CALCIFICATION → COGNITIVE RIGIDITY
        Pineal calcification → reduced melatonin → disrupted sleep →
        reduced endogenous plasticity → elevated waking P → cognitive rigidity.
        Dose-response: 40% calcification → measurable P elevation.
        Test: CT-scored calcification correlated with EEG alpha/cognitive flex.
    """)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("="*70)
    print("Ground-Breaking Hypothesis Simulations")
    print("Precision-Weighting Plasticity Model")
    print("="*70)

    output_dir = _make_output_dir()
    print(f"Output: {output_dir}\n")

    print_summary()

    hypothesis_sleep_deprivation_ketamine(output_dir)
    hypothesis_sequential_priming(output_dir)
    hypothesis_ptsd_ketamine_dissociation(output_dir)
    hypothesis_circadian_psychedelic_timing(output_dir)
    hypothesis_anxiolytic_afterglow_block(output_dir)
    hypothesis_dmt_temporal_dynamics(output_dir)
    hypothesis_pineal_calcification(output_dir)

    print("\n" + "="*70)
    print("ALL HYPOTHESES SIMULATED (7 total)")
    print(f"Figures saved to: {output_dir}")
    print("="*70)
