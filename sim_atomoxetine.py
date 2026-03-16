"""
Atomoxetine Validation Simulations
====================================

Out-of-sample predictions: atomoxetine (selective NET inhibitor) should produce
OPPOSITE effects to psilocybin on precision-derived biomarkers. No fitting was
performed on atomoxetine data — predictions are pure out-of-sample.

Simulations:
  1. Acute single dose in healthy subject → biomarker predictions
  2. Chronic 4-6 weeks in ADHD → does it normalize P variability?
  3. Head-to-head atomoxetine vs psilocybin on all biomarkers
  4. Atomoxetine in depression → different P trajectory than SSRIs/ketamine?

Run:
    python3 sim_atomoxetine.py
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
    atomoxetine_perturbation,
)


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
def _make_output_dir():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'atomoxetine')
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


# ============================================================================
# SIM 1: Acute single dose in healthy subject
# ============================================================================
def sim_acute_single_dose(output_dir):
    """
    Acute atomoxetine in a healthy subject. Single dose at 2 PM.
    Predictions: increased NE → increased P, alpha power, pupil dilation.
    Opposite to psilocybin on most biomarkers.
    """
    print("\n" + "="*70)
    print("SIM 1: Acute Atomoxetine — Single Dose in Healthy Subject")
    print("="*70)

    # --- Baseline: normal 24h ---
    t_base, P_base, st_base, nm_base = simulate_v2(
        t_span=(6.0, 30.0), dt=0.05, seed=42,
    )

    # --- With atomoxetine at 2 PM (t=14.0) ---
    t_atx, P_atx, st_atx, nm_atx = simulate_v2(
        t_span=(6.0, 30.0), dt=0.05, seed=42,
        pharma_atomoxetine=[(14.0, 0.5)],
    )

    # Extract snapshots at peak (t=15.5, ~1.5h post-dose) and 6h post (t=20.0)
    snap_base_peak = _extract_snapshot(t_base, P_base, st_base, nm_base, 15.0, 16.0)
    snap_atx_peak = _extract_snapshot(t_atx, P_atx, st_atx, nm_atx, 15.0, 16.0)
    snap_base_6h = _extract_snapshot(t_base, P_base, st_base, nm_base, 19.5, 20.5)
    snap_atx_6h = _extract_snapshot(t_atx, P_atx, st_atx, nm_atx, 19.5, 20.5)

    # Print biomarker comparison
    biomarkers = ['alpha', 'lzw', 'hrv', 'pupil', 'bdnf', 'P_s', 'P_c', 'P_sm', 'NE']
    print(f"\n  {'Biomarker':<12} {'Baseline':>10} {'ATX peak':>10} {'delta':>10} "
          f"{'ATX 6h':>10} {'delta':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for bm in biomarkers:
        b = snap_base_peak[bm]
        ap = snap_atx_peak[bm]
        a6 = snap_atx_6h[bm]
        b6 = snap_base_6h[bm]
        print(f"  {bm:<12} {b:10.4f} {ap:10.4f} {ap-b:+10.4f} "
              f"{a6:10.4f} {a6-b6:+10.4f}")

    # --- FIGURE: 3-panel ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel (a): P trajectories for 3 levels
    ax = axes[0]
    levels = ['sensory', 'conceptual', 'selfmodel']
    level_labels = ['Sensory', 'Conceptual', 'Self-model']
    colors_lvl = ['#e74c3c', '#3498db', '#2ecc71']
    for j, (level, lbl, clr) in enumerate(zip(levels, level_labels, colors_lvl)):
        ax.plot(t_base - 6.0, P_base[level], '--', color=clr, alpha=0.4,
                linewidth=1, label=f'{lbl} (baseline)')
        ax.plot(t_atx - 6.0, P_atx[level], '-', color=clr, alpha=0.8,
                linewidth=1.5, label=f'{lbl} (ATX)')
    ax.axvline(14.0 - 6.0, color='red', alpha=0.3, linestyle=':', label='Dose')
    ax.set_xlabel('Hours from 6 AM')
    ax.set_ylabel('Precision (P)')
    ax.set_title('(a) P trajectories by hierarchy level')
    ax.legend(fontsize=6, ncol=2)

    # Panel (b): Biomarker comparison bars at peak
    ax = axes[1]
    bm_plot = ['alpha', 'lzw', 'hrv', 'pupil', 'bdnf']
    bm_labels = ['EEG alpha', 'LZW', 'HRV', 'Pupil', 'BDNF']
    x = np.arange(len(bm_plot))
    width = 0.35
    base_vals = [snap_base_peak[bm] for bm in bm_plot]
    atx_vals = [snap_atx_peak[bm] for bm in bm_plot]
    ax.bar(x - width/2, base_vals, width, label='Baseline', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, atx_vals, width, label='Atomoxetine (peak)', color='firebrick', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bm_labels, fontsize=8)
    ax.set_ylabel('Biomarker value')
    ax.set_title('(b) Biomarker comparison at peak effect')
    ax.legend(fontsize=7)

    # Panel (c): Neuromodulator time courses
    ax = axes[2]
    ax.plot(t_base - 6.0, nm_base['NE'], '--', color='steelblue', alpha=0.5,
            linewidth=1, label='NE (baseline)')
    ax.plot(t_atx - 6.0, nm_atx['NE'], '-', color='steelblue', alpha=0.9,
            linewidth=1.5, label='NE (ATX)')
    ax.plot(t_base - 6.0, st_base['cortisol'], '--', color='darkorange', alpha=0.5,
            linewidth=1, label='Cortisol (baseline)')
    ax.plot(t_atx - 6.0, st_atx['cortisol'], '-', color='darkorange', alpha=0.9,
            linewidth=1.5, label='Cortisol (ATX)')
    ax.axvline(14.0 - 6.0, color='red', alpha=0.3, linestyle=':', label='Dose')
    ax.set_xlabel('Hours from 6 AM')
    ax.set_ylabel('Neuromodulator level')
    ax.set_title('(c) Neuromodulator time courses')
    ax.legend(fontsize=6)

    fig.suptitle('SIM 1: Acute Atomoxetine in Healthy Subject', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'S1_acute_atomoxetine.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: S1_acute_atomoxetine.png")

    return snap_base_peak, snap_atx_peak


# ============================================================================
# SIM 2: Chronic 4-6 weeks in ADHD
# ============================================================================
def sim_chronic_adhd(output_dir):
    """
    ADHD state (DAT/NET dysfunction) with chronic atomoxetine treatment.
    Prediction: atomoxetine should normalize P variability (reduce moment-to-moment
    fluctuations caused by noisy catecholamine signaling).
    """
    print("\n" + "="*70)
    print("SIM 2: Chronic Atomoxetine in ADHD")
    print("="*70)

    adhd_weeks_pre = 2
    treatment_weeks = 4
    total_hours = (adhd_weeks_pre + treatment_weeks) * 7 * 24

    # --- ADHD baseline (no treatment), run the full duration ---
    t_ctrl, P_ctrl, st_ctrl, nm_ctrl = simulate_v2(
        t_span=(6.0, 6.0 + total_hours),
        dt=0.2, seed=42,
        dat_dysfunction=0.6,
        net_dysfunction=0.8,
    )

    # --- ADHD with atomoxetine starting at week 2 ---
    # Build daily doses at 8 AM for 4 weeks, starting at day 14
    treatment_start_h = 6.0 + adhd_weeks_pre * 7 * 24  # hour when treatment begins
    atx_doses = []
    for day in range(treatment_weeks * 7):
        dose_time = treatment_start_h + day * 24 + 2.0  # 8 AM = 6+2=8
        atx_doses.append((dose_time, 0.5))

    t_atx, P_atx, st_atx, nm_atx = simulate_v2(
        t_span=(6.0, 6.0 + total_hours),
        dt=0.2, seed=42,
        dat_dysfunction=0.6,
        net_dysfunction=0.8,
        pharma_atomoxetine=atx_doses,
    )

    # Compute P variability: rolling 24h std of conceptual P (waking only)
    def _rolling_p_variability(t_arr, P_arr, nm_arr, window_h=24.0):
        """Compute rolling std of conceptual P in 24h windows, waking only."""
        centers = np.arange(t_arr[0] + window_h/2, t_arr[-1] - window_h/2, 12.0)
        variabilities = []
        times = []
        for tc in centers:
            mask = (t_arr >= tc - window_h/2) & (t_arr <= tc + window_h/2)
            wake = nm_arr['sleep'][mask] < 0.3
            if np.sum(wake) > 10:
                variabilities.append(np.std(P_arr['conceptual'][mask][wake]))
                times.append(tc)
        return np.array(times), np.array(variabilities)

    t_var_ctrl, var_ctrl = _rolling_p_variability(t_ctrl, P_ctrl, nm_ctrl)
    t_var_atx, var_atx = _rolling_p_variability(t_atx, P_atx, nm_atx)

    # Convert to days from start
    t_var_ctrl_days = (t_var_ctrl - 6.0) / 24.0
    t_var_atx_days = (t_var_atx - 6.0) / 24.0

    # Summary statistics
    pre_mask_ctrl = t_var_ctrl_days < adhd_weeks_pre * 7
    post_mask_ctrl = t_var_ctrl_days >= adhd_weeks_pre * 7
    pre_mask_atx = t_var_atx_days < adhd_weeks_pre * 7
    post_mask_atx = t_var_atx_days >= adhd_weeks_pre * 7

    print(f"\n  P variability (24h rolling std of waking P_conceptual):")
    print(f"    ADHD baseline (pre-treatment):  {np.mean(var_ctrl[pre_mask_ctrl]):.4f}")
    print(f"    ADHD no-treatment (weeks 3-6):  {np.mean(var_ctrl[post_mask_ctrl]):.4f}")
    print(f"    ADHD + ATX (weeks 3-6):         {np.mean(var_atx[post_mask_atx]):.4f}")

    # Also check mean P levels
    late_mask = t_ctrl > (t_ctrl[-1] - 7 * 24)
    wake_late_ctrl = nm_ctrl['sleep'][late_mask] < 0.3
    wake_late_atx = nm_atx['sleep'][late_mask] < 0.3

    if np.any(wake_late_ctrl):
        mean_Pc_ctrl = np.mean(P_ctrl['conceptual'][late_mask][wake_late_ctrl])
    else:
        mean_Pc_ctrl = np.nan
    if np.any(wake_late_atx):
        mean_Pc_atx = np.mean(P_atx['conceptual'][late_mask][wake_late_atx])
    else:
        mean_Pc_atx = np.nan

    print(f"\n  Mean waking P_c (last week):")
    print(f"    ADHD no-treatment: {mean_Pc_ctrl:.4f}")
    print(f"    ADHD + ATX:        {mean_Pc_atx:.4f}")

    variability_reduction = 1.0 - np.mean(var_atx[post_mask_atx]) / np.mean(var_ctrl[post_mask_ctrl])
    print(f"\n  Variability reduction: {variability_reduction*100:.1f}%")

    # --- FIGURE ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel A: P variability over time
    ax = axes[0]
    ax.plot(t_var_ctrl_days, var_ctrl, 'gray', alpha=0.6, linewidth=1.5,
            label='ADHD (no treatment)')
    ax.plot(t_var_atx_days, var_atx, 'steelblue', alpha=0.8, linewidth=1.5,
            label='ADHD + Atomoxetine')
    ax.axvline(adhd_weeks_pre * 7, color='red', linestyle=':', alpha=0.5,
               label='Treatment starts')
    ax.set_xlabel('Day')
    ax.set_ylabel('P variability (24h rolling std)')
    ax.set_title('A: P Variability Over Time')
    ax.legend(fontsize=8)

    # Panel B: Raw P_c traces (last 3 days of treatment)
    ax = axes[1]
    last3d_start = t_ctrl[-1] - 3 * 24
    mask_ctrl_3d = t_ctrl >= last3d_start
    mask_atx_3d = t_atx >= last3d_start
    ax.plot((t_ctrl[mask_ctrl_3d] - last3d_start) / 24, P_ctrl['conceptual'][mask_ctrl_3d],
            'gray', alpha=0.5, linewidth=0.8, label='ADHD (no treatment)')
    ax.plot((t_atx[mask_atx_3d] - last3d_start) / 24, P_atx['conceptual'][mask_atx_3d],
            'steelblue', alpha=0.7, linewidth=0.8, label='ADHD + ATX')
    ax.set_xlabel('Days (last 3 days of simulation)')
    ax.set_ylabel('P conceptual')
    ax.set_title('B: Raw P Traces (last 3 days) — ATX reduces moment-to-moment noise')
    ax.legend(fontsize=8)

    fig.suptitle('SIM 2: Chronic Atomoxetine in ADHD — P Variability Normalization',
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'S2_chronic_adhd_atomoxetine.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: S2_chronic_adhd_atomoxetine.png")

    return variability_reduction


# ============================================================================
# SIM 3: Head-to-head atomoxetine vs psilocybin
# ============================================================================
def sim_head_to_head(output_dir):
    """
    Compare atomoxetine (daily for 1 week) vs psilocybin (single dose day 1)
    in a healthy subject. Key prediction: opposite directions on most biomarkers.
    """
    print("\n" + "="*70)
    print("SIM 3: Head-to-Head Atomoxetine vs Psilocybin")
    print("="*70)

    sim_hours = 7 * 24  # 1 week

    # --- Baseline: healthy, no treatment ---
    t_base, P_base, st_base, nm_base = simulate_v2(
        t_span=(6.0, 6.0 + sim_hours), dt=0.1, seed=42,
    )

    # --- Atomoxetine: daily for 1 week (8 AM dose) ---
    atx_doses = [(6.0 + d * 24 + 2.0, 0.5) for d in range(7)]  # 8 AM each day
    t_atx, P_atx, st_atx, nm_atx = simulate_v2(
        t_span=(6.0, 6.0 + sim_hours), dt=0.1, seed=42,
        pharma_atomoxetine=atx_doses,
    )

    # --- Psilocybin: single dose day 1 at 10 AM ---
    t_psi, P_psi, st_psi, nm_psi = simulate_v2(
        t_span=(6.0, 6.0 + sim_hours), dt=0.1, seed=42,
        pharma_psilocybin=[(10.0, 0.6)],
    )

    # Extract day-7 snapshot (last day waking hours)
    day7_start = 6.0 + 6 * 24 + 8  # day 7, 2 PM
    day7_end = 6.0 + 6 * 24 + 20   # day 7, 2 AM
    snap_base = _extract_snapshot(t_base, P_base, st_base, nm_base, day7_start, day7_end)
    snap_atx = _extract_snapshot(t_atx, P_atx, st_atx, nm_atx, day7_start, day7_end)
    snap_psi = _extract_snapshot(t_psi, P_psi, st_psi, nm_psi, day7_start, day7_end)

    # Print comparison
    biomarkers = ['P_c', 'P_s', 'P_sm', 'alpha', 'lzw', 'hrv', 'pupil', 'bdnf', 'p300', 'NE']
    bm_labels = ['P_c', 'P_s', 'P_sm', 'Alpha', 'LZW', 'HRV', 'Pupil', 'BDNF', 'P300', 'NE']
    print(f"\n  Day 7 biomarker comparison:")
    print(f"  {'Biomarker':<12} {'Baseline':>10} {'ATX':>10} {'ATX delta':>10} "
          f"{'Psilo':>10} {'Psilo delta':>10} {'Opposite?':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    opposite_count = 0
    total_count = 0
    directions = {}
    for bm, lbl in zip(biomarkers, bm_labels):
        b = snap_base[bm]
        a = snap_atx[bm]
        p = snap_psi[bm]
        d_atx = a - b
        d_psi = p - b
        opposite = "YES" if (d_atx * d_psi < 0 and abs(d_atx) > 0.001 and abs(d_psi) > 0.001) else "no"
        if abs(d_atx) > 0.001 and abs(d_psi) > 0.001:
            total_count += 1
            if d_atx * d_psi < 0:
                opposite_count += 1
        directions[bm] = {'atx': d_atx, 'psi': d_psi}
        print(f"  {lbl:<12} {b:10.4f} {a:10.4f} {d_atx:+10.4f} "
              f"{p:10.4f} {d_psi:+10.4f} {opposite:>10}")

    print(f"\n  Opposite direction: {opposite_count}/{total_count} biomarkers")

    # --- FIGURE: grouped bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Grouped bar chart of day-7 biomarker deltas
    ax = axes[0]
    bm_plot = ['P_c', 'P_s', 'P_sm', 'alpha', 'lzw', 'hrv', 'pupil', 'bdnf']
    bm_plot_labels = ['P_c', 'P_s', 'P_sm', 'Alpha', 'LZW', 'HRV', 'Pupil', 'BDNF']
    x = np.arange(len(bm_plot))
    width = 0.35
    atx_deltas = [snap_atx[bm] - snap_base[bm] for bm in bm_plot]
    psi_deltas = [snap_psi[bm] - snap_base[bm] for bm in bm_plot]
    bars1 = ax.bar(x - width/2, atx_deltas, width, label='Atomoxetine (7d daily)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, psi_deltas, width, label='Psilocybin (single dose)',
                   color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bm_plot_labels, fontsize=8, rotation=15)
    ax.set_ylabel('Change from baseline')
    ax.set_title('A: Day 7 Biomarker Changes (vs baseline)')
    ax.legend(fontsize=7)
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel B: P_conceptual trajectory over 7 days
    ax = axes[1]
    ax.plot((t_base - 6.0) / 24, P_base['conceptual'], 'gray', alpha=0.4,
            linewidth=1, label='Baseline')
    ax.plot((t_atx - 6.0) / 24, P_atx['conceptual'], 'steelblue', alpha=0.7,
            linewidth=1.5, label='Atomoxetine (daily)')
    ax.plot((t_psi - 6.0) / 24, P_psi['conceptual'], 'darkorange', alpha=0.7,
            linewidth=1.5, label='Psilocybin (day 1)')
    ax.set_xlabel('Day')
    ax.set_ylabel('P conceptual')
    ax.set_title('B: P Conceptual Trajectory Over 7 Days')
    ax.legend(fontsize=7)

    fig.suptitle('SIM 3: Atomoxetine vs Psilocybin — Opposite Biomarker Directions',
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'S3_head_to_head.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: S3_head_to_head.png")

    return directions


# ============================================================================
# SIM 4: Atomoxetine in depression vs SSRI vs ketamine
# ============================================================================
def sim_depression_comparison(output_dir):
    """
    Compare atomoxetine, SSRI, and ketamine in a depressed state.
    Each has a distinct P trajectory:
      - Ketamine: rapid P drop, fades over 1-2 weeks
      - SSRI: initial P increase (side effects), then slow decrease (weeks 3+)
      - Atomoxetine: gradual P increase via NE (different mechanism)
    """
    print("\n" + "="*70)
    print("SIM 4: Atomoxetine in Depression — Comparison with SSRI and Ketamine")
    print("="*70)

    treatment_weeks = 4
    evolution_weeks = 4
    treatment_hours = treatment_weeks * 7 * 24
    total_hours = evolution_weeks * 7 * 24 + treatment_hours

    # Evolve depressed state
    t_evo, P_evo, st_evo, nm_evo = simulate_v2(
        t_span=(6.0, 6.0 + evolution_weeks * 7 * 24),
        dt=0.2, seed=42,
        chronic_stress=0.6,
    )
    late = t_evo > (t_evo[-1] - 3 * 24)
    wake = nm_evo['sleep'][late] < 0.3
    idx = np.where(late)[0]

    dep_state = SimulationState(
        P_s=np.mean(P_evo['sensory'][idx][wake]) if np.any(wake) else P_SENSORY_BASELINE,
        P_c=np.mean(P_evo['conceptual'][idx][wake]) if np.any(wake) else P_CONCEPTUAL_BASELINE,
        P_sm=np.mean(P_evo['selfmodel'][idx][wake]) if np.any(wake) else P_SELFMODEL_BASELINE,
        hpa_sensitivity=st_evo['hpa_sensitivity'][idx[-1]],
        allostatic_load=st_evo['allostatic_load'][idx[-1]],
        cortisol=st_evo['cortisol'][idx[-1]],
    )

    dep_snap = _extract_snapshot(t_evo, P_evo, st_evo, nm_evo,
                                  t_evo[-1] - 3*24, t_evo[-1])
    print(f"  Depressed baseline: P_c={dep_snap['P_c']:.3f}")

    treatment_start = 6.0  # treatments begin at sim start (depressed state already evolved)

    # --- No treatment (depressed control) ---
    t_ctrl, P_ctrl, st_ctrl, nm_ctrl = simulate_v2(
        t_span=(treatment_start, treatment_start + treatment_hours),
        dt=0.2, seed=42,
        state0=dep_state,
        chronic_stress=0.5,
    )

    # --- Atomoxetine: daily at 8 AM for 4 weeks ---
    atx_doses = [(treatment_start + d * 24 + 2.0, 0.5) for d in range(treatment_weeks * 7)]
    t_atx, P_atx, st_atx, nm_atx = simulate_v2(
        t_span=(treatment_start, treatment_start + treatment_hours),
        dt=0.2, seed=42,
        state0=dep_state,
        chronic_stress=0.5,
        pharma_atomoxetine=atx_doses,
    )

    # --- SSRI: start at day 0 ---
    t_ssri, P_ssri, st_ssri, nm_ssri = simulate_v2(
        t_span=(treatment_start, treatment_start + treatment_hours),
        dt=0.2, seed=42,
        state0=dep_state,
        chronic_stress=0.5,
        pharma_ssri=(treatment_start, 0.15),
    )

    # --- Ketamine: single dose at day 0, 2 PM ---
    t_ket, P_ket, st_ket, nm_ket = simulate_v2(
        t_span=(treatment_start, treatment_start + treatment_hours),
        dt=0.2, seed=42,
        state0=dep_state,
        chronic_stress=0.5,
        pharma_ketamine=[(treatment_start + 8.0, 0.5)],  # 2 PM on day 1
    )

    # Extract daily waking P_c for each condition
    days = list(range(1, treatment_weeks * 7 + 1))
    conditions = {
        'No treatment': (t_ctrl, P_ctrl, st_ctrl, nm_ctrl),
        'Atomoxetine': (t_atx, P_atx, st_atx, nm_atx),
        'SSRI': (t_ssri, P_ssri, st_ssri, nm_ssri),
        'Ketamine': (t_ket, P_ket, st_ket, nm_ket),
    }
    daily_P = {}
    for name, (t_arr, P_arr, st_arr, nm_arr) in conditions.items():
        pc_daily = []
        for d in days:
            t_start = treatment_start + (d - 1) * 24 + 8
            t_end = treatment_start + (d - 1) * 24 + 20
            mask = (t_arr >= t_start) & (t_arr <= t_end) & (nm_arr['sleep'] < 0.3)
            if np.any(mask):
                pc_daily.append(np.mean(P_arr['conceptual'][mask]))
            else:
                pc_daily.append(np.nan)
        daily_P[name] = pc_daily

    # Print comparison at key timepoints
    check_days = [1, 3, 7, 14, 21, 28]
    print(f"\n  Daily waking P_c:")
    print(f"  {'Condition':<20} " + "".join(f"{'Day '+str(d):>10}" for d in check_days))
    print(f"  {'-'*20} " + "-"*10*len(check_days))
    for name, vals in daily_P.items():
        vs = [vals[d-1] for d in check_days if d <= len(vals)]
        print(f"  {name:<20} " + "".join(f"{v:10.4f}" for v in vs))

    # Get normal baseline for reference
    t_norm, P_norm, st_norm, nm_norm = simulate_v2(
        t_span=(6.0, 30.0), dt=0.05, seed=42,
    )
    normal_snap = _extract_snapshot(t_norm, P_norm, st_norm, nm_norm, 8.0, 20.0)
    print(f"\n  Normal baseline P_c: {normal_snap['P_c']:.3f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        'No treatment': 'gray',
        'Atomoxetine': 'steelblue',
        'SSRI': 'seagreen',
        'Ketamine': 'firebrick',
    }

    # Panel A: P_c trajectory over 4 weeks
    ax = axes[0]
    for name, vals in daily_P.items():
        ax.plot(days, vals, '-o', markersize=2, color=colors[name],
                label=name, linewidth=1.5, alpha=0.8)
    ax.axhline(normal_snap['P_c'], color='green', linestyle='--', alpha=0.3, label='Normal')
    ax.axhline(dep_snap['P_c'], color='gray', linestyle=':', alpha=0.3, label='Depressed')
    ax.set_xlabel('Day')
    ax.set_ylabel('P conceptual (waking mean)')
    ax.set_title('A: P Trajectory Over 4 Weeks of Treatment')
    ax.legend(fontsize=7)

    # Panel B: Change from depressed baseline at key timepoints
    ax = axes[1]
    cond_list = ['Atomoxetine', 'SSRI', 'Ketamine']
    x = np.arange(len(check_days))
    width = 0.25
    for i, cond in enumerate(cond_list):
        vals = [daily_P[cond][d-1] - dep_snap['P_c'] for d in check_days if d <= len(daily_P[cond])]
        ax.bar(x + i * width, vals, width, label=cond,
               color=colors[cond], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Day {d}' for d in check_days], fontsize=8)
    ax.set_ylabel('Change from depressed baseline')
    ax.set_title('B: P Change from Depression (negative = improvement)')
    ax.legend(fontsize=7)
    ax.axhline(0, color='black', linewidth=0.5)

    fig.suptitle('SIM 4: Atomoxetine vs SSRI vs Ketamine in Depression', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'S4_depression_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: S4_depression_comparison.png")

    return daily_P


# ============================================================================
# SUMMARY TABLE
# ============================================================================
def print_summary():
    print("\n" + "="*70)
    print("SUMMARY: ATOMOXETINE PREDICTIONS FROM PLASTICITY MODEL")
    print("="*70)
    print("""
    S1: ACUTE SINGLE DOSE IN HEALTHY SUBJECT
        Atomoxetine increases NE via NET blockade → P increases at all levels.
        Predicted biomarker directions:
          Alpha power: UP (higher P → more alpha)
          LZW complexity: DOWN (higher P → less entropy)
          Pupil: UP (NE dilation)
          HRV: DOWN (NE → vagal withdrawal)
          BDNF: minimal change (no direct plasticity pathway)

    S2: CHRONIC 4-6 WEEKS IN ADHD
        Daily atomoxetine normalizes P variability. ADHD = noisy catecholamine
        signaling → high moment-to-moment P fluctuations. NET blockade stabilizes
        NE → more consistent P → improved sustained attention.

    S3: HEAD-TO-HEAD vs PSILOCYBIN
        Atomoxetine and psilocybin produce OPPOSITE effects on most biomarkers.
        ATX: P UP, alpha UP, LZW DOWN, pupil UP.
        Psilocybin: P DOWN, alpha DOWN, LZW UP, pupil DOWN.
        This is a strong out-of-sample prediction of the model.

    S4: ATOMOXETINE IN DEPRESSION
        Different trajectory than SSRI or ketamine:
        - Ketamine: rapid P drop, fades 1-2 weeks
        - SSRI: initial worsening, then gradual improvement (weeks 3+)
        - Atomoxetine: gradual P elevation via NE — NOT a P-lowering treatment
        Prediction: atomoxetine is poor monotherapy for depression (wrong direction)
        but may help ADHD+depression comorbidity via variability reduction.
    """)


def print_direction_table():
    """Print predicted biomarker direction table."""
    print("\n" + "="*70)
    print("PREDICTED BIOMARKER DIRECTIONS")
    print("="*70)
    print(f"  {'Biomarker':<15} {'Atomoxetine':>12} {'Psilocybin':>12} {'Opposite?':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    predictions = [
        ('EEG alpha',     'UP',   'DOWN', 'YES'),
        ('LZW complexity','DOWN', 'UP',   'YES'),
        ('HRV',           'DOWN', 'UP',   'YES'),
        ('Pupil',         'UP',   'DOWN', 'YES'),
        ('BDNF',          '~',    'UP',   'n/a'),
        ('P conceptual',  'UP',   'DOWN', 'YES'),
        ('P sensory',     'UP',   'DOWN', 'YES'),
        ('P self-model',  'UP',   'DOWN', 'YES'),
    ]
    for bm, atx, psi, opp in predictions:
        print(f"  {bm:<15} {atx:>12} {psi:>12} {opp:>10}")
    print()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("="*70)
    print("Atomoxetine Validation Simulations")
    print("Precision-Weighting Plasticity Model")
    print("="*70)

    output_dir = _make_output_dir()
    print(f"Output: {output_dir}\n")

    print_summary()

    sim_acute_single_dose(output_dir)
    sim_chronic_adhd(output_dir)
    sim_head_to_head(output_dir)
    sim_depression_comparison(output_dir)

    print_direction_table()

    print("\n" + "="*70)
    print("ALL ATOMOXETINE SIMULATIONS COMPLETE")
    print(f"Figures saved to: {output_dir}")
    print("="*70)
