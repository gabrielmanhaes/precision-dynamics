"""
Publication Figures
===================

Generates all publication-quality figures from a single script.
Figures are designed for a computational neuroscience paper on the
precision-weighting plasticity model.

Figures:
  1. Model architecture schematic (state vector, dynamics)
  2. Target scatter plot (predicted vs published)
  3. Target bar charts by category (alpha, LZW, cortisol, P, clinical, biomarker)
  4. Model comparison table (AIC/BIC for full vs reduced)
  5. Out-of-sample predictions (atomoxetine vs psilocybin direction comparison)
  6. Novel predictions panel (meditation, aging, anesthesia depth curve)
  7. P operationalization functions (alpha, LZW, HRV, pupil vs P)

Run:
    python3 paper_figures.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitting_v3 import (
    build_targets, evaluate_model, compute_r_squared,
    model_comparison, generate_novel_predictions,
    CONSOLIDATED_FIXED_VALUES,
)
from parameters import *
from model import (
    simulate_v2, SimulationState,
    p_to_eeg_alpha, p_to_eeg_alpha_state, p_to_lzw, p_to_lzw_state, p_to_p300,
    ne_to_pupil, ne_cort_to_hrv, plasticity_to_bdnf,
    norepinephrine, serotonin, dopamine, acetylcholine,
    endogenous_plasticity, cortisol_rhythm, is_sleep, sleep_stage,
)


# Best-fit parameters (consolidated: 9 fitted + 5 fixed)
# Matches validation_summary.py BEST_FIT exactly
BEST_FIT = {
    'BETA_PLAST': 0.9095,
    'ALPHA_POWER_EXPONENT': 2.5805,
    'LZW_EXPONENT': 0.3925,
    'P_CONCEPTUAL_NREM': 0.5723,
    'CORTISOL_STRESS_GAIN': 1.8104,
    'GABA_NE_GAIN_MOD': 0.6857,
    'PSILOCYBIN_PHARMA_GAIN': 0.3741,
    'PTSD_DISSOC_COEFF': 0.1518,
    'ALPHA_NE_PHASIC': 1.1660,
}

FIXED = dict(CONSOLIDATED_FIXED_VALUES)


def _make_output_dir():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'figures', 'paper')
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


def fig1_neuromodulator_dynamics(output_dir):
    """Figure 1: 24h neuromodulator time courses and P dynamics."""
    print("  Generating Figure 1: Neuromodulator dynamics...")

    sim_override = {k: v for k, v in {**FIXED, **BEST_FIT}.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT')}

    t, P, st, nm = simulate_v2(
        t_span=(6.0, 30.0), dt=0.02, seed=42,
        params_override=sim_override,
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Panel A: Sleep architecture
    ax = axes[0]
    ax.fill_between(t, nm['sleep'], alpha=0.3, color='navy', label='Sleep drive')
    ax.fill_between(t, nm['REM'], alpha=0.4, color='purple', label='REM')
    ax.set_ylabel('Sleep drive')
    ax.set_title('A. Sleep Architecture')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 1.1)

    # Panel B: Neuromodulators
    ax = axes[1]
    ax.plot(t, nm['NE'], 'r-', linewidth=1.0, label='NE', alpha=0.8)
    ax.plot(t, nm['5-HT'], 'b-', linewidth=1.0, label='5-HT', alpha=0.8)
    ax.plot(t, nm['DA'], 'g-', linewidth=1.0, label='DA', alpha=0.8)
    ax.plot(t, nm['ACh'], 'm-', linewidth=1.0, label='ACh', alpha=0.8)
    ax.plot(t, nm['endogenous_plasticity'], 'k-', linewidth=1.0,
            label='Plasticity', alpha=0.6)
    ax.set_ylabel('Neuromodulator level')
    ax.set_title('B. Neuromodulator Time Courses')
    ax.legend(fontsize=7, loc='upper right', ncol=3)

    # Panel C: Precision per level
    ax = axes[2]
    ax.plot(t, P['sensory'], 'b-', linewidth=1.0, label='P_sensory', alpha=0.8)
    ax.plot(t, P['conceptual'], 'g-', linewidth=1.0, label='P_conceptual', alpha=0.8)
    ax.plot(t, P['selfmodel'], 'r-', linewidth=1.0, label='P_selfmodel', alpha=0.8)
    ax.axhline(P_WAKING, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('Precision (P)')
    ax.set_title('C. Per-Level Precision Dynamics')
    ax.legend(fontsize=7, loc='upper right')

    # Panel D: HPA axis
    ax = axes[3]
    ax.plot(t, st['cortisol'], 'orange', linewidth=1.0, label='Cortisol')
    ax.plot(t, st['allostatic_load'], 'brown', linewidth=1.0,
            label='Allostatic load', alpha=0.7)
    ax.set_ylabel('Level')
    ax.set_xlabel('Time (hours)')
    ax.set_title('D. HPA Axis')
    ax.legend(fontsize=7, loc='upper right')

    # Time axis labels
    for ax in axes:
        ax.set_xlim(6, 30)
        ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
        ax.set_xticklabels(['6:00', '8:00', '10:00', '12:00', '14:00',
                            '16:00', '18:00', '20:00', '22:00', '0:00',
                            '2:00', '4:00', '6:00'])

    fig.suptitle('Normal 24-Hour Simulation', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_neuromodulator_dynamics.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig2_target_scatter(output_dir):
    """Figure 2: Scatter plot — predicted vs published for 19 empirical targets."""
    print("  Generating Figure 2: Target scatter plot...")

    targets = build_targets()
    predictions = evaluate_model(BEST_FIT, targets, dt=0.1,
                                 fixed_override=FIXED)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Color by category
    cat_colors = {
        'alpha': 'blue', 'lzw': 'green', 'cortisol': 'orange',
        'P': 'purple', 'clinical': 'red', 'biomarker': 'brown',
    }

    for tgt in targets:
        pred = predictions.get(tgt.name, 0.0)
        pub = tgt.published_value
        color = cat_colors.get(tgt.category, 'gray')
        marker = 'o' if tgt.train else 's'
        ax.scatter(pub, pred, c=color, marker=marker, s=50, alpha=0.7,
                   edgecolors='black', linewidths=0.5)

    # Diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

    # Legend for categories
    for cat, color in cat_colors.items():
        ax.scatter([], [], c=color, marker='o', label=cat, s=50)
    ax.scatter([], [], c='gray', marker='o', label='Train', s=50)
    ax.scatter([], [], c='gray', marker='s', label='Test', s=50)
    ax.legend(fontsize=8, loc='upper left')

    # R² annotation
    param_names = list(BEST_FIT.keys())
    param_vector = np.array([BEST_FIT[k] for k in param_names])
    all_r2, _, _ = compute_r_squared(param_vector, param_names, targets,
                                      dt=0.1, fixed_override=FIXED)
    train_r2, _, _ = compute_r_squared(
        param_vector, param_names, [t for t in targets if t.train],
        dt=0.1, fixed_override=FIXED)
    test_r2, _, _ = compute_r_squared(
        param_vector, param_names, [t for t in targets if not t.train],
        dt=0.1, fixed_override=FIXED)
    ax.text(0.05, 0.95, f'Train R² = {train_r2:.3f}\nTest R² = {test_r2:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Published value', fontsize=12)
    ax.set_ylabel('Model prediction', fontsize=12)
    ax.set_title('Predicted vs Published Values (24 targets)', fontsize=13)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_target_scatter.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig3_target_bars(output_dir):
    """Figure 3: Bar charts of predictions by category."""
    print("  Generating Figure 3: Target bar charts...")

    targets = build_targets()
    predictions = evaluate_model(BEST_FIT, targets, dt=0.1,
                                 fixed_override=FIXED)

    # Group by category
    categories = {}
    for tgt in targets:
        if tgt.category not in categories:
            categories[tgt.category] = []
        categories[tgt.category].append(tgt)

    n_cats = len(categories)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    cat_colors = {
        'alpha': 'steelblue', 'lzw': 'seagreen', 'cortisol': 'darkorange',
        'P': 'mediumpurple', 'clinical': 'indianred', 'biomarker': 'sienna',
    }

    for idx, (cat, cat_targets) in enumerate(categories.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        names = [t.name.replace('_', '\n') for t in cat_targets]
        pubs = [t.published_value for t in cat_targets]
        preds = [predictions.get(t.name, 0.0) for t in cat_targets]

        x = np.arange(len(names))
        w = 0.35
        color = cat_colors.get(cat, 'gray')

        ax.bar(x - w/2, pubs, w, color='lightgray', edgecolor='black',
               linewidth=0.5, label='Published')
        ax.bar(x + w/2, preds, w, color=color, edgecolor='black',
               linewidth=0.5, alpha=0.7, label='Predicted')

        # Error bars (tolerance)
        for i, tgt in enumerate(cat_targets):
            ax.errorbar(i - w/2, tgt.published_value, yerr=tgt.tolerance,
                        fmt='none', color='black', capsize=3, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=6, rotation=45, ha='right')
        ax.set_title(f'{cat.upper()} targets', fontweight='bold')
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(len(categories), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Model Predictions by Category', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_target_bars.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig4_operationalization(output_dir):
    """Figure 4: P operationalization functions."""
    print("  Generating Figure 4: P operationalization...")

    P_range = np.linspace(0.15, 0.95, 100)
    alpha_exp = BEST_FIT.get('ALPHA_POWER_EXPONENT', ALPHA_POWER_EXPONENT)
    lzw_exp = BEST_FIT.get('LZW_EXPONENT', LZW_EXPONENT)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Alpha vs P
    ax = axes[0, 0]
    alpha_vals = [p_to_eeg_alpha(p, exponent=alpha_exp) for p in P_range]
    ax.plot(P_range, alpha_vals, 'b-', linewidth=2)
    ax.set_xlabel('Precision (P)')
    ax.set_ylabel('EEG Alpha Power')
    ax.set_title(f'A. Alpha = P^{alpha_exp:.1f}')

    # LZW vs P
    ax = axes[0, 1]
    lzw_vals = [p_to_lzw(p, exponent=lzw_exp) for p in P_range]
    ax.plot(P_range, lzw_vals, 'g-', linewidth=2)
    ax.set_xlabel('Precision (P)')
    ax.set_ylabel('LZW Complexity')
    ax.set_title(f'B. LZW = (1-P)^{lzw_exp:.2f}')

    # HRV vs NE (at fixed cortisol)
    ax = axes[1, 0]
    NE_range = np.linspace(0.2, 0.8, 100)
    hrv_vals = [ne_cort_to_hrv(ne, 0.4) for ne in NE_range]
    ax.plot(NE_range, hrv_vals, 'r-', linewidth=2)
    ax.set_xlabel('NE level')
    ax.set_ylabel('HRV')
    ax.set_title('C. HRV = 1/(1 + k_NE*NE + k_cort*cort)')

    # Pupil vs NE
    ax = axes[1, 1]
    pupil_vals = [ne_to_pupil(ne) for ne in NE_range]
    ax.plot(NE_range, pupil_vals, 'm-', linewidth=2)
    ax.set_xlabel('NE level')
    ax.set_ylabel('Pupil diameter')
    ax.set_title('D. Pupil = baseline * (1 + k*NE)')

    fig.suptitle('Operationalization Functions: P/NE → Biomarkers',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_operationalization.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig5_pharmacology_comparison(output_dir):
    """Figure 5: Pharmacological comparison — psilocybin vs atomoxetine vs DMT."""
    print("  Generating Figure 5: Pharmacology comparison...")

    sim_override = {k: v for k, v in {**FIXED, **BEST_FIT}.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT')}

    # Normal baseline
    t_n, P_n, _, _ = simulate_v2(
        t_span=(12.0, 20.0), dt=0.02, seed=42,
        params_override=sim_override,
    )

    # Psilocybin
    t_psi, P_psi, _, _ = simulate_v2(
        t_span=(12.0, 20.0), dt=0.02, seed=42,
        pharma_psilocybin=[(14.0, 0.6)],
        params_override=sim_override,
    )

    # DMT
    t_dmt, P_dmt, _, _ = simulate_v2(
        t_span=(12.0, 20.0), dt=0.02, seed=42,
        pharma_dmt=[(14.0, 0.6)],
        params_override=sim_override,
    )

    # Atomoxetine
    t_atx, P_atx, _, _ = simulate_v2(
        t_span=(12.0, 20.0), dt=0.02, seed=42,
        pharma_atomoxetine=[(14.0, 0.5)],
        params_override=sim_override,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for lvl_idx, (lvl_name, lvl_key) in enumerate(
        [('Sensory', 'sensory'), ('Conceptual', 'conceptual'),
         ('Self-Model', 'selfmodel')]
    ):
        ax = axes[lvl_idx]
        ax.plot(t_n - 14.0, P_n[lvl_key], 'k-', linewidth=1, alpha=0.5,
                label='Baseline')
        ax.plot(t_psi - 14.0, P_psi[lvl_key], 'b-', linewidth=1.5,
                label='Psilocybin', alpha=0.8)
        ax.plot(t_dmt - 14.0, P_dmt[lvl_key], 'r-', linewidth=1.5,
                label='DMT', alpha=0.8)
        ax.plot(t_atx - 14.0, P_atx[lvl_key], 'g-', linewidth=1.5,
                label='Atomoxetine', alpha=0.8)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Hours from dose')
        ax.set_ylabel(f'P_{lvl_name.lower()}')
        ax.set_title(f'{lvl_name} Precision')
        ax.legend(fontsize=7)

    fig.suptitle('Pharmacological Effects on Per-Level Precision',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_pharmacology.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig6_condition_profiles(output_dir):
    """Figure 6: Condition-specific P profiles (all emerge from upstream params)."""
    print("  Generating Figure 6: Condition profiles...")

    sim_override = {k: v for k, v in {**FIXED, **BEST_FIT}.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT')}

    conditions = {
        'Normal': {},
        'Depression': {'chronic_stress': 0.6},
        'Psychosis': {'da_excess': 1.5, 'gaba_deficit': 0.3, 'chronic_stress': 0.3},
        'PTSD': {'ne_sensitization': 1.8, 'coupling_breakdown': 0.5,
                 'chronic_stress': 0.4, 'td_coupling_scale': PTSD_TD_BREAKDOWN},
        'ADHD': {'dat_dysfunction': 0.6, 'net_dysfunction': 0.8},
        'Anxiety': {'gaba_deficit': 0.175, 'chronic_stress': 0.3},
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, kwargs) in enumerate(conditions.items()):
        ax = axes[idx]
        t_span = (6.0, 30.0) if name != 'Depression' else (6.0, 6.0 + 2*7*24)
        dt = 0.05 if name != 'Depression' else 0.2

        t, P, st, nm = simulate_v2(
            t_span=t_span, dt=dt, seed=42,
            params_override=sim_override,
            **kwargs,
        )

        # Plot last 24h
        if name == 'Depression':
            mask = t > (t[-1] - 24)
        else:
            mask = np.ones(len(t), dtype=bool)

        t_plot = t[mask] - t[mask][0]
        ax.plot(t_plot, P['sensory'][mask], 'b-', linewidth=0.8,
                label='Sensory', alpha=0.7)
        ax.plot(t_plot, P['conceptual'][mask], 'g-', linewidth=0.8,
                label='Conceptual', alpha=0.7)
        ax.plot(t_plot, P['selfmodel'][mask], 'r-', linewidth=0.8,
                label='Self-model', alpha=0.7)

        # Waking means
        wake = nm['sleep'][mask] < 0.3
        if np.any(wake):
            mean_s = np.mean(P['sensory'][mask][wake])
            mean_c = np.mean(P['conceptual'][mask][wake])
            mean_sm = np.mean(P['selfmodel'][mask][wake])
            ax.set_title(f'{name}\nP_s={mean_s:.2f}, P_c={mean_c:.2f}, P_sm={mean_sm:.2f}',
                         fontsize=9)
        else:
            ax.set_title(name)

        ax.set_ylim(0.1, 1.0)
        ax.set_xlabel('Hours')
        ax.set_ylabel('P')
        if idx == 0:
            ax.legend(fontsize=6)

    fig.suptitle('Emergent P Profiles Across Conditions\n'
                 '(all from upstream parameters, no per-condition P overrides)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig6_condition_profiles.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig7_parameter_sensitivity(output_dir):
    """Figure 7: Parameter sensitivity — how each fitted param affects key targets."""
    print("  Generating Figure 7: Parameter sensitivity...")

    targets = build_targets()
    param_names = list(BEST_FIT.keys())
    param_vector = np.array([BEST_FIT[k] for k in param_names])
    key_targets = ['alpha_nrem', 'lzw_psilocybin', 'cortisol_dawn_nadir_ratio',
                   'depression_alpha_change', 'ptsd_sensory_p']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for p_idx, p_name in enumerate(param_names):
        if p_idx >= len(axes):
            break
        ax = axes[p_idx]

        # Sweep parameter ±50%
        base_val = param_vector[p_idx]
        sweep = np.linspace(base_val * 0.5, base_val * 1.5, 11)

        for tgt_name in key_targets:
            tgt = next((t for t in targets if t.name == tgt_name), None)
            if tgt is None:
                continue
            preds = []
            for sv in sweep:
                params = dict(zip(param_names, param_vector))
                params[p_name] = sv
                pred = evaluate_model(params, [tgt], dt=0.1,
                                      fixed_override=FIXED)
                preds.append(pred.get(tgt_name, 0.0))
            ax.plot(sweep / base_val, preds, '-o', markersize=3,
                    label=tgt_name[:20], linewidth=1)

        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel(f'{p_name} (relative to best-fit)')
        ax.set_ylabel('Prediction')
        ax.set_title(p_name, fontsize=9)
        if p_idx == 0:
            ax.legend(fontsize=5)

    fig.suptitle('Parameter Sensitivity Analysis',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig7_sensitivity.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_all(output_dir):
    """Generate all publication figures."""
    print(f"Output: {output_dir}\n")

    fig1_neuromodulator_dynamics(output_dir)
    fig2_target_scatter(output_dir)
    fig3_target_bars(output_dir)
    fig4_operationalization(output_dir)
    fig5_pharmacology_comparison(output_dir)
    fig6_condition_profiles(output_dir)
    fig7_parameter_sensitivity(output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    print("=" * 70)
    print("Publication Figure Generation")
    print("Precision-Weighting Plasticity Model")
    print("=" * 70)
    output_dir = _make_output_dir()
    generate_all(output_dir)
