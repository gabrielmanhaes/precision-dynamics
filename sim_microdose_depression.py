"""
Simulation: Psilocybin Microdosing Protocols for Depression
===========================================================

Compares multiple microdosing schedules applied to an evolved depressed state.
Tracks P levels, 5-HT2A receptor density (tolerance), biomarkers, and
therapeutic response across protocols.

Protocols:
  1. Fadiman: 1 day on / 2 days off (every 3rd day)
  2. Stamets: 4 days on / 3 days off
  3. Daily: continuous low-dose
  4. Weekend: 2 days on / 5 days off
  5. Control: depression, no treatment
  6. Full dose (single macro): for comparison

Output: timestamped folder in figures/v2/
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
    p_to_p300, ne_to_pupil, ne_cort_to_hrv, plasticity_to_bdnf,
    p_to_eeg_alpha, p_to_lzw,
)

# --- Fitted parameter overrides ---
# These are passed to every simulate_v2() call via params_override.
# Source: parsimonious mode best-fit (6 fitted + 6 fixed sloppy).
# Update after each full fitting run.
FITTED_PARAMS = {
    # 6 fitted (32-target parsimonious run, 2026-03-13)
    'ALPHA_NE':                0.0050,
    'ALPHA_5HT':               0.0050,
    'LZW_EXPONENT':            0.4326,
    'P_CONCEPTUAL_NREM':       0.5305,
    'GABA_NE_GAIN_MOD':        0.4662,
    'PSILOCYBIN_PHARMA_GAIN':  0.4317,
    # 6 fixed sloppy
    'BETA_PLAST':              1.2000,
    'GAMMA_SENSORY':           0.7215,
    'ALPHA_POWER_EXPONENT':    2.7136,
    'CORTISOL_STRESS_GAIN':    1.4606,
    'KETAMINE_PHARMA_GAIN':    5.0000,
    'PTSD_DISSOC_COEFF':       0.2211,
}

# --- Output directory ---
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'figures', 'v2', f'{STAMP}_microdose_depression')
os.makedirs(OUT_DIR, exist_ok=True)

# Symlink latest
LATEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'figures', 'v2', 'latest')
if os.path.islink(LATEST):
    os.remove(LATEST)
try:
    os.symlink(OUT_DIR, LATEST)
except OSError:
    pass


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PHASE 1: Evolve depressed state
# ============================================================================

print("=" * 70)
print("Microdosing Depression Simulation")
print("=" * 70)

DEP_STRESS = 0.8  # more severe depression
DEP_WEEKS = 8     # longer evolution for allostatic load buildup

print(f"\n  Phase 1: Evolving depressed baseline ({DEP_WEEKS} weeks, stress={DEP_STRESS})...")

t_dep, P_dep, st_dep, nm_dep = simulate_v2(
    t_span=(6.0, 6.0 + DEP_WEEKS * 7 * 24),
    dt=0.1, seed=42,
    chronic_stress=DEP_STRESS,
    params_override=FITTED_PARAMS,
)

# Extract depressed state from last waking period
wake_mask = nm_dep['sleep'] < 0.3
last_week = t_dep > (t_dep[-1] - 5 * 24)
wake_late = wake_mask & last_week
idx = np.where(wake_late)[0]

dep_state = SimulationState(
    P_s=np.mean(P_dep['sensory'][idx]),
    P_c=np.mean(P_dep['conceptual'][idx]),
    P_sm=np.mean(P_dep['selfmodel'][idx]),
    hpa_sensitivity=st_dep['hpa_sensitivity'][idx[-1]],
    allostatic_load=st_dep['allostatic_load'][idx[-1]],
    cortisol=st_dep['cortisol'][idx[-1]],
)

print(f"    Depressed state: P_s={dep_state.P_s:.3f}, P_c={dep_state.P_c:.3f}, "
      f"P_sm={dep_state.P_sm:.3f}")
print(f"    Allostatic load: {dep_state.allostatic_load:.3f}, "
      f"HPA sensitivity: {dep_state.hpa_sensitivity:.3f}")

# Also get normal baseline for comparison
t_norm, P_norm, st_norm, nm_norm = simulate_v2(
    t_span=(6.0, 30.0), dt=0.05, seed=42,
    params_override=FITTED_PARAMS)
wake_norm = nm_norm['sleep'] < 0.3
P_normal_c = np.mean(P_norm['conceptual'][wake_norm])
NE_normal = np.mean(nm_norm['NE'][wake_norm])
DA_normal = np.mean(nm_norm['DA'][wake_norm])
cort_normal = np.mean(st_norm['cortisol'][wake_norm])
plast_normal = np.mean(nm_norm['endogenous_plasticity'][wake_norm])

print(f"    Normal baseline: P_c={P_normal_c:.3f}")


# ============================================================================
# PHASE 2: Build microdosing schedules
# ============================================================================

TREATMENT_WEEKS = 8
TREATMENT_HOURS = TREATMENT_WEEKS * 7 * 24
SIM_START = 6.0
DOSE_TIME_OF_DAY = 8.0  # morning dose at 08:00
MICRODOSE_STRENGTH = 0.10  # ~1/6 of a full dose (0.6)
FULL_DOSE_STRENGTH = 0.6


def build_dose_schedule(protocol: str) -> list:
    """Return list of (dose_time, dose_strength) for the treatment period."""
    doses = []
    for day in range(TREATMENT_WEEKS * 7):
        t_dose = SIM_START + day * 24 + DOSE_TIME_OF_DAY

        if protocol == 'fadiman':
            # Day 1: dose, Day 2-3: off, repeat
            if day % 3 == 0:
                doses.append((t_dose, MICRODOSE_STRENGTH))

        elif protocol == 'stamets':
            # Days 1-4: dose, Days 5-7: off
            if day % 7 < 4:
                doses.append((t_dose, MICRODOSE_STRENGTH))

        elif protocol == 'daily':
            doses.append((t_dose, MICRODOSE_STRENGTH))

        elif protocol == 'weekend':
            # Saturday + Sunday (days 5,6 of week)
            if day % 7 >= 5:
                doses.append((t_dose, MICRODOSE_STRENGTH))

        elif protocol == 'full_dose':
            # Single macro dose on day 1
            if day == 0:
                doses.append((t_dose, FULL_DOSE_STRENGTH))

        elif protocol == 'control':
            pass  # no treatment

    return doses


PROTOCOLS = {
    'control': {'color': '#888888', 'ls': '--', 'label': 'Depression (no tx)'},
    'fadiman': {'color': '#2196F3', 'ls': '-', 'label': 'Fadiman (1on/2off)'},
    'stamets': {'color': '#4CAF50', 'ls': '-', 'label': 'Stamets (4on/3off)'},
    'daily':   {'color': '#FF9800', 'ls': '-', 'label': 'Daily microdose'},
    'weekend': {'color': '#9C27B0', 'ls': '-', 'label': 'Weekend (2on/5off)'},
    'full_dose': {'color': '#F44336', 'ls': '-.', 'label': 'Single full dose'},
}


# ============================================================================
# PHASE 3: Run simulations
# ============================================================================

print(f"\n  Phase 2: Running {len(PROTOCOLS)} protocols ({TREATMENT_WEEKS} weeks each)...")

results = {}

for proto_name, style in PROTOCOLS.items():
    doses = build_dose_schedule(proto_name)
    print(f"    {proto_name}: {len(doses)} doses over {TREATMENT_WEEKS} weeks")

    t, P, st, nm = simulate_v2(
        t_span=(SIM_START, SIM_START + TREATMENT_HOURS),
        dt=0.1,
        seed=42,
        state0=SimulationState(
            P_s=dep_state.P_s, P_c=dep_state.P_c, P_sm=dep_state.P_sm,
            hpa_sensitivity=dep_state.hpa_sensitivity,
            allostatic_load=dep_state.allostatic_load,
            cortisol=dep_state.cortisol,
        ),
        chronic_stress=0.6,  # depression continues
        pharma_psilocybin=doses if doses else None,
        noise_scale=1.0,
        params_override=FITTED_PARAMS,
    )

    results[proto_name] = {
        't': t, 'P': P, 'st': st, 'nm': nm,
        'doses': doses, 'style': style,
    }


# ============================================================================
# PHASE 4: Extract weekly metrics
# ============================================================================

print("\n  Phase 3: Extracting weekly metrics...")

weekly_metrics = {proto: {'week': [], 'P_c': [], 'P_sm': [], 'P_s': [],
                           'R_5HT2A': [], 'hrv': [], 'bdnf': [], 'p300': [],
                           'improvement': []}
                  for proto in PROTOCOLS}

for proto_name, data in results.items():
    t = data['t']
    P = data['P']
    st = data['st']
    nm = data['nm']

    for week in range(TREATMENT_WEEKS):
        week_start = SIM_START + week * 7 * 24
        week_end = week_start + 7 * 24
        mask = (t >= week_start) & (t < week_end)
        wake = nm['sleep'][mask] < 0.3

        if not np.any(wake):
            continue

        P_c = np.mean(P['conceptual'][mask][wake])
        P_sm = np.mean(P['selfmodel'][mask][wake])
        P_s = np.mean(P['sensory'][mask][wake])
        NE = np.mean(nm['NE'][mask][wake])
        cort = np.mean(st['cortisol'][mask][wake])
        plast = np.mean(nm['endogenous_plasticity'][mask][wake])
        r_5ht2a = np.mean(st['R_5HT2A'][mask][wake])

        DA = np.mean(nm['DA'][mask][wake])
        hrv = ne_cort_to_hrv(NE, cort)
        bdnf = plasticity_to_bdnf(plast)
        p300 = p_to_p300(P_c, DA=DA)

        # Improvement: absolute P reduction from depressed baseline
        # Negative means P dropped (therapeutic — reducing over-precision)
        improvement = dep_state.P_c - P_c  # positive = therapeutic

        m = weekly_metrics[proto_name]
        m['week'].append(week + 1)
        m['P_c'].append(P_c)
        m['P_sm'].append(P_sm)
        m['P_s'].append(P_s)
        m['R_5HT2A'].append(r_5ht2a)
        m['hrv'].append(hrv)
        m['bdnf'].append(bdnf)
        m['p300'].append(p300)
        m['improvement'].append(improvement)


# ============================================================================
# PHASE 5: Print results table
# ============================================================================

print(f"\n  {'Protocol':<20} {'Doses':>5} {'Wk1 dP':>8} {'Wk4 dP':>8} {'Wk8 dP':>8} {'R_5HT2A':>8} {'P_c end':>8}")
print(f"  {'-'*70}")

for proto_name in PROTOCOLS:
    m = weekly_metrics[proto_name]
    n_doses = len(results[proto_name]['doses'])
    w1 = m['improvement'][0] if len(m['improvement']) > 0 else 0
    w4 = m['improvement'][3] if len(m['improvement']) > 3 else 0
    w8 = m['improvement'][-1] if len(m['improvement']) > 0 else 0
    r = m['R_5HT2A'][-1] if len(m['R_5HT2A']) > 0 else 1.0
    pc = m['P_c'][-1] if m['P_c'] else dep_state.P_c
    print(f"  {proto_name:<20} {n_doses:>5} {w1:>+8.4f} {w4:>+8.4f} {w8:>+8.4f} {r:>8.3f} {pc:>8.3f}")


# ============================================================================
# FIGURE 1: P_conceptual trajectories (daily averages)
# ============================================================================

print("\n  Phase 4: Generating figures...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Psilocybin Microdosing Protocols for Depression', fontsize=14, fontweight='bold')

# Panel 1: Weekly P_conceptual
ax = axes[0, 0]
for proto_name, style in PROTOCOLS.items():
    m = weekly_metrics[proto_name]
    ax.plot(m['week'], m['P_c'], color=style['color'], ls=style['ls'],
            lw=2, marker='o', markersize=5, label=style['label'])
ax.axhline(P_normal_c, color='black', ls=':', alpha=0.5, label=f'Normal P_c={P_normal_c:.2f}')
ax.axhline(dep_state.P_c, color='red', ls=':', alpha=0.3, label=f'Dep baseline={dep_state.P_c:.2f}')
ax.set_xlabel('Week')
ax.set_ylabel('Mean Waking P (conceptual)')
ax.set_title('Conceptual Precision Over Time', fontweight='bold')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# Panel 2: Weekly improvement ratio
ax = axes[0, 1]
for proto_name, style in PROTOCOLS.items():
    m = weekly_metrics[proto_name]
    ax.plot(m['week'], m['improvement'],
            color=style['color'], ls=style['ls'], lw=2, marker='o', markersize=5,
            label=style['label'])
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Week')
ax.set_ylabel('P reduction from baseline')
ax.set_title('Therapeutic P Reduction (positive = better)', fontweight='bold')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# Panel 3: 5-HT2A receptor density (tolerance)
ax = axes[1, 0]
for proto_name, style in PROTOCOLS.items():
    m = weekly_metrics[proto_name]
    ax.plot(m['week'], m['R_5HT2A'], color=style['color'], ls=style['ls'],
            lw=2, marker='s', markersize=5, label=style['label'])
ax.axhline(1.0, color='black', ls=':', alpha=0.5, label='Normal density')
ax.set_xlabel('Week')
ax.set_ylabel('5-HT2A Receptor Density')
ax.set_title('Receptor Tolerance (lower = more tolerance)', fontweight='bold')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# Panel 4: Self-model P (identity flexibility)
ax = axes[1, 1]
for proto_name, style in PROTOCOLS.items():
    m = weekly_metrics[proto_name]
    ax.plot(m['week'], m['P_sm'], color=style['color'], ls=style['ls'],
            lw=2, marker='d', markersize=5, label=style['label'])
ax.axhline(np.mean(P_norm['selfmodel'][wake_norm]), color='black', ls=':', alpha=0.5,
           label='Normal P_sm')
ax.set_xlabel('Week')
ax.set_ylabel('Mean Waking P (self-model)')
ax.set_title('Self-Model Precision (identity rigidity)', fontweight='bold')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, '01_microdose_trajectories.png')


# ============================================================================
# FIGURE 2: Biomarker comparison at week 6
# ============================================================================

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f'Biomarker Profiles at Week {TREATMENT_WEEKS}', fontsize=14, fontweight='bold')

biomarker_keys = ['p300', 'hrv', 'bdnf', 'improvement']
biomarker_labels = ['P300 Amplitude', 'HRV', 'BDNF', 'Improvement (%)']
biomarker_normals = [
    p_to_p300(P_normal_c, DA=DA_normal),
    ne_cort_to_hrv(NE_normal, cort_normal),
    plasticity_to_bdnf(plast_normal),
    1.0,
]

for ax_idx, (key, label, norm_val) in enumerate(zip(biomarker_keys, biomarker_labels, biomarker_normals)):
    ax = axes[ax_idx]
    proto_names = list(PROTOCOLS.keys())
    vals = []
    colors = []
    for proto in proto_names:
        m = weekly_metrics[proto]
        v = m[key][-1] if m[key] else 0
        if key == 'improvement':
            v *= 100  # percent
        vals.append(v)
        colors.append(PROTOCOLS[proto]['color'])

    bars = ax.bar(range(len(proto_names)), vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(proto_names)))
    ax.set_xticklabels([PROTOCOLS[p]['label'].split('(')[0].strip() for p in proto_names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(label, fontweight='bold')
    if key != 'improvement':
        ax.axhline(norm_val, color='black', ls=':', alpha=0.5, label='Normal')
        ax.legend(fontsize=7)
    else:
        ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.93])
savefig(fig, '02_biomarker_week6.png')


# ============================================================================
# FIGURE 3: High-resolution P trace for first 2 weeks (show individual doses)
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
fig.suptitle('First 2 Weeks: Detailed P Trajectories', fontsize=14, fontweight='bold')

TWO_WEEKS = 14 * 24
for ax, (level, level_name) in zip(axes, [('conceptual', 'Conceptual'),
                                           ('selfmodel', 'Self-Model'),
                                           ('sensory', 'Sensory')]):
    for proto_name in ['control', 'fadiman', 'stamets', 'daily', 'full_dose']:
        data = results[proto_name]
        style = data['style']
        t = data['t']
        mask_2w = t < (SIM_START + TWO_WEEKS)
        t_days = (t[mask_2w] - SIM_START) / 24.0
        ax.plot(t_days, data['P'][level][mask_2w],
                color=style['color'], ls=style['ls'], lw=0.8, alpha=0.8,
                label=style['label'])

    ax.set_ylabel(f'P ({level_name})')
    ax.set_title(f'{level_name} Level', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.1, 0.95)

axes[-1].set_xlabel('Days')
plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig(fig, '03_first_2weeks_detail.png')


# ============================================================================
# FIGURE 4: Dose-response exploration (varying microdose strength)
# ============================================================================

print("\n  Phase 5: Dose-response exploration...")

dose_strengths = [0.03, 0.05, 0.08, 0.12, 0.18]
dose_results = {}

for ds in dose_strengths:
    # Use Fadiman protocol at each dose level
    doses = []
    for day in range(TREATMENT_WEEKS * 7):
        if day % 3 == 0:
            doses.append((SIM_START + day * 24 + DOSE_TIME_OF_DAY, ds))

    t, P, st, nm = simulate_v2(
        t_span=(SIM_START, SIM_START + TREATMENT_HOURS),
        dt=0.1, seed=42,
        state0=SimulationState(
            P_s=dep_state.P_s, P_c=dep_state.P_c, P_sm=dep_state.P_sm,
            hpa_sensitivity=dep_state.hpa_sensitivity,
            allostatic_load=dep_state.allostatic_load,
            cortisol=dep_state.cortisol,
        ),
        chronic_stress=0.6,
        pharma_psilocybin=doses,
        params_override=FITTED_PARAMS,
    )

    # Week 6 metrics
    last_week = t > (t[-1] - 7 * 24)
    wake = nm['sleep'][last_week] < 0.3
    P_c_final = np.mean(P['conceptual'][last_week][wake]) if np.any(wake) else dep_state.P_c
    r_5ht2a_final = np.mean(st['R_5HT2A'][last_week][wake]) if np.any(wake) else 1.0

    improvement = dep_state.P_c - P_c_final

    dose_results[ds] = {
        'P_c': P_c_final,
        'R_5HT2A': r_5ht2a_final,
        'improvement': improvement,
    }
    print(f"    Fadiman @ strength={ds:.2f}: dP={improvement:+.4f}, "
          f"P_c={P_c_final:.3f}, R_5HT2A={r_5ht2a_final:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Dose-Response: Fadiman Protocol at Different Microdose Strengths',
             fontsize=13, fontweight='bold')

ds_list = sorted(dose_results.keys())

ax = axes[0]
ax.plot(ds_list, [dose_results[d]['improvement'] for d in ds_list],
        'bo-', lw=2, markersize=8)
ax.set_xlabel('Microdose Strength (fraction of full dose)')
ax.set_ylabel(f'P reduction at Week {TREATMENT_WEEKS}')
ax.set_title('Therapeutic Response', fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ds_list, [dose_results[d]['R_5HT2A'] for d in ds_list],
        'rs-', lw=2, markersize=8)
ax.set_xlabel('Microdose Strength')
ax.set_ylabel('5-HT2A Density at Week 6')
ax.set_title('Receptor Tolerance', fontweight='bold')
ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(ds_list, [dose_results[d]['P_c'] for d in ds_list],
        'gd-', lw=2, markersize=8)
ax.axhline(P_normal_c, color='black', ls=':', alpha=0.5, label='Normal')
ax.axhline(dep_state.P_c, color='red', ls=':', alpha=0.3, label='Dep baseline')
ax.set_xlabel('Microdose Strength')
ax.set_ylabel('P (conceptual) at Week 6')
ax.set_title('Final Precision', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
savefig(fig, '04_dose_response.png')


# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"\n  Normal P_conceptual:    {P_normal_c:.3f}")
print(f"  Depressed P_conceptual: {dep_state.P_c:.3f} (delta: {dep_state.P_c - P_normal_c:+.3f})")
print(f"\n  Week 6 outcomes:")

best_proto = None
best_improvement = -999

for proto_name in PROTOCOLS:
    m = weekly_metrics[proto_name]
    if m['improvement']:
        imp = m['improvement'][-1]
        r = m['R_5HT2A'][-1]
        pc = m['P_c'][-1]
        print(f"    {PROTOCOLS[proto_name]['label']:<25} P_c={pc:.3f}  "
              f"dP={imp:+.4f}  R_5HT2A={r:.3f}")
        if imp > best_improvement:
            best_improvement = imp
            best_proto = proto_name

print(f"\n  Best protocol: {PROTOCOLS[best_proto]['label']} "
      f"(dP={best_improvement:+.4f})")
print(f"\n  Figures saved to: {OUT_DIR}")
print("Done!")
