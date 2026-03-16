"""
Task 1D: Parameter Sensitivity Analysis
Vary each effective (non-fitted) parameter by ±20% and measure impact on targets.
Parameters with negligible impact (<1% max change) are "phantom DoF".
"""
import sys
import numpy as np
import importlib
from fitting_v3 import build_targets, evaluate_model

# Best-fit values for the 14 fitted parameters
BEST_FIT = {
    'ALPHA_NE': 0.0500,
    'ALPHA_5HT': 0.0500,
    'BETA_PLAST': 1.2000,
    'GAMMA_SENSORY': 0.7215,
    'ALPHA_POWER_EXPONENT': 2.7136,
    'LZW_EXPONENT': 0.4331,
    'P_CONCEPTUAL_NREM': 0.5878,
    'CORTISOL_STRESS_GAIN': 1.4606,
    'GABA_NE_GAIN_MOD': 0.4764,
    'PSILOCYBIN_PHARMA_GAIN': 0.4333,
    'KETAMINE_PHARMA_GAIN': 5.0000,
    'PTSD_DISSOC_COEFF': 0.2211,
    'ALPHA_NE_PHASIC': 1.0000,
    'ALPHA_5HT_PHASIC': 0.5960,
}

# Non-fitted parameters that go through simulate_v2's params_override
# (coupling, SC, etc.) - these CAN be passed via params_dict
SIM_OVERRIDEABLE = {
    'COUPLING_TD': 0.15,
    'COUPLING_BU': 0.10,
    'CORT_NE_COUPLING': 0.20,
    'CORT_5HT_COUPLING': -0.10,
    'CORT_DA_COUPLING': -0.30,
    'CORT_PLASTICITY_COUPLING': -0.25,
    'ALPHA_DA': 0.15,
    'BETA_ACH': 0.30,
    'BETA_GLU': 0.20,
    'K_CONSOLIDATION': 0.10,
    'SC_CONSOLIDATION_TAU': 336.0,
    'GAMMA_CONCEPTUAL': 0.80,
    'GAMMA_SELFMODEL': 0.65,
}

# Operationalization params — used in model functions imported by fitting_v3
# These need module-level reload to take effect
OPERATIONALIZATION = {
    'HRV_NE_GAIN': 1.0,
    'HRV_CORT_GAIN': 0.5,
    'PUPIL_NE_GAIN': 0.5,
    'BDNF_EXPONENT': 0.8,
}


def run_with_extra_params(extra_params, targets):
    """Run model with best-fit + extra non-fitted params in override."""
    params = dict(BEST_FIT)
    params.update(extra_params)
    return evaluate_model(params, targets, dt=0.1)


def run_with_operationalization_change(param_name, new_value, targets):
    """For operationalization params, we need to reload modules."""
    import parameters as pmod
    import model as p2mod

    # Save originals
    orig_pmod = getattr(pmod, param_name)
    orig_p2mod = getattr(p2mod, param_name, None)

    # Set new values in both modules
    setattr(pmod, param_name, new_value)
    if orig_p2mod is not None:
        setattr(p2mod, param_name, new_value)

    # Also update in fitting_v3 namespace
    import fitting_v3 as f3mod
    orig_f3 = getattr(f3mod, param_name, None)
    if orig_f3 is not None:
        setattr(f3mod, param_name, new_value)

    try:
        preds = evaluate_model(BEST_FIT, targets, dt=0.1)
    finally:
        # Restore
        setattr(pmod, param_name, orig_pmod)
        if orig_p2mod is not None:
            setattr(p2mod, param_name, orig_pmod)
        if orig_f3 is not None:
            setattr(f3mod, param_name, orig_f3)

    return preds


def max_delta(baseline, varied, target_names):
    """Compute max % change and list of affected targets."""
    max_pct = 0.0
    affected = []
    for name in target_names:
        bval = baseline.get(name, 0.0)
        vval = varied.get(name, 0.0)
        if abs(bval) > 1e-6:
            pct = abs(vval - bval) / abs(bval) * 100.0
        elif abs(vval) > 1e-6:
            pct = 100.0  # went from ~0 to nonzero
        else:
            pct = 0.0

        if pct > max_pct:
            max_pct = pct
        if pct > 1.0 and name not in affected:
            affected.append((name, pct))

    affected.sort(key=lambda x: -x[1])
    return max_pct, affected


def main():
    targets = build_targets(include_theoretical=False)
    target_names = [t.name for t in targets]

    print("=" * 90)
    print("PARAMETER SENSITIVITY ANALYSIS — ±20% variation")
    print("=" * 90)

    # Baseline predictions
    print("\nRunning baseline...")
    baseline = run_with_extra_params({}, targets)

    print(f"\nBaseline predictions ({len(target_names)} targets):")
    for name in target_names:
        print(f"  {name:35s} = {baseline.get(name, 0.0):.4f}")

    results = []  # (name, max_delta_pct, status, affected_str)

    # 1. Test sim-overrideable params (coupling, SC, etc.)
    print(f"\n{'='*90}")
    print(f"{'Parameter':<35} {'Max Δ (%)':>10} {'Status':>10} {'Top affected targets'}")
    print(f"{'='*90}")

    for param_name, base_value in SIM_OVERRIDEABLE.items():
        if abs(base_value) < 1e-10:
            results.append((param_name, 0.0, "SKIP", "zero value"))
            continue

        overall_max = 0.0
        overall_affected = []

        for factor in [0.8, 1.2]:
            extra = {param_name: base_value * factor}
            preds = run_with_extra_params(extra, targets)
            md, aff = max_delta(baseline, preds, target_names)
            if md > overall_max:
                overall_max = md
            for a in aff:
                if a[0] not in [x[0] for x in overall_affected]:
                    overall_affected.append(a)

        overall_affected.sort(key=lambda x: -x[1])
        status = "PHANTOM" if overall_max < 1.0 else "ACTIVE"
        aff_str = ", ".join(f"{n}({p:.1f}%)" for n, p in overall_affected[:3]) or "(none)"
        results.append((param_name, overall_max, status, aff_str))
        print(f"  {param_name:<33} {overall_max:>8.2f}%  {status:>8s}  {aff_str}")

    # 2. Test operationalization params
    print(f"\n--- Operationalization Parameters ---")

    for param_name, base_value in OPERATIONALIZATION.items():
        if abs(base_value) < 1e-10:
            results.append((param_name, 0.0, "SKIP", "zero value"))
            continue

        overall_max = 0.0
        overall_affected = []

        for factor in [0.8, 1.2]:
            preds = run_with_operationalization_change(
                param_name, base_value * factor, targets)
            md, aff = max_delta(baseline, preds, target_names)
            if md > overall_max:
                overall_max = md
            for a in aff:
                if a[0] not in [x[0] for x in overall_affected]:
                    overall_affected.append(a)

        overall_affected.sort(key=lambda x: -x[1])
        status = "PHANTOM" if overall_max < 1.0 else "ACTIVE"
        aff_str = ", ".join(f"{n}({p:.1f}%)" for n, p in overall_affected[:3]) or "(none)"
        results.append((param_name, overall_max, status, aff_str))
        print(f"  {param_name:<33} {overall_max:>8.2f}%  {status:>8s}  {aff_str}")

    # Summary
    phantom = [r for r in results if r[2] == "PHANTOM"]
    active = [r for r in results if r[2] == "ACTIVE"]

    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"  Total non-fitted params tested: {len(results)}")
    print(f"  Phantom DoF (max Δ < 1%): {len(phantom)}")
    for p in phantom:
        print(f"    {p[0]}")
    print(f"  Active DoF (max Δ ≥ 1%): {len(active)}")
    for a in active:
        print(f"    {a[0]} — max {a[1]:.1f}% — {a[3]}")

    # Per-scenario params — separately counted
    print(f"\n  Per-scenario hardcoded params (always ACTIVE): 15")
    print(f"    (dep_chronic_stress, psy_da_excess, psy_gaba_deficit, psy_chronic_stress,")
    print(f"     adhd_dat_dysfunction, adhd_net_dysfunction, adhd_noise_scale, adhd_gamma_scale,")
    print(f"     ptsd_ne_sensitization, ptsd_coupling_breakdown, ptsd_chronic_stress,")
    print(f"     psi_strength, psi_noise_scale, ket_strength, prop_gaba_boost)")

    n_phantom = len(phantom)
    print(f"\n  Original effective DoF estimate: 29-34")
    print(f"  = 14 fitted + {len(results)} coupling/operationalization/SC + 15 scenario")
    print(f"  Phantom DoF to subtract: {n_phantom}")
    revised_low = 14 + len(active) + 15  # fitted + active non-fitted + scenario
    revised_high = revised_low  # same since we tested all non-scenario params
    print(f"  Revised effective DoF: {revised_low}")
    print(f"  Revised constraint ratio: 19/{revised_low} = {19/revised_low:.2f}")


if __name__ == '__main__':
    main()
