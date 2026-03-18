"""
Sensitivity Analysis: Robustness of Bidirectional ATX vs Psilocybin Prediction
===============================================================================

Tests whether the 6/6 opposing biomarker directions between atomoxetine and
psilocybin are robust to parameter uncertainty in the sloppy parameter subspace.

Two analyses:
1. Individual sweeps: vary each parameter ±50% in 10 steps
2. Joint sweep: 1000 random perturbations of all 9 parameters ±30%
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import simulate_v2, p_to_eeg_alpha, p_to_lzw, ne_cort_to_hrv
from validation_summary import BEST_FIT, CONSOLIDATED_FIXED
from parameters import ALPHA_POWER_EXPONENT, LZW_EXPONENT

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def run_bidirectional_test(params_dict, fixed_override):
    """Run ATX and psilocybin simulations, return direction deltas for 6 biomarkers."""
    sim_override = {k: v for k, v in {**fixed_override, **params_dict}.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT')}
    alpha_exp = params_dict.get('ALPHA_POWER_EXPONENT', ALPHA_POWER_EXPONENT)
    lzw_exp = params_dict.get('LZW_EXPONENT', LZW_EXPONENT)

    window = (15.0, 17.0)

    try:
        # Baseline
        t_b, P_b, st_b, nm_b = simulate_v2(
            t_span=(6.0, 30.0), dt=0.1, seed=42, params_override=sim_override)
        mask_b = (t_b >= window[0]) & (t_b <= window[1])

        # ATX
        t_a, P_a, st_a, nm_a = simulate_v2(
            t_span=(6.0, 30.0), dt=0.1, seed=42,
            pharma_atomoxetine=[(14.0, 0.5)], params_override=sim_override)
        mask_a = (t_a >= window[0]) & (t_a <= window[1])

        # Psilocybin
        t_p, P_p, st_p, nm_p = simulate_v2(
            t_span=(6.0, 30.0), dt=0.1, seed=42,
            pharma_psilocybin=[(14.0, 0.6)], params_override=sim_override)
        mask_p = (t_p >= window[0]) & (t_p <= window[1])

        def extract(P, st, nm, mask):
            Pc = np.mean(P['conceptual'][mask])
            Ps = np.mean(P['sensory'][mask])
            Psm = np.mean(P['selfmodel'][mask])
            alpha = p_to_eeg_alpha(Pc, exponent=alpha_exp)
            lzw = p_to_lzw(Pc, exponent=lzw_exp)
            ne = np.mean(nm['NE'][mask])
            cort = np.mean(st['cortisol'][mask])
            hrv = ne_cort_to_hrv(ne, cort)
            return {'P_c': Pc, 'P_s': Ps, 'P_sm': Psm,
                    'alpha': alpha, 'lzw': lzw, 'hrv': hrv}

        base = extract(P_b, st_b, nm_b, mask_b)
        atx = extract(P_a, st_a, nm_a, mask_a)
        psi = extract(P_p, st_p, nm_p, mask_p)

        # Check opposing directions for dynamical biomarkers
        # These are the 3 non-tautological markers from validation_summary.py
        dynamical = ['P_c', 'alpha', 'lzw']
        correct = 0
        for m in dynamical:
            d_atx = atx[m] - base[m]
            d_psi = psi[m] - base[m]
            # Opposing means different signs (and both nonzero)
            if abs(d_atx) > 1e-6 and abs(d_psi) > 1e-6:
                if np.sign(d_atx) != np.sign(d_psi):
                    correct += 1
        return correct
    except Exception:
        return 0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    param_names = list(BEST_FIT.keys())
    param_values = {k: v for k, v in BEST_FIT.items()}
    fixed = dict(CONSOLIDATED_FIXED)

    print("SENSITIVITY ANALYSIS: Bidirectional ATX vs Psilocybin Prediction")
    print("=" * 65)

    # --- Individual parameter sweeps ---
    print("\nIndividual parameter sweeps (+/-50%, 10 steps each):")
    print(f"{'Parameter':<25} | {'Min deviation causing failure':>30} | {'Robust?':>7}")
    print("-" * 68)

    individual_results = {}
    for pname in param_names:
        best_val = param_values[pname]
        fail_pct = None

        for frac in np.linspace(0.5, 1.5, 21):
            if abs(frac - 1.0) < 0.01:
                continue
            test_params = dict(param_values)
            test_params[pname] = best_val * frac
            n_correct = run_bidirectional_test(test_params, fixed)
            if n_correct < 3:
                deviation = abs(frac - 1.0) * 100
                if fail_pct is None or deviation < fail_pct:
                    fail_pct = deviation

        if fail_pct is None:
            print(f"  {pname:<23} | {'Never':>30} |     Yes")
            individual_results[pname] = ('Never', True)
        else:
            print(f"  {pname:<23} | {fail_pct:>28.0f}% |      No")
            individual_results[pname] = (f'{fail_pct:.0f}%', False)

    # --- Joint parameter sweep ---
    print(f"\nJoint parameter sweep (N=1000, +/-30% all parameters):")
    n_samples = 1000
    n_correct_total = 0
    rng = np.random.RandomState(42)

    for i in range(n_samples):
        test_params = {}
        for pname in param_names:
            scale = rng.uniform(0.7, 1.3)
            test_params[pname] = param_values[pname] * scale
        n_correct = run_bidirectional_test(test_params, fixed)
        if n_correct == 3:
            n_correct_total += 1
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples} "
                  f"({n_correct_total}/{i+1} = {n_correct_total/(i+1)*100:.0f}%)",
                  flush=True)

    pct = n_correct_total / n_samples * 100
    print(f"\nFraction maintaining 3/3 correct directions: "
          f"{n_correct_total}/{n_samples} ({pct:.1f}%)")

    # Find most/least sensitive
    robust_params = [p for p, (v, r) in individual_results.items() if r]
    fragile_params = [(p, v) for p, (v, r) in individual_results.items() if not r]

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'sensitivity_analysis_results.txt')
    with open(results_path, 'w') as f:
        f.write("SENSITIVITY ANALYSIS: Bidirectional ATX vs Psilocybin Prediction\n")
        f.write("=" * 65 + "\n\n")
        f.write("Individual parameter sweeps (+/-50%, 21 steps each):\n")
        f.write(f"{'Parameter':<25} | {'Min deviation causing failure':>30} | {'Robust?':>7}\n")
        f.write("-" * 68 + "\n")
        for pname in param_names:
            v, r = individual_results[pname]
            f.write(f"  {pname:<23} | {v:>30} | {'Yes' if r else 'No':>7}\n")
        f.write(f"\nJoint parameter sweep (N={n_samples}, +/-30% all parameters):\n")
        f.write(f"Fraction maintaining 3/3 correct directions: "
                f"{n_correct_total}/{n_samples} ({pct:.1f}%)\n")
        f.write(f"\nRobust across full +/-50%: {', '.join(robust_params) if robust_params else 'None'}\n")
        if fragile_params:
            most_sens = min(fragile_params, key=lambda x: float(x[1].rstrip('%')))
            f.write(f"Most sensitive: {most_sens[0]} (fails at +/-{most_sens[1]})\n")

    print(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
