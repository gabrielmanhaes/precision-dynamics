"""
Profile Likelihood Analysis for Precision Dynamics Model
=========================================================

For each of the 9 fitted parameters, fix it at grid points across its
bounds while re-optimizing all other 8 parameters. Plot the chi² profile.

Parameters with a clear minimum (chi² rising steeply on both sides) are
identifiable. Parameters with flat profiles are sloppy/unidentifiable.

Strategy: Use dt=0.5 for fast inner optimization (shape discovery), then
verify key points at dt=0.1 (publication-quality values).

Reference: Raue et al. (2009) Bioinformatics 25(15):1923-1929

Run:
    python3 profile_likelihood.py           # Full analysis (~1-2 hours)
    python3 profile_likelihood.py --quick   # 9 grid points (~20-30 min)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitting_v3 import (
    build_targets, objective, evaluate_model,
    CONSOLIDATED_FIXED_VALUES, FITTED_PARAMS_CONSOLIDATED,
)
from validation_summary import BEST_FIT, CONSOLIDATED_FIXED

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'figures', 'profile_likelihood')


def _profile_one_point(args):
    """Worker function for a single grid point optimization."""
    (param_name, fixed_val, param_names, best_vector, idx,
     free_indices, free_bounds, free_start, dt) = args

    targets = build_targets(include_theoretical=False)
    fixed_override = dict(CONSOLIDATED_FIXED)

    def obj_fixed(free_vector):
        full = np.array(best_vector, dtype=float)
        full[idx] = fixed_val
        for j, fi in enumerate(free_indices):
            full[fi] = free_vector[j]
        return objective(full, param_names, targets, dt=dt,
                         fixed_override=fixed_override)

    # Try Powell (no gradients needed, fast for bounded problems)
    result = minimize(obj_fixed, free_start, method='Powell',
                      options={'maxiter': 200, 'ftol': 1e-6})

    # Clip to bounds and refine with Nelder-Mead
    x_clipped = np.clip(result.x,
                        [b[0] for b in free_bounds],
                        [b[1] for b in free_bounds])
    result2 = minimize(obj_fixed, x_clipped, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6})

    best_result = result2 if result2.fun < result.fun else result
    return fixed_val, best_result.fun, best_result.x


def profile_likelihood_fast(param_name, param_names, best_vector, bounds_dict,
                            targets, fixed_override, n_points=15, dt=0.5):
    """Compute profile likelihood for one parameter using fast dt.

    Uses sequential sweep (warm-starting from previous grid point).
    """
    idx = param_names.index(param_name)
    lo, hi = bounds_dict[param_name]

    # Put best-fit value in the grid for reference
    grid = np.linspace(lo, hi, n_points)

    chi2_values = np.zeros(n_points)

    free_names = [n for n in param_names if n != param_name]
    free_indices = [param_names.index(n) for n in free_names]
    free_bounds = [(bounds_dict[n][0], bounds_dict[n][1]) for n in free_names]
    free_start = np.array([best_vector[i] for i in free_indices])

    # Find the grid point closest to best-fit and sweep outward from there
    best_val = best_vector[idx]
    center_idx = np.argmin(np.abs(grid - best_val))

    # Sweep right from center
    start = free_start.copy()
    for k in range(center_idx, n_points):
        fixed_val = grid[k]

        def obj_fixed(free_vector, fv=fixed_val):
            full = np.array(best_vector, dtype=float)
            full[idx] = fv
            for j, fi in enumerate(free_indices):
                full[fi] = free_vector[j]
            return objective(full, param_names, targets, dt=dt,
                             fixed_override=fixed_override)

        result = minimize(obj_fixed, start, method='Powell',
                          options={'maxiter': 200, 'ftol': 1e-6})
        x_clipped = np.clip(result.x,
                            [b[0] for b in free_bounds],
                            [b[1] for b in free_bounds])
        result2 = minimize(obj_fixed, x_clipped, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
        best_r = result2 if result2.fun < result.fun else result

        chi2_values[k] = best_r.fun
        start = np.clip(best_r.x,
                        [b[0] for b in free_bounds],
                        [b[1] for b in free_bounds])

        print(f"  {param_name}: [{k+1}/{n_points}] "
              f"val={fixed_val:.4f} chi2={best_r.fun:.4f}", flush=True)

    # Sweep left from center
    start = free_start.copy()
    for k in range(center_idx - 1, -1, -1):
        fixed_val = grid[k]

        def obj_fixed(free_vector, fv=fixed_val):
            full = np.array(best_vector, dtype=float)
            full[idx] = fv
            for j, fi in enumerate(free_indices):
                full[fi] = free_vector[j]
            return objective(full, param_names, targets, dt=dt,
                             fixed_override=fixed_override)

        result = minimize(obj_fixed, start, method='Powell',
                          options={'maxiter': 200, 'ftol': 1e-6})
        x_clipped = np.clip(result.x,
                            [b[0] for b in free_bounds],
                            [b[1] for b in free_bounds])
        result2 = minimize(obj_fixed, x_clipped, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
        best_r = result2 if result2.fun < result.fun else result

        chi2_values[k] = best_r.fun
        start = np.clip(best_r.x,
                        [b[0] for b in free_bounds],
                        [b[1] for b in free_bounds])

        print(f"  {param_name}: [{n_points - k}/{n_points}] "
              f"val={fixed_val:.4f} chi2={best_r.fun:.4f}", flush=True)

    return grid, chi2_values


def refine_at_dt01(param_name, grid, chi2_fast, param_names, best_vector,
                   bounds_dict, targets, fixed_override):
    """Re-evaluate key points (min, endpoints, steepest) at dt=0.1."""
    idx = param_names.index(param_name)
    free_names = [n for n in param_names if n != param_name]
    free_indices = [param_names.index(n) for n in free_names]
    free_bounds = [(bounds_dict[n][0], bounds_dict[n][1]) for n in free_names]
    free_start = np.array([best_vector[i] for i in free_indices])

    # Select key indices: min, endpoints, and 2 intermediate points
    imin = np.argmin(chi2_fast)
    key_indices = sorted(set([0, imin, len(grid)-1,
                              max(0, imin-2), min(len(grid)-1, imin+2)]))

    chi2_refined = chi2_fast.copy()
    for k in key_indices:
        fixed_val = grid[k]

        def obj_fixed(free_vector, fv=fixed_val):
            full = np.array(best_vector, dtype=float)
            full[idx] = fv
            for j, fi in enumerate(free_indices):
                full[fi] = free_vector[j]
            return objective(full, param_names, targets, dt=0.1,
                             fixed_override=fixed_override)

        result = minimize(obj_fixed, free_start, method='Powell',
                          options={'maxiter': 200, 'ftol': 1e-6})
        chi2_refined[k] = result.fun

    return chi2_refined


def classify_identifiability(grid, chi2, threshold=3.84):
    """Classify parameter as identifiable or sloppy.

    threshold=3.84 is the 95% confidence chi² increase for 1 DoF.
    """
    imin = np.argmin(chi2)
    chi2_min = chi2[imin]

    left_max = np.max(chi2[:imin+1]) - chi2_min if imin > 0 else 0.0
    right_max = np.max(chi2[imin:]) - chi2_min if imin < len(chi2) - 1 else 0.0

    identifiable = (left_max > threshold) and (right_max > threshold)
    return identifiable, chi2_min, left_max, right_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Use 9 grid points instead of 15')
    parser.add_argument('--points', type=int, default=15,
                        help='Number of grid points per parameter')
    parser.add_argument('--dt', type=float, default=0.5,
                        help='dt for inner optimizations (0.5=fast, 0.1=accurate)')
    args = parser.parse_args()

    n_points = 9 if args.quick else args.points

    os.makedirs(OUTDIR, exist_ok=True)

    targets = build_targets(include_theoretical=False)
    fixed_override = dict(CONSOLIDATED_FIXED)

    param_names = list(BEST_FIT.keys())
    best_vector = np.array([BEST_FIT[k] for k in param_names])

    bounds_dict = {}
    for name, default, lo, hi in FITTED_PARAMS_CONSOLIDATED:
        bounds_dict[name] = (lo, hi)

    # Baseline chi² at BEST_FIT (always at dt=0.1 for reference)
    chi2_best = objective(best_vector, param_names, targets, dt=0.1,
                          fixed_override=fixed_override)
    chi2_best_fast = objective(best_vector, param_names, targets, dt=args.dt,
                               fixed_override=fixed_override)
    print(f"Baseline chi² at BEST_FIT: {chi2_best:.4f} (dt=0.1), "
          f"{chi2_best_fast:.4f} (dt={args.dt})")
    print(f"Grid points per parameter: {n_points}")
    print(f"Inner optimization dt: {args.dt}")
    print(f"Total optimizations: {len(param_names)} x {n_points} = "
          f"{len(param_names) * n_points}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")

    results = {}
    for pname in param_names:
        t0 = datetime.now()
        print(f"Profiling {pname} [{t0.strftime('%H:%M:%S')}]...")

        grid, chi2 = profile_likelihood_fast(
            pname, param_names, best_vector, bounds_dict,
            targets, fixed_override, n_points=n_points, dt=args.dt)

        ident, chi2_min, left_rise, right_rise = classify_identifiability(
            grid, chi2)

        elapsed = (datetime.now() - t0).total_seconds()
        results[pname] = {
            'grid': grid.tolist(),
            'chi2': chi2.tolist(),
            'identifiable': bool(ident),
            'chi2_min': float(chi2_min),
            'left_rise': float(left_rise),
            'right_rise': float(right_rise),
            'best_fit_value': float(BEST_FIT[pname]),
            'bounds': list(bounds_dict[pname]),
        }

        status = "IDENTIFIABLE" if ident else "SLOPPY"
        print(f"  -> {status} (min={chi2_min:.2f}, "
              f"L+={left_rise:.2f}, R+={right_rise:.2f}) "
              f"[{elapsed:.0f}s]\n")

    # Save raw results
    with open(os.path.join(OUTDIR, 'profile_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Generate figure
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for i, pname in enumerate(param_names):
        ax = axes[i]
        r = results[pname]
        grid = np.array(r['grid'])
        chi2 = np.array(r['chi2'])

        ax.plot(grid, chi2, 'b.-', linewidth=1.5, markersize=4)
        ax.axvline(r['best_fit_value'], color='r', linestyle='--',
                    linewidth=1, alpha=0.7, label='Best fit')
        ax.axhline(r['chi2_min'] + 3.84, color='gray', linestyle=':',
                    linewidth=0.8, label='95% CI')

        status = "identifiable" if r['identifiable'] else "SLOPPY"
        color = 'green' if r['identifiable'] else 'red'
        ax.set_title(f"{pname}\n({status}, rise: L={r['left_rise']:.1f} "
                      f"R={r['right_rise']:.1f})",
                      fontsize=8, color=color)
        ax.set_xlabel('Parameter value', fontsize=8)
        ax.set_ylabel('chi-sq', fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, loc='upper right')
    n_ident = sum(1 for r in results.values() if r['identifiable'])
    fig.suptitle(f'Profile Likelihood Analysis -- 9 Fitted Parameters\n'
                 f'Baseline chi-sq={chi2_best:.2f} (dt=0.1), '
                 f'{n_ident}/9 identifiable, '
                 f'threshold=3.84 (95% CI, 1 DoF)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig_path = os.path.join(OUTDIR, 'profile_likelihood_9params.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")

    # Summary table
    print(f"\n{'='*75}")
    print("PROFILE LIKELIHOOD SUMMARY")
    print(f"{'='*75}")
    print(f"{'Parameter':<25} {'Status':<15} {'chi2_min':>8} "
          f"{'Left+':>8} {'Right+':>8} {'Best fit':>10}")
    print(f"{'-'*75}")
    n_ident = 0
    for pname in param_names:
        r = results[pname]
        status = "IDENTIFIABLE" if r['identifiable'] else "SLOPPY"
        if r['identifiable']:
            n_ident += 1
        print(f"  {pname:<23} {status:<15} {r['chi2_min']:>8.2f} "
              f"{r['left_rise']:>8.2f} {r['right_rise']:>8.2f} "
              f"{r['best_fit_value']:>10.4f}")
    print(f"{'-'*75}")
    print(f"  Identifiable: {n_ident}/{len(param_names)}")
    print(f"  Sloppy: {len(param_names) - n_ident}/{len(param_names)}")
    print(f"  (Threshold: delta-chi-sq > 3.84 on both sides = 95% CI for 1 DoF)")
    print(f"  Inner optimization dt: {args.dt}")
    print(f"\nCompleted: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
