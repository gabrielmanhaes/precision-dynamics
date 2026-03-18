"""
Precision Dynamics — ODE implementation of the CANAL framework
(Carhart-Harris et al. 2022, Neuropharmacology 226, 109398)

Addresses the dynamical methods gap identified in CANAL Box 1:
'Dynamical methods will be required to better capture the emergence
of different free-energy minimizing processes.'

Key variables:
  P_sensory    — precision of sensory processing (NE, ACh modulated)
  P_conceptual — precision of conceptual priors (NE, 5-HT modulated)
  P_selfmodel  — precision of self-model (most affected by psychedelics;
                  proposed as consciousness amplitude variable)

See README.md for full documentation and THEORY.md for relation to CANAL.
"""

import numpy as np
from model import simulate_v2
from validation_summary import BEST_FIT, CONSOLIDATED_FIXED


def main():
    """Run a 24-hour normal simulation and print key biomarker values."""
    params = {**CONSOLIDATED_FIXED, **BEST_FIT}
    sim_override = {k: v for k, v in params.items()
                    if k not in ('ALPHA_POWER_EXPONENT', 'LZW_EXPONENT')}

    print("Precision Dynamics — 24h Normal Simulation")
    print("=" * 50)
    print(f"Parameters: {len(BEST_FIT)} fitted + {len(CONSOLIDATED_FIXED)} fixed")
    print(f"dt = 0.1h\n")

    t, P, st, nm = simulate_v2(
        t_span=(6.0, 30.0), dt=0.1, seed=42,
        params_override=sim_override,
    )

    wake_mask = nm['sleep'] < 0.3
    nrem_mask = (nm['sleep'] > 0.5) & (nm['REM'] < 0.3)
    rem_mask = nm['REM'] > 0.3

    print("Precision levels (mean):")
    for level in ['sensory', 'conceptual', 'selfmodel']:
        w = np.mean(P[level][wake_mask])
        n = np.mean(P[level][nrem_mask]) if np.any(nrem_mask) else 0.0
        r = np.mean(P[level][rem_mask]) if np.any(rem_mask) else 0.0
        print(f"  P_{level:<12s}  wake={w:.3f}  NREM={n:.3f}  REM={r:.3f}")

    print(f"\nNeuromodulators (waking mean):")
    for nm_name in ['NE', '5-HT', 'DA', 'ACh']:
        val = np.mean(nm[nm_name][wake_mask])
        print(f"  {nm_name:<4s} = {val:.3f}")

    print(f"\nHPA axis (waking mean):")
    print(f"  cortisol = {np.mean(st['cortisol'][wake_mask]):.3f}")
    print(f"  allostatic_load = {np.mean(st['allostatic_load'][wake_mask]):.3f}")

    cort = st['cortisol']
    dawn_mask = (t >= 6.5) & (t <= 8.0)
    nadir_mask = (t >= 23.0) & (t <= 25.0)
    dawn_val = np.mean(cort[dawn_mask]) if np.any(dawn_mask) else 0.0
    nadir_val = np.mean(cort[nadir_mask]) if np.any(nadir_mask) else 0.01
    print(f"  dawn/nadir ratio = {dawn_val / nadir_val:.2f}")

    print(f"\nSimulation complete: {len(t)} timepoints, "
          f"{t[-1] - t[0]:.0f}h span")
    print("See validation_summary.py for full target comparison.")


if __name__ == '__main__':
    main()
