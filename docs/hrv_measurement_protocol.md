# HRV Measurement Protocol for Precision Biomarker Assessment

## Rationale

Baseline HRV is proposed as a proxy for canalization depth (precision
level) in the precision dynamics framework. For this operationalization
to be valid as a treatment selection biomarker, HRV must be measured
under standardized conditions that minimize known confounds.

## Known confounds and controls

| Confound | Effect | Control |
|----------|--------|---------|
| Breathing rate | Strong — slow breathing artificially elevates HRV | Paced breathing at 0.1 Hz (6 breaths/min) for 5-minute measurement OR measure free-breathing with concurrent respiratory monitoring |
| Time of day | Moderate — circadian variation ~15% | Standardize to morning measurement, 30-60 min after waking |
| Recent exercise | Strong — elevates HRV for hours | No vigorous exercise 12h before measurement |
| Recent caffeine | Moderate — reduces HRV | No caffeine 4h before measurement |
| Recent alcohol | Strong — reduces HRV for 24h | No alcohol 24h before measurement |
| Posture | Moderate — supine vs standing | Standardize to supine, 5 min rest before measurement |
| Age | Strong systematic — HRV declines with age | Use age-corrected normative values for cutoff |
| Cardiac medication | Strong — beta-blockers, SSRIs alter HRV directly | Document and control statistically |
| Current antidepressant | Moderate — SSRIs increase HRV over weeks | Measure at baseline before any pharmacological intervention |

## Recommended protocol for clinical research

1. **Timing:** Morning, 30-60 minutes after waking, before caffeine
2. **Posture:** Supine, 5 minutes rest before measurement begins
3. **Duration:** Minimum 5-minute recording; 24-hour ambulatory preferred
4. **Breathing:** If 5-minute: paced at 6 breaths/min (0.1 Hz)
   If 24-hour: free breathing with actigraphy
5. **Metric:** RMSSD (root mean square of successive differences) —
   less sensitive to breathing confounds than frequency-domain HF-HRV
6. **Timing relative to treatment:** Measure >=2 weeks before any
   pharmacological intervention while medication-free if possible

## Proposed treatment selection cutoff

**Research proposal (not clinical guidance):**
- High-HRV (RMSSD > age-adjusted median): predicted better psilocybin
  response — sufficient residual autonomic flexibility suggesting lower
  canalization depth
- Low-HRV (RMSSD < age-adjusted median): predicted better SSRI response —
  autonomic rigidity suggesting deeper canalization requiring gradual
  recalibration

**Validation requirement:** This cutoff is a theoretical prediction
requiring prospective validation in treatment-naive depressed patients
with HRV measured before treatment assignment. No clinical decisions
should be made based on this prediction until validated.

## Convergent validity design

HRV alone is a noisy proxy for cortical precision. Convergent validity
requires HRV to correlate with independent precision proxies:

1. **Pupil dilation variability** — NE-linked precision signal, measurable
   with eye-tracking; should correlate with HRV at r > 0.3
2. **EEG alpha power variability** — thalamocortical precision marker;
   SD(alpha) should correlate with RMSSD
3. **P300 amplitude** — event-related precision signal; should correlate
   inversely with HRV in depression (high P -> large P300 -> low HRV)

If three independent physiological signals load on a single latent factor
that predicts treatment response, the HRV precision biomarker claim becomes
a convergent construct rather than a single noisy proxy.

## Limitations

- HRV reflects autonomic nervous system balance, not cortical precision
  directly. The chain: cortical P -> NE -> sympathetic activation -> HRV
  involves multiple noisy, nonlinear steps
- HRV confounded by cardiac disease, diabetes, and other autonomic
  neuropathies — these patient populations require separate consideration
- The proposed cutoff (median split) is exploratory; optimal threshold
  likely varies by age, sex, and comorbidity
- This protocol has not been validated; it is a research proposal
