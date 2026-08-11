# Broiler breeder hen growth curve modeling

Parametric growth-curve analysis of a single broiler breeder hen's
body-weight time series, with an explicit treatment of the growth stall
that appears at the onset of egg production. Interactive viewer built with
Streamlit; all models are fitted offline and the app only reads saved
parameters.

## Why mechanistic modeling, not machine learning

The dataset is a **single bird's time series**: 59 rows over ages 143–224
days, of which only **24 are weighing days** carrying body weight (the
remaining rows are near-daily egg-weight records once lay begins). There
are no replicates, pens, or individual IDs, so mixed models do not apply,
and an effective n of 24 cannot support machine learning, train/test
splits, or explainability tooling. Nonlinear least-squares fitting of
classical growth functions (`lmfit`) is the approach the data can carry —
this was a deliberate choice, not a limitation discovered late.

**Scope of the findings:** everything reported here describes this one
hen. No claim is made that any estimate generalizes to a flock, a strain,
or breeder hens at large.

## What the analysis found

1. **Verification (Phase 1).** Body weight rises steadily (~21–34 g/day)
   until just after the first egg (day 173), then drops ~71 g between days
   178 and 181 and plateaus through day 188, while feed intake keeps
   rising. The deviation from the rearing program's target weight collapses
   from −40 g to ~−300 g over the same window.
2. **Classical curves (Phase 2).** Gompertz, Logistic, von Bertalanffy and
   Gompertz-Laird are statistically indistinguishable on this window
   (ΔAIC ≤ 0.4, RMSE ≈ 43 g). All four leave the same structural residual
   pattern around onset of lay: a run of positive residuals (mean +67 g,
   days 167–178) followed by a run of negative ones (mean −49 g, days
   181–195).
3. **Correction (Phase 3).** A post-lay dummy level shift does **not**
   help (ΔAIC +1.7): the lay effect is a transient stall, not a permanent
   offset. A segmented model — Gompertz to a breakpoint, then an immediate
   drop and linear regrowth — with the breakpoint estimated by profile
   likelihood places the break at **day 181** (8 days after first egg),
   with a drop of **175 ± 27 g** and regrowth of **7.9 ± 0.5 g/day**
   (ΔAIC −31 vs base, RMSE halved to 20 g).
4. **Feed efficiency (Phase 4).** The intake–gain correlation collapses
   at onset of lay (+0.65 → +0.06), and intake runs above what
   maintenance + growth predict exactly when eggs appear — consistent with
   feed being diverted to egg production.

## Honest limitations (kept on purpose)

- The observed window covers only the **tail** of the growth trajectory,
  so asymptotes and inflection ages are extrapolations; inflection
  estimates fall before the first observation and should be read as
  model-implied only.
- In the segmented model the asymptote is **not identified** (estimate
  pinned at its bound with SE larger than the estimate); that model
  describes the trajectory but does not estimate mature weight.
- The estimated breakpoint was selected by profiling, which makes its AIC
  advantage mildly optimistic.
- Six effective parameters on 24 observations is thin (4 obs/parameter);
  model expansion was stopped deliberately.
- Von Bertalanffy and Gompertz-Laird required re-anchoring time at the
  first observation (day 143) to converge; in the Gompertz-Laird fit `W0`
  is therefore the weight at day 143, not hatch weight.

## Repository layout

```
src/          phase1_exploration.py … phase5_package.py  (analysis, in order)
figures/      all figures, 300 dpi
models/       comparison tables (md) and fitted parameters (json)
app/          app.py, requirements.txt, models/, data/   (Streamlit app)
```

Seeds are fixed (`SEED = 42`); the least-squares fits are deterministic.
Run phases in order; the app reads `app/models/app_params.json` and
performs no fitting.

## Citations

1. `agridatasets` Python package (PyPI), Renzo Caceres Rossi.
2. `agridat` R package (CRAN), Kevin Wright — dataset original name
   `zuidhof.broiler`.
3. Zuidhof, M.J., Schneider, B.L., Carney, V.L., Korver, D.R., Robinson,
   F.E. (2014). Growth, efficiency, and yield of commercial broilers from
   1957, 1978, and 2005. *Poultry Science*, 93(12), 2970–2982.
   DOI: 10.3382/ps.2014-04291
