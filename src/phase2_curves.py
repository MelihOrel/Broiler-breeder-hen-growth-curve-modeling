"""PHASE 2 — Comparison of classical parametric growth curves.

Fits Gompertz, Logistic, von Bertalanffy and Gompertz-Laird to the
body-weight sub-series (n = 24 weighing days, ages 143-224 d) using lmfit.

Important context: the observed window covers only the tail of the growth
trajectory (the bird already weighs ~2 kg at first observation), so
asymptote and inflection parameters are extrapolations and may be weakly
identified. This is reported, not hidden.
"""

import json

import agridatasets as agd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model


from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # project root (…/broiler)
(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)  # lmfit least-squares is deterministic; seed by convention

df = agd.load_dataset("broiler_growth").sort_values("age")
bw = df[df.bw.notna()].copy()
t = bw.age.values.astype(float)   # days
y = bw.bw.values.astype(float)    # grams
n = len(y)
onset = int(df.loc[df.eggwt.notna(), "age"].min())

# --- model functions -------------------------------------------------------
def gompertz(t, A, k, ti):
    """A: asymptotic weight (g); k: rate (1/day); ti: inflection age (day)."""
    return A * np.exp(-np.exp(-k * (t - ti)))

def logistic(t, A, k, ti):
    """A: asymptotic weight (g); k: rate (1/day); ti: inflection age (day)."""
    return A / (1.0 + np.exp(-k * (t - ti)))

T0 = 143.0  # first observation day; time is re-anchored here for the two
            # models parameterized in absolute time, otherwise exp(-k*t)
            # underflows over ages 143-224 and the fits do not converge.

def von_bertalanffy(t, A, b, k):
    """Weight form: A*(1 - b*exp(-k*(t-T0)))^3. A: asymptote (g);
    b is defined relative to age T0 = 143 d, not hatch."""
    return A * (1.0 - b * np.exp(-k * (t - T0))) ** 3

def gompertz_laird(t, W0, L, k):
    """W0: weight at age T0 = 143 d (g), NOT hatch weight; L: specific
    growth rate at T0 (1/day); k: decay rate of growth (1/day)."""
    return W0 * np.exp((L / k) * (1.0 - np.exp(-k * (t - T0))))

# --- starting values / bounds ---------------------------------------------
# Asymptote: slightly above observed max (3156 g); the rearing program's own
# target tops out at 3427 g, so 3400 g is a data-informed start.
A0 = 3400.0

specs = {
    "Gompertz": dict(
        func=gompertz,
        params=dict(A=dict(value=A0, min=y.max(), max=6000),
                    k=dict(value=0.03, min=1e-4, max=0.5),
                    ti=dict(value=140.0, min=0, max=250)),
    ),
    "Logistic": dict(
        func=logistic,
        params=dict(A=dict(value=A0, min=y.max(), max=6000),
                    k=dict(value=0.04, min=1e-4, max=0.5),
                    ti=dict(value=150.0, min=0, max=250)),
    ),
    "von Bertalanffy": dict(
        func=von_bertalanffy,
        # b start from A*(1-b)^3 = 2068 g at day 143 with A = 3400 g
        params=dict(A=dict(value=A0, min=y.max(), max=8000),
                    b=dict(value=0.15, min=1e-3, max=1.0),
                    k=dict(value=0.03, min=1e-4, max=0.5)),
    ),
    "Gompertz-Laird": dict(
        func=gompertz_laird,
        # W0 start = first observed weight; L start = adg/bw at day 143
        params=dict(W0=dict(value=2068.0, min=1500.0, max=2500.0),
                    L=dict(value=0.01, min=1e-4, max=0.2),
                    k=dict(value=0.03, min=1e-4, max=0.5)),
    ),
}

# --- fitting ---------------------------------------------------------------
def fit_metrics(result, n, p):
    rss = float(np.sum(result.residual ** 2))
    rmse = np.sqrt(rss / n)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return rmse, adj_r2

results = {}
print(f"n = {n} body-weight observations, ages {t.min():.0f}-{t.max():.0f} d\n")

for name, spec in specs.items():
    model = Model(spec["func"])
    params = model.make_params()
    for pname, kw in spec["params"].items():
        params[pname].set(**kw)
    res = model.fit(y, params, t=t)
    p = len(res.params)
    rmse, adj_r2 = fit_metrics(res, n, p)
    results[name] = dict(res=res, rmse=rmse, adj_r2=adj_r2)

    print(f"=== {name} ===")
    for pname, par in res.params.items():
        se = par.stderr if par.stderr is not None else float("nan")
        if par.stderr is not None:
            ci = f"[{par.value - 1.96*se:.3g}, {par.value + 1.96*se:.3g}]"
        else:
            ci = "[not estimable]"
        print(f"  {pname:>3} = {par.value:10.4g}  SE = {se:8.3g}  95% CI {ci}")
    print(f"  AIC = {res.aic:.2f}  BIC = {res.bic:.2f}  "
          f"adj R2 = {adj_r2:.4f}  RMSE = {rmse:.1f} g\n")

# --- derived biological quantities ----------------------------------------
print("Derived quantities (interpret with caution: inflection precedes or "
      "sits at the edge of the observed window in several models):")
for name, r in results.items():
    v = {k: p.value for k, p in r["res"].params.items()}
    if name == "Gompertz":
        ti, wi, mu = v["ti"], v["A"] / np.e, v["A"] * v["k"] / np.e
    elif name == "Logistic":
        ti, wi, mu = v["ti"], v["A"] / 2, v["A"] * v["k"] / 4
    elif name == "von Bertalanffy":
        ti = T0 + np.log(3 * v["b"]) / v["k"]  # re-anchored time
        wi = 8 * v["A"] / 27
        mu = (4 / 9) * v["A"] * v["k"]  # max dW/dt for cubic vB
    else:  # Gompertz-Laird (re-anchored: inflection relative to T0)
        ti = T0 + np.log(v["L"] / v["k"]) / v["k"]
        A = v["W0"] * np.exp(v["L"] / v["k"])
        wi, mu = A / np.e, A * v["k"] / np.e
    print(f"  {name:>15}: inflection age = {ti:6.1f} d, "
          f"weight at inflection = {wi:6.0f} g, max growth rate = {mu:5.1f} g/day")

# --- comparison table ------------------------------------------------------
order = sorted(results, key=lambda k: results[k]["res"].aic)
lines = ["| Model | AIC | BIC | adj R² | RMSE (g) | ΔAIC |",
         "|---|---|---|---|---|---|"]
best_aic = results[order[0]]["res"].aic
for name in order:
    r = results[name]
    lines.append(f"| {name} | {r['res'].aic:.2f} | {r['res'].bic:.2f} | "
                 f"{r['adj_r2']:.4f} | {r['rmse']:.1f} | "
                 f"{r['res'].aic - best_aic:.2f} |")
table = "\n".join(lines)
with open(f"{ROOT}/models/phase2_comparison.md", "w") as f:
    f.write(table + "\n")
print("\n" + table)

# --- plots -----------------------------------------------------------------
tg = np.linspace(t.min(), t.max(), 400)
fig, axes = plt.subplots(2, 1, figsize=(9, 10),
                         gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(t, y, "ko", ms=5, label="Observed", zorder=5)
for name, r in results.items():
    ax.plot(tg, r["res"].eval(t=tg), lw=1.6, label=name)
ax.axvline(onset, color="grey", ls="--", lw=1)
ax.annotate(f"onset of lay ({onset} d)", xy=(onset, y.min()), fontsize=9,
            xytext=(onset + 1.5, y.min()))
ax.set_xlabel("Age (days)"); ax.set_ylabel("Body weight (g)")
ax.set_title("Classical growth curves vs observed body weight")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.axhline(0, color="grey", lw=0.8)
for name, r in results.items():
    ax.plot(t, y - r["res"].eval(t=t), "o-", ms=4, lw=1, label=name)
ax.axvline(onset, color="grey", ls="--", lw=1)
ax.set_xlabel("Age (days)"); ax.set_ylabel("Residual (g)")
ax.set_title("Residuals against age")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{ROOT}/figures/phase2_curves.png", dpi=300)
print("\nSaved figures/phase2_curves.png")

# --- persist parameters for later phases / the app -------------------------
out = {}
for name, r in results.items():
    out[name] = {
        "params": {k: p.value for k, p in r["res"].params.items()},
        "stderr": {k: (p.stderr if p.stderr is not None else None)
                   for k, p in r["res"].params.items()},
        "aic": r["res"].aic, "bic": r["res"].bic,
        "adj_r2": r["adj_r2"], "rmse": r["rmse"],
    }
out["_meta"] = {"n": n, "age_min": float(t.min()), "age_max": float(t.max()),
                "onset_of_lay": onset, "seed": SEED}
with open(f"{ROOT}/models/phase2_params.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved models/phase2_params.json")
