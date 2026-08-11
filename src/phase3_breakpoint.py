"""PHASE 3 — Handling the onset-of-lay body-weight drop.

1. Shows the systematic residual pattern of the best PHASE 2 model
   (Gompertz) around onset of lay, and quantifies pre/post mean residuals.
2. Fits and compares:
   - Approach B: Gompertz + post-onset level shift (dummy), 4 params
   - Approach A: segmented model (Gompertz pre-break; immediate drop +
     linear regrowth post-break), breakpoint fixed at 173 d (5 params)
     and estimated by profile likelihood over a grid (6 effective params)

n = 24 body-weight observations: parameter counts are kept deliberately
low and the small-sample caveat is stated in the report.
"""

import json

import agridatasets as agd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters, minimize


from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # project root (…/broiler)
(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

df = agd.load_dataset("broiler_growth").sort_values("age")
bw = df[df.bw.notna()].copy()
t = bw.age.values.astype(float)
y = bw.bw.values.astype(float)
n = len(y)
ONSET = int(df.loc[df.eggwt.notna(), "age"].min())  # 173 d

# --- base Gompertz ---------------------------------------------------------
def gompertz(t, A, k, ti):
    return A * np.exp(-np.exp(-k * (t - ti)))

base = Model(gompertz)
p0 = base.make_params(A=dict(value=3400, min=y.max(), max=6000),
                      k=dict(value=0.03, min=1e-4, max=0.5),
                      ti=dict(value=140, min=0, max=250))
res_base = base.fit(y, p0, t=t)
resid_base = y - res_base.eval(t=t)

pre, post = resid_base[t < ONSET], resid_base[t >= ONSET]
print("=== 1-2. Residual structure of the base Gompertz ===")
print(f"Mean residual pre-lay  (t < {ONSET}, n={len(pre)}): {pre.mean():+6.1f} g")
print(f"Mean residual post-lay (t >= {ONSET}, n={len(post)}): {post.mean():+6.1f} g")
print("Longest same-sign residual run:",
      max(len(list(g)) for _, g in
          __import__('itertools').groupby(np.sign(resid_base))))

# --- generic IC helpers (consistent with lmfit's definitions) --------------
def aic_bic(rss, n, p):
    aic = n * np.log(rss / n) + 2 * p
    bic = n * np.log(rss / n) + np.log(n) * p
    return aic, bic

def metrics(yhat, p):
    rss = float(np.sum((y - yhat) ** 2))
    rmse = np.sqrt(rss / n)
    return (*aic_bic(rss, n, p), rmse, rss)

# --- Approach B: indicator level shift -------------------------------------
def gompertz_dummy(t, A, k, ti, delta):
    """Gompertz minus a level shift delta (g) once lay has begun."""
    lay = (t >= ONSET).astype(float)
    return A * np.exp(-np.exp(-k * (t - ti))) - delta * lay

mB = Model(gompertz_dummy)
pB = mB.make_params(A=dict(value=3400, min=y.max(), max=6000),
                    k=dict(value=0.03, min=1e-4, max=0.5),
                    ti=dict(value=140, min=0, max=250),
                    delta=dict(value=80, min=-200, max=400))
res_B = mB.fit(y, pB, t=t)

# --- Approach A: segmented (drop + linear regrowth after breakpoint) -------
def segmented_pred(t, A, k, ti, delta, beta, tb):
    """Gompertz for t < tb; for t >= tb: value at tb minus an immediate
    drop delta (g), then linear regrowth at beta (g/day)."""
    g = A * np.exp(-np.exp(-k * (t - ti)))
    g_tb = A * np.exp(-np.exp(-k * (tb - ti)))
    out = np.where(t < tb, g, g_tb - delta + beta * (t - tb))
    return out

def fit_segmented(tb):
    def residual(pars):
        v = pars.valuesdict()
        return y - segmented_pred(t, v["A"], v["k"], v["ti"],
                                  v["delta"], v["beta"], tb)
    pars = Parameters()
    pars.add("A", value=3400, min=y.max(), max=6000)
    pars.add("k", value=0.03, min=1e-4, max=0.5)
    pars.add("ti", value=140, min=0, max=250)
    pars.add("delta", value=60, min=-200, max=400)
    pars.add("beta", value=10, min=0, max=40)
    out = minimize(residual, pars, method="leastsq")
    rss = float(np.sum(out.residual ** 2))
    return out, rss

# fixed breakpoint at onset of lay
out_fix, rss_fix = fit_segmented(float(ONSET))

# estimated breakpoint: profile likelihood over a grid (avoids the
# non-differentiability of tb inside a gradient-based optimizer)
grid = np.arange(168.0, 201.0, 1.0)
profile = [(tb, fit_segmented(tb)[1]) for tb in grid]
tb_hat, rss_hat = min(profile, key=lambda x: x[1])
out_est, _ = fit_segmented(tb_hat)

# --- comparison ------------------------------------------------------------
models = {
    "Gompertz (base)": (res_base.eval(t=t), 3, res_base.params, None),
    "B: dummy shift": (res_B.eval(t=t), 4, res_B.params, None),
    "A: segmented, tb fixed 173": (
        y - out_fix.residual, 5, out_fix.params, float(ONSET)),
    "A: segmented, tb estimated": (
        y - out_est.residual, 6, out_est.params, tb_hat),  # +1 param for tb
}

print(f"\n=== 3-4. Model comparison (n = {n}) ===")
print(f"Estimated breakpoint (profile likelihood): {tb_hat:.0f} d "
      f"(grid 168-200 d)")
rows = ["| Model | params | AIC | BIC | RMSE (g) | mean resid post-lay (g) |",
        "|---|---|---|---|---|---|"]
store = {}
for name, (yhat, p, pars, tb) in models.items():
    aic, bic, rmse, rss = metrics(yhat, p)
    post_mean = float((y - yhat)[t >= ONSET].mean())
    rows.append(f"| {name} | {p} | {aic:.2f} | {bic:.2f} | {rmse:.1f} "
                f"| {post_mean:+.1f} |")
    store[name] = dict(aic=aic, bic=bic, rmse=rmse, p=p,
                       params={k: float(v.value) for k, v in pars.items()},
                       stderr={k: (float(v.stderr) if v.stderr else None)
                               for k, v in pars.items()},
                       tb=tb)
table = "\n".join(rows)
print("\n" + table)

print("\nSegmented (tb estimated) parameters:")
for k_, v in out_est.params.items():
    se = v.stderr if v.stderr is not None else float("nan")
    print(f"  {k_:>6} = {v.value:9.4g}  SE = {se:.3g}")

# --- figure ----------------------------------------------------------------
tg = np.linspace(t.min(), t.max(), 600)
fig, axes = plt.subplots(2, 1, figsize=(9, 10),
                         gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(t, y, "ko", ms=5, label="Observed", zorder=5)
ax.plot(tg, res_base.eval(t=tg), lw=1.5, label="Gompertz (base)")
ax.plot(tg, res_B.eval(t=tg), lw=1.5, label="B: dummy shift")
v = out_est.params.valuesdict()
ax.plot(tg, segmented_pred(tg, v["A"], v["k"], v["ti"], v["delta"],
                           v["beta"], tb_hat),
        lw=1.5, label=f"A: segmented (tb = {tb_hat:.0f} d)")
ax.axvline(ONSET, color="grey", ls="--", lw=1)
ax.annotate(f"onset of lay ({ONSET} d)", xy=(ONSET, y.min()),
            xytext=(ONSET + 1.5, y.min()), fontsize=9)
ax.set_xlabel("Age (days)"); ax.set_ylabel("Body weight (g)")
ax.set_title("Correcting the growth curve for the onset-of-lay drop")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.axhline(0, color="grey", lw=0.8)
ax.plot(t, resid_base, "o-", ms=4, lw=1, label="Gompertz (base)")
ax.plot(t, y - res_B.eval(t=t), "o-", ms=4, lw=1, label="B: dummy shift")
ax.plot(t, out_est.residual, "o-", ms=4, lw=1,
        label=f"A: segmented (tb = {tb_hat:.0f} d)")
ax.axvline(ONSET, color="grey", ls="--", lw=1)
ax.set_xlabel("Age (days)"); ax.set_ylabel("Residual (g)")
ax.set_title("Residuals against age")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{ROOT}/figures/phase3_breakpoint.png", dpi=300)
print("\nSaved figures/phase3_breakpoint.png")

store["_meta"] = {"onset_of_lay": ONSET, "tb_estimated": float(tb_hat),
                  "n": n, "seed": SEED}
with open(f"{ROOT}/models/phase3_params.json", "w") as f:
    json.dump(store, f, indent=2)
with open(f"{ROOT}/models/phase3_comparison.md", "w") as f:
    f.write(table + "\n")
print("Saved models/phase3_params.json and models/phase3_comparison.md")
