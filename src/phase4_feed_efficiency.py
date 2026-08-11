"""PHASE 4 — Feed efficiency side analysis (deliberately short).

1. Models the adfi-adg relationship two ways: a naive correlation and a
   simple energy-partition regression adfi ~ metabolic weight + gain,
   whose residuals after onset of lay indicate feed diverted to eggs.
2. Feed conversion: FCR = adfi/adg blows up when adg <= 0, so the primary
   plot is the bounded inverse (gross efficiency = adg/adfi); FCR is shown
   only for days with adg > 0 and the exclusions are reported.
"""

import agridatasets as agd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # project root (…/broiler)
(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

df = agd.load_dataset("broiler_growth").sort_values("age")
d = df[df.bw.notna()].copy()
ONSET = int(df.loc[df.eggwt.notna(), "age"].min())
d["lay"] = (d.age >= ONSET).astype(int)

# --- 1. adfi-adg relationship ---------------------------------------------
r_pre = np.corrcoef(d.loc[d.lay == 0, "adfi"], d.loc[d.lay == 0, "adg"])[0, 1]
r_post = np.corrcoef(d.loc[d.lay == 1, "adfi"], d.loc[d.lay == 1, "adg"])[0, 1]
print(f"Pearson r(adfi, adg): pre-lay {r_pre:+.2f} (n={ (d.lay==0).sum() }), "
      f"post-lay {r_post:+.2f} (n={ (d.lay==1).sum() })")

# energy-partition regression: intake = maintenance (metabolic weight,
# kg^0.75) + cost of gain. Fitted on PRE-LAY days only, then applied to all
# days: post-lay residuals estimate feed not explained by maintenance+growth.
d["mbw"] = (d.bw / 1000.0) ** 0.75
X = sm.add_constant(d.loc[d.lay == 0, ["mbw", "adg"]])
fit = sm.OLS(d.loc[d.lay == 0, "adfi"], X).fit()
print("\nPartition model (pre-lay only): adfi = b0 + b1*BW_kg^0.75 + b2*adg")
print(fit.summary2().tables[1].round(3).to_string())
d["adfi_hat"] = fit.predict(sm.add_constant(d[["mbw", "adg"]]))
d["surplus"] = d.adfi - d.adfi_hat
print(f"\nMean intake surplus over maintenance+growth: "
      f"pre-lay {d.loc[d.lay==0,'surplus'].mean():+.1f} g/day, "
      f"post-lay {d.loc[d.lay==1,'surplus'].mean():+.1f} g/day")

# --- 2. feed conversion ----------------------------------------------------
d["eff"] = d.adg / d.adfi                      # gross efficiency, bounded
mask = d.adg > 0
d.loc[mask, "fcr"] = d.loc[mask, "adfi"] / d.loc[mask, "adg"]
excl = d.loc[~mask, "age"].tolist()
print(f"\nFCR undefined/meaningless where adg <= 0; excluded ages: {excl}")

# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=False)

ax = axes[0]
ax.scatter(d.loc[d.lay == 0, "adg"], d.loc[d.lay == 0, "adfi"],
           c="tab:blue", label="Pre-lay")
ax.scatter(d.loc[d.lay == 1, "adg"], d.loc[d.lay == 1, "adfi"],
           c="tab:red", label="Post-lay")
ax.set_xlabel("Daily gain (g/day)"); ax.set_ylabel("Feed intake (g/day)")
ax.set_title("A. Feed intake vs daily gain, split at onset of lay")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.axhline(0, color="grey", lw=0.8)
ax.plot(d.age, d.surplus, "o-", ms=4,
        color="tab:brown")
ax.axvline(ONSET, color="grey", ls="--", lw=1)
ax.set_xlabel("Age (days)")
ax.set_ylabel("Intake surplus (g/day)")
ax.set_title("B. Intake not explained by maintenance + growth\n"
             "(partition model fitted pre-lay; post-lay surplus ≈ feed "
             "diverted to egg production)")
ax.grid(alpha=0.3)

ax = axes[2]
ax.axhline(0, color="grey", lw=0.8)
ax.plot(d.age, d.eff, "o-", ms=4, color="tab:green",
        label="Gross efficiency adg/adfi (g gain per g feed)")
ax.plot(d.loc[mask, "age"], d.loc[mask, "fcr"] / 100, "s--", ms=4,
        color="tab:purple", label="FCR/100 (g feed per g gain; adg>0 only)")
ax.axvline(ONSET, color="grey", ls="--", lw=1)
ax.set_xlabel("Age (days)"); ax.set_ylabel("Ratio (dimensionless)")
ax.set_title("C. Feed efficiency against age")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{ROOT}/figures/phase4_feed_efficiency.png", dpi=300)
print("\nSaved figures/phase4_feed_efficiency.png")
