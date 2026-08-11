"""PHASE 1 — Exploration and verification (decision gate).

Loads the zuidhof.broiler series (via agridatasets), establishes structure,
and checks whether a body-weight drop coincides with the onset of lay.

Data: single broiler breeder hen, ages 143-224 d. Body weight is recorded
every 3-4 days; egg weight is recorded near-daily once lay begins, so the
two measurement streams are interleaved across the 59 rows.
"""

import agridatasets as agd
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # project root (…/broiler)
(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42  # no randomness in this phase; kept for convention
np.random.seed(SEED)

df = agd.load_dataset("broiler_growth").sort_values("age").reset_index(drop=True)

# --- basic structure -------------------------------------------------------
print(f"Rows: {len(df)}")
print(f"Age range: {df.age.min()}-{df.age.max()} days")
print("Missing values per column:")
print(df.isna().sum().to_string())

onset = int(df.loc[df.eggwt.notna(), "age"].min())
print(f"\nOnset of lay (first non-missing eggwt): day {onset}")

# body-weight sub-series (weighing days only)
bw = df[df.bw.notna()].copy()
print(f"Body-weight observations: {len(bw)}")

# first differences, expressed per day because weighing intervals are 3-4 d
bw["dbw_per_day"] = bw.bw.diff() / bw.age.diff()

# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(9, 16), sharex=True)

ax = axes[0]
ax.plot(bw.age, bw.bw, "o-", color="tab:blue", ms=5, label="Observed body weight")
ax.set_ylabel("Body weight (g)")
ax.set_title("A. Raw body weight — single broiler breeder hen")

ax = axes[1]
ax.axhline(0, color="grey", lw=0.8)
ax.plot(bw.age, bw.dbw_per_day, "o-", color="tab:purple", ms=5)
ax.set_ylabel("Δ body weight (g/day)")
ax.set_title("B. First difference of body weight (per day, unequal 3-4 d intervals)")

ax = axes[2]
ax.plot(bw.age, bw.adfi, "o-", color="tab:orange", ms=5)
ax.set_ylabel("Feed intake (g/day)")
ax.set_title("C. Average daily feed intake (adfi)")

ax = axes[3]
ax.axhline(0, color="grey", lw=0.8)
ax.plot(bw.age, bw.adg, "o-", color="tab:green", ms=5)
ax.set_ylabel("Daily gain (g/day)")
ax.set_title("D. Average daily gain (adg)")

ax = axes[4]
ax.axhline(0, color="grey", lw=0.8)
ax.plot(bw.age, bw.bw - bw.targetbw, "o-", color="tab:red", ms=5)
ax.set_ylabel("bw − targetbw (g)")
ax.set_title("E. Deviation from rearing-program target weight")
ax.set_xlabel("Age (days)")

for ax in axes:
    ax.axvline(onset, color="black", ls="--", lw=1)
    ax.grid(alpha=0.3)
axes[0].annotate(f"onset of lay (day {onset})", xy=(onset, bw.bw.min()),
                 xytext=(onset + 2, bw.bw.min()), fontsize=9)

fig.tight_layout()
fig.savefig(f"{ROOT}/figures/phase1_exploration.png", dpi=300)
print("\nSaved figures/phase1_exploration.png")

# --- quantitative check around onset --------------------------------------
print("\nBody weight around onset of lay:")
print(bw.loc[(bw.age >= 167) & (bw.age <= 195),
             ["age", "bw", "adg"]].to_string(index=False))
