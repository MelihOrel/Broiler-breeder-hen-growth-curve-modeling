"""PHASE 5 (prep) — Package fitted parameters and a data snapshot for the app.

Combines the PHASE 2 classical-curve fits and the PHASE 3 segmented model
into a single JSON, plus a CSV snapshot of the body-weight series, so that
the Streamlit app loads instantly and performs no fitting at runtime.
"""

import json

import agridatasets as agd


from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # project root (…/broiler)
(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "app" / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "app" / "data").mkdir(parents=True, exist_ok=True)
df = agd.load_dataset("broiler_growth").sort_values("age")
bw = df[df.bw.notna()][["age", "bw"]]
onset = int(df.loc[df.eggwt.notna(), "age"].min())

with open(f"{ROOT}/models/phase2_params.json") as f:
    p2 = json.load(f)
with open(f"{ROOT}/models/phase3_params.json") as f:
    p3 = json.load(f)

seg = p3["A: segmented, tb estimated"]
app_params = {
    "models": {
        "Gompertz": p2["Gompertz"],
        "Logistic": p2["Logistic"],
        "von Bertalanffy": p2["von Bertalanffy"],
        "Gompertz-Laird": p2["Gompertz-Laird"],
        "Segmented (lay-corrected)": {
            "params": seg["params"], "stderr": seg["stderr"],
            "aic": seg["aic"], "bic": seg["bic"], "rmse": seg["rmse"],
            "tb": seg["tb"],
            "note": ("Asymptote A and inflection ti are NOT identified in "
                     "this model (A pinned at its bound); it describes the "
                     "trajectory, it does not estimate mature weight."),
        },
    },
    "meta": {
        "onset_of_lay": onset,
        "age_min": 143, "age_max": 224, "t0": 143.0,
        "n_rows": int(len(df)), "n_bw": int(len(bw)),
        "drop_verified": True,
        "drop_summary": ("Body weight fell ~71 g between days 178 and 181 "
                         "and then plateaued through day 188, roughly one "
                         "week after the first egg (day 173). The segmented "
                         "model places the breakpoint at day 181 with a "
                         "drop of 175 +/- 27 g and post-break regrowth of "
                         "7.9 +/- 0.5 g/day."),
    },
}

with open(f"{ROOT}/app/models/app_params.json", "w") as f:
    json.dump(app_params, f, indent=2)
bw.to_csv(f"{ROOT}/app/data/bw_series.csv", index=False)
print("Wrote app/models/app_params.json and app/data/bw_series.csv")
