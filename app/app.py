"""Broiler breeder hen growth curves — interactive viewer.

Streamlit app over pre-fitted parametric growth models (no fitting happens
here: parameters are read from models/app_params.json, produced by the
analysis pipeline). Data: zuidhof.broiler — a single broiler breeder hen's
time series, ages 143-224 d, 24 body-weight observations.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
T0 = 143.0  # time anchor for von Bertalanffy and Gompertz-Laird


# ---------------------------------------------------------------- loading
@st.cache_data
def load_assets():
    with open(HERE / "models" / "app_params.json") as f:
        assets = json.load(f)
    data = pd.read_csv(HERE / "data" / "bw_series.csv")
    return assets, data


assets, data = load_assets()
MODELS = assets["models"]
META = assets["meta"]
ONSET = META["onset_of_lay"]
TB = MODELS["Segmented (lay-corrected)"]["tb"]


# ------------------------------------------------- curves and derivatives
def predict(name, t, p):
    t = np.asarray(t, dtype=float)
    if name == "Gompertz":
        return p["A"] * np.exp(-np.exp(-p["k"] * (t - p["ti"])))
    if name == "Logistic":
        return p["A"] / (1 + np.exp(-p["k"] * (t - p["ti"])))
    if name == "von Bertalanffy":
        return p["A"] * (1 - p["b"] * np.exp(-p["k"] * (t - T0))) ** 3
    if name == "Gompertz-Laird":
        return p["W0"] * np.exp((p["L"] / p["k"])
                                * (1 - np.exp(-p["k"] * (t - T0))))
    # Segmented: Gompertz before the breakpoint; immediate drop delta and
    # linear regrowth beta after it.
    g = p["A"] * np.exp(-np.exp(-p["k"] * (t - p["ti"])))
    g_tb = p["A"] * np.exp(-np.exp(-p["k"] * (TB - p["ti"])))
    return np.where(t < TB, g, g_tb - p["delta"] + p["beta"] * (t - TB))


def growth_rate(name, t, p):
    """Instantaneous growth rate dW/dt (g/day), analytic."""
    t = np.asarray(t, dtype=float)
    if name == "Gompertz":
        z = np.exp(-p["k"] * (t - p["ti"]))
        return p["A"] * p["k"] * z * np.exp(-z)
    if name == "Logistic":
        w = predict(name, t, p)
        return p["k"] * w * (1 - w / p["A"])
    if name == "von Bertalanffy":
        e = p["b"] * np.exp(-p["k"] * (t - T0))
        return 3 * p["A"] * p["k"] * e * (1 - e) ** 2
    if name == "Gompertz-Laird":
        w = predict(name, t, p)
        return w * p["L"] * np.exp(-p["k"] * (t - T0))
    z = np.exp(-p["k"] * (t - p["ti"]))
    pre = p["A"] * p["k"] * z * np.exp(-z)
    return np.where(t < TB, pre, p["beta"])


# --------------------------------------------------------------------- UI
st.set_page_config(page_title="Breeder hen growth curves", layout="wide")
st.title("Broiler breeder hen growth curves")
st.caption("One hen, ages 143-224 days. Pre-fitted parametric models — "
           "nothing is refitted in this app.")

tab_model, tab_about = st.tabs(["Models", "About the data"])

with tab_model:
    left, right = st.columns([1, 2])

    with left:
        model_name = st.selectbox("Model", list(MODELS.keys()))
        overlay = st.multiselect("Overlay other models",
                                 [m for m in MODELS if m != model_name])
        age = st.number_input(
            "Age (days)", min_value=float(META["age_min"]),
            max_value=float(META["age_max"]), value=180.0, step=1.0,
            help="Limited to the observed range 143-224 days; the models "
                 "are not validated outside it.")

        p = MODELS[model_name]["params"]
        w_hat = float(predict(model_name, age, p))
        r_hat = float(growth_rate(model_name, age, p))
        st.metric("Predicted body weight", f"{w_hat:,.0f} g")
        st.metric("Instantaneous growth rate", f"{r_hat:+.1f} g/day")

        st.subheader("Parameters")
        rows = []
        for k, v in p.items():
            se = MODELS[model_name]["stderr"].get(k)
            ci = (f"[{v - 1.96 * se:,.4g}, {v + 1.96 * se:,.4g}]"
                  if se else "not estimable")
            rows.append({"parameter": k, "estimate": f"{v:,.4g}",
                         "95% CI": ci})
        st.table(pd.DataFrame(rows))
        st.caption(f"AIC {MODELS[model_name]['aic']:.1f} · "
                   f"BIC {MODELS[model_name]['bic']:.1f} · "
                   f"RMSE {MODELS[model_name]['rmse']:.1f} g")
        if "note" in MODELS[model_name]:
            st.warning(MODELS[model_name]["note"])

    with right:
        tg = np.linspace(META["age_min"], META["age_max"], 600)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data.age, data.bw, "ko", ms=5, label="Observed", zorder=5)
        ax.plot(tg, predict(model_name, tg, p), lw=2, label=model_name)
        for name in overlay:
            ax.plot(tg, predict(name, tg, MODELS[name]["params"]),
                    lw=1.2, alpha=0.8, label=name)
        ax.axvline(ONSET, color="grey", ls="--", lw=1)
        ax.axvline(age, color="tab:red", ls=":", lw=1)
        ax.annotate(f"onset of lay ({ONSET} d)",
                    xy=(ONSET, data.bw.min()), fontsize=9,
                    xytext=(ONSET + 1.5, data.bw.min()))
        ax.set_xlabel("Age (days)")
        ax.set_ylabel("Body weight (g)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.info("**Onset of lay.** " + META["drop_summary"] + " Classical "
                "smooth curves miss this: their residuals form same-signed "
                "runs around onset. The segmented model captures it.")

with tab_about:
    st.markdown(f"""
### What this data is

- A **single broiler breeder hen** (not a meat broiler), observed from
  **{META['age_min']} to {META['age_max']} days** of age.
- {META['n_rows']} rows, of which **{META['n_bw']} are weighing days**
  carrying body weight; the rest are near-daily egg-weight records after
  lay begins. The effective sample size for curve fitting is therefore
  {META['n_bw']}, not {META['n_rows']}.
- One bird, no replicates, no pens: this is a single time series. Mixed
  models do not apply; the analysis is single-series nonlinear regression,
  and **no finding here generalizes beyond this bird**.
- The observed window covers only the tail of the growth trajectory, so
  asymptote and inflection parameters are extrapolations with the
  uncertainty that implies.
- Mechanistic parametric modeling was a deliberate choice: with an
  effective n of {META['n_bw']}, machine learning would be unsupportable.

### Provenance

1. `agridatasets` Python package (PyPI), Renzo Caceres Rossi
2. `agridat` R package (CRAN), Kevin Wright — original name
   `zuidhof.broiler`
3. Zuidhof, M.J., Schneider, B.L., Carney, V.L., Korver, D.R.,
   Robinson, F.E. (2014). Growth, efficiency, and yield of commercial
   broilers from 1957, 1978, and 2005. *Poultry Science*, 93(12),
   2970-2982. DOI:
   [10.3382/ps.2014-04291](https://doi.org/10.3382/ps.2014-04291)
""")
