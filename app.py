"""
Titanic Survival Predictor - interactive Streamlit app.

Run:  streamlit run app.py

Tabs
  1. Prediction      live gauge, tree-vote breakdown, downloadable report
  2. Why this result per-feature effect compared with a "typical" passenger
  3. What-if lab     sweep one feature / 2-D heatmap, split by group
  4. Data explorer   filter the real passenger data and chart it
  5. Model report    accuracy, confusion matrix, ROC, threshold tuning
  6. Compare         save scenarios and compare them side by side
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

import train_model as tm

BASE = Path(__file__).parent
PORTS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
NUMERIC_SWEEP = {
    "age": ("Age", np.linspace(0, 80, 81)),
    "fare": ("Fare ($)", np.linspace(0, 300, 61)),
    "pclass": ("Passenger class", np.array([1, 2, 3])),
    "sibsp": ("Siblings/Spouses", np.arange(0, 9)),
    "parch": ("Parents/Children", np.arange(0, 7)),
}
GREEN, RED = "#2e9e6b", "#d9534f"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")


# --------------------------------------------------------------------------
# Loading (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    """Load saved model files; retrain automatically if missing/incompatible."""
    try:
        model = joblib.load(BASE / "model.pkl")
        encoders = joblib.load(BASE / "encoders.pkl")
        features = joblib.load(BASE / "features.pkl")
    except Exception:
        model, encoders = tm.train()
        features = tm.FEATURES
    return model, encoders, features


@st.cache_data(show_spinner=False)
def load_raw():
    return pd.read_csv(BASE / "titanic.csv")


model, encoders, FEATURES = load_artifacts()
raw = load_raw()
ENC_MAP = {c: dict(zip(e.classes_, e.transform(e.classes_))) for c, e in encoders.items()}


def encode_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in ENC_MAP.items():
        out[col] = out[col].map(mapping)
    return out[FEATURES]


def predict_prob(df: pd.DataFrame) -> np.ndarray:
    """Survival probability for rows given in human-readable form."""
    return model.predict_proba(encode_frame(df))[:, 1]


@st.cache_data(show_spinner=False)
def test_set_scores():
    """Hold-out test set (same split as the notebook) -> y_true, y_prob."""
    data, _ = tm.encode(tm.load_clean_data())
    _, X_test, _, y_test = tm.split(data)
    return y_test.to_numpy(), model.predict_proba(X_test[FEATURES])[:, 1]


@st.cache_data(show_spinner=False)
def clean_reference():
    d = tm.load_clean_data()
    return {
        "pclass": int(d["pclass"].mode()[0]),
        "sex": d["sex"].mode()[0],
        "age": float(d["age"].median()),
        "sibsp": int(d["sibsp"].median()),
        "parch": int(d["parch"].median()),
        "fare": float(d["fare"].median()),
        "embarked": d["embarked"].mode()[0],
    }


BASELINE = clean_reference()
BASE_RATE = raw["survived"].mean()

# --------------------------------------------------------------------------
# Session state + presets
# --------------------------------------------------------------------------
DEFAULTS = dict(pclass=3, sex="male", age=30, sibsp=0, parch=0, fare=32.0, embarked="S")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("scenarios", [])
st.session_state.setdefault("threshold", 0.5)

PRESETS = {
    "👨 3rd-class young man": dict(pclass=3, sex="male", age=22, sibsp=0, parch=0, fare=8.0, embarked="S"),
    "👩 1st-class woman": dict(pclass=1, sex="female", age=35, sibsp=1, parch=0, fare=120.0, embarked="C"),
    "👧 Child with family": dict(pclass=2, sex="female", age=7, sibsp=1, parch=2, fare=30.0, embarked="S"),
    "👴 Elderly 1st-class man": dict(pclass=1, sex="male", age=68, sibsp=0, parch=0, fare=80.0, embarked="S"),
    "👨‍👩‍👧‍👦 Large 3rd-class family": dict(pclass=3, sex="male", age=40, sibsp=3, parch=3, fare=25.0, embarked="S"),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def reset_inputs():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def randomize_inputs():
    rng = np.random.default_rng()
    st.session_state.update(
        pclass=int(rng.choice([1, 2, 3])),
        sex=str(rng.choice(["male", "female"])),
        age=int(rng.integers(1, 80)),
        sibsp=int(rng.integers(0, 4)),
        parch=int(rng.integers(0, 3)),
        fare=float(round(rng.uniform(5, 150), 1)),
        embarked=str(rng.choice(["S", "C", "Q"])),
    )


# --------------------------------------------------------------------------
# Sidebar: passenger builder
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🧳 Build a passenger")
    st.caption("Everything updates live - no button needed.")

    st.write("**Quick presets**")
    for name in PRESETS:
        st.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), width="stretch")
    c1, c2 = st.columns(2)
    c1.button("🎲 Random", on_click=randomize_inputs, width="stretch")
    c2.button("↺ Reset", on_click=reset_inputs, width="stretch")

    st.divider()
    st.selectbox("Passenger class", [1, 2, 3], key="pclass", format_func=lambda x: f"Class {x}")
    st.radio("Sex", ["male", "female"], key="sex", horizontal=True)
    st.slider("Age", 0, 90, key="age")
    st.slider("Siblings / spouses aboard", 0, 8, key="sibsp")
    st.slider("Parents / children aboard", 0, 6, key="parch")
    st.slider("Fare paid ($)", 0.0, 520.0, step=1.0, key="fare")
    st.selectbox("Port of embarkation", list(PORTS), key="embarked", format_func=lambda x: PORTS[x])

    st.divider()
    st.slider(
        "Decision threshold", 0.10, 0.90, step=0.05, key="threshold",
        help="Predict 'Survived' when probability is at least this value. "
             "Lower = catches more survivors but more false alarms.",
    )

    st.divider()
    st.caption(
        f"**Model:** Random Forest · {model.n_estimators} trees · depth {model.max_depth}\n\n"
        f"**Features:** {', '.join(FEATURES)}"
    )

# Current passenger
current = {k: st.session_state[k] for k in DEFAULTS}
current_df = pd.DataFrame([current])
prob = float(predict_prob(current_df)[0])
threshold = st.session_state["threshold"]
survived = prob >= threshold

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.write(
    "A Random Forest trained on real Titanic passenger records. Change the passenger in the sidebar "
    "and explore the prediction, the reasoning behind it, what-if scenarios, the data and the model."
)

tab_pred, tab_why, tab_lab, tab_data, tab_model, tab_cmp = st.tabs(
    ["🎯 Prediction", "🔍 Why this result", "🧪 What-if lab", "📊 Data explorer", "📈 Model report", "⚖️ Compare"]
)

# ============================== 1. Prediction ==============================
with tab_pred:
    left, right = st.columns([1, 1])

    with left:
        if survived:
            st.success(f"### ✅ Predicted: Survived\nProbability **{prob:.1%}**")
        else:
            st.error(f"### ❌ Predicted: Did not survive\nSurvival probability **{prob:.1%}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Survival probability", f"{prob:.1%}")
        m2.metric("vs. average passenger", f"{(prob - BASE_RATE) * 100:+.1f} pts",
                  help=f"Overall survival rate in the data is {BASE_RATE:.1%}.")
        m3.metric("Threshold", f"{threshold:.0%}")

        who = "woman" if current["sex"] == "female" else "man"
        if current["age"] < 13:
            who = "girl" if current["sex"] == "female" else "boy"
        st.caption(
            f"A {current['age']}-year-old {who} travelling in class {current['pclass']}, "
            f"embarked at {PORTS[current['embarked']]}, with {current['sibsp']} sibling(s)/spouse and "
            f"{current['parch']} parent(s)/child(ren) aboard, fare ${current['fare']:.0f}."
        )

        b1, b2 = st.columns(2)
        if b1.button("💾 Save to Compare tab", width="stretch"):
            n = len(st.session_state["scenarios"]) + 1
            st.session_state["scenarios"].append({"Name": f"Scenario {n}", **current, "Probability": prob})
            st.toast(f"Saved as Scenario {n}", icon="💾")
        report = current_df.assign(
            survival_probability=round(prob, 4), threshold=threshold,
            prediction="Survived" if survived else "Did not survive",
        )
        b2.download_button("⬇️ Download report (CSV)", report.to_csv(index=False),
                           "titanic_prediction.csv", "text/csv", width="stretch")

    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Survival probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": GREEN if survived else RED},
                "steps": [{"range": [0, 33], "color": "rgba(217,83,79,0.15)"},
                          {"range": [33, 66], "color": "rgba(240,173,78,0.15)"},
                          {"range": [66, 100], "color": "rgba(46,158,107,0.15)"}],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.8,
                              "value": threshold * 100},
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Black line = your decision threshold.")

    st.subheader("🌲 How did the 300 trees vote?")
    X_row = encode_frame(current_df).to_numpy(dtype=np.float32)
    tree_p = np.array([t.predict_proba(X_row)[0, 1] for t in model.estimators_])
    votes_yes = int((tree_p >= 0.5).sum())
    vc1, vc2 = st.columns([2, 1])
    with vc1:
        hist = px.histogram(
            x=tree_p, nbins=20, range_x=[0, 1],
            labels={"x": "Survival probability given by a single tree", "y": "Number of trees"},
            color_discrete_sequence=["#4c78a8"],
        )
        hist.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Number of trees")
        st.plotly_chart(hist, width="stretch")
    with vc2:
        st.metric("Trees voting 'Survived'", f"{votes_yes} / {len(tree_p)}")
        st.metric("Trees voting 'Died'", f"{len(tree_p) - votes_yes} / {len(tree_p)}")
        st.caption(f"Spread between trees (std): {tree_p.std():.2f}. A wide spread means the model is less certain.")

# ============================== 2. Why this result ==============================
with tab_why:
    st.subheader("Which inputs push this prediction up or down?")
    st.write(
        "For each feature we swap this passenger's value with a **typical value** "
        "(median / most common) and see how the probability changes. "
        "Green = this feature *helps* survival, red = it *hurts*."
    )
    rows = []
    for f in FEATURES:
        alt = current.copy()
        alt[f] = BASELINE[f]
        p_alt = float(predict_prob(pd.DataFrame([alt]))[0])
        rows.append({"Feature": f, "Your value": str(current[f]), "Typical value": str(BASELINE[f]),
                     "Effect (pts)": (prob - p_alt) * 100})
    eff = pd.DataFrame(rows).sort_values("Effect (pts)")
    fig = go.Figure(go.Bar(
        x=eff["Effect (pts)"], y=eff["Feature"], orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in eff["Effect (pts)"]],
        text=[f"{v:+.1f}" for v in eff["Effect (pts)"]], textposition="outside",
    ))
    fig.update_layout(height=380, xaxis_title="Change in survival probability (percentage points)",
                      margin=dict(l=10, r=30, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    top = eff.iloc[eff["Effect (pts)"].abs().argmax()]
    direction = "raises" if top["Effect (pts)"] > 0 else "lowers"
    st.info(f"💡 The biggest factor is **{top['Feature']}** (= {top['Your value']}), "
            f"which {direction} the probability by about **{abs(top['Effect (pts)']):.1f} points** "
            f"compared with a typical passenger.")
    st.dataframe(eff.sort_values("Effect (pts)", ascending=False).round(2), hide_index=True, width="stretch")

    st.subheader("Global feature importance")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig = px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""},
                 color_discrete_sequence=["#4c78a8"])
    fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# ============================== 3. What-if lab ==============================
with tab_lab:
    mode = st.radio("Experiment", ["📈 Sweep one feature", "🟧 2-D heatmap"], horizontal=True)

    if mode.startswith("📈"):
        c1, c2 = st.columns(2)
        feat = c1.selectbox("Feature to vary", list(NUMERIC_SWEEP), format_func=lambda k: NUMERIC_SWEEP[k][0])
        split_by = c2.selectbox("Draw separate lines for", ["(none)", "sex", "pclass", "embarked"])
        label, grid = NUMERIC_SWEEP[feat]

        def sweep(base_row):
            df = pd.DataFrame([base_row] * len(grid))
            df[feat] = grid
            return predict_prob(df)

        fig = go.Figure()
        if split_by == "(none)":
            fig.add_trace(go.Scatter(x=grid, y=sweep(current) * 100, mode="lines+markers", name="Your passenger",
                                     line=dict(width=3)))
        else:
            opts = {"sex": ["male", "female"], "pclass": [1, 2, 3], "embarked": ["S", "C", "Q"]}[split_by]
            for o in opts:
                if split_by == feat:
                    continue
                fig.add_trace(go.Scatter(x=grid, y=sweep({**current, split_by: o}) * 100, mode="lines+markers",
                                         name=f"{split_by} = {o}"))
        fig.add_trace(go.Scatter(x=[current[feat]], y=[prob * 100], mode="markers", name="Current setting",
                                 marker=dict(size=14, color="black", symbol="star")))
        fig.add_hline(y=threshold * 100, line_dash="dash", annotation_text="threshold")
        fig.update_layout(height=450, xaxis_title=label, yaxis_title="Survival probability (%)",
                          yaxis_range=[0, 100], hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        if split_by == feat:
            st.warning("Pick a different 'separate lines' option than the feature you are varying.")
    else:
        keys = list(NUMERIC_SWEEP)
        c1, c2 = st.columns(2)
        fx = c1.selectbox("X axis", keys, index=0, format_func=lambda k: NUMERIC_SWEEP[k][0])
        fy = c2.selectbox("Y axis", [k for k in keys if k != fx], index=0,
                          format_func=lambda k: NUMERIC_SWEEP[k][0])
        gx, gy = NUMERIC_SWEEP[fx][1], NUMERIC_SWEEP[fy][1]
        xx, yy = np.meshgrid(gx, gy)
        grid_df = pd.DataFrame([current] * xx.size)
        grid_df[fx], grid_df[fy] = xx.ravel(), yy.ravel()
        z = predict_prob(grid_df).reshape(xx.shape) * 100
        fig = go.Figure(go.Heatmap(x=gx, y=gy, z=z, colorscale="RdYlGn", zmin=0, zmax=100,
                                   colorbar=dict(title="Survival %"),
                                   hovertemplate=f"{fx}=%{{x}}<br>{fy}=%{{y}}<br>%{{z:.1f}}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[current[fx]], y=[current[fy]], mode="markers",
                                 marker=dict(size=14, color="black", symbol="star"), name="Current"))
        fig.update_layout(height=480, xaxis_title=NUMERIC_SWEEP[fx][0], yaxis_title=NUMERIC_SWEEP[fy][0],
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("All other inputs are taken from the sidebar. The star marks your current passenger.")

# ============================== 4. Data explorer ==============================
with tab_data:
    st.subheader("Explore the real passenger data")
    f1, f2, f3 = st.columns(3)
    sex_f = f1.multiselect("Sex", ["male", "female"], default=["male", "female"])
    cls_f = f2.multiselect("Class", [1, 2, 3], default=[1, 2, 3])
    age_f = f3.slider("Age range", 0, 80, (0, 80))

    d = raw[raw["sex"].isin(sex_f) & raw["pclass"].isin(cls_f)]
    d = d[d["age"].between(*age_f) | d["age"].isna()].copy()
    d["Outcome"] = d["survived"].map({1: "Survived", 0: "Died"})
    d["age_group"] = pd.cut(d["age"], [0, 12, 18, 35, 60, 100],
                            labels=["Child (0-12)", "Teen (13-18)", "Adult (19-35)", "Middle (36-60)", "Senior (60+)"])

    k1, k2, k3 = st.columns(3)
    k1.metric("Passengers", len(d))
    k2.metric("Survival rate", f"{d['survived'].mean():.1%}" if len(d) else "-",
              delta=f"{(d['survived'].mean() - BASE_RATE) * 100:+.1f} pts vs all" if len(d) else None)
    k3.metric("Median fare", f"${d['fare'].median():.1f}" if len(d) else "-")

    if len(d) == 0:
        st.warning("No passengers match these filters.")
    else:
        group = st.selectbox("Survival rate by", ["sex", "pclass", "embarked", "who", "alone", "age_group"])
        rate = d.groupby(group, observed=True)["survived"].agg(["mean", "count"]).reset_index()
        fig = px.bar(rate, x=group, y="mean", text=rate["mean"].map("{:.0%}".format),
                     hover_data={"count": True}, labels={"mean": "Survival rate"},
                     color="mean", color_continuous_scale="RdYlGn", range_color=[0, 1])
        fig.update_layout(height=340, yaxis_tickformat=".0%", coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        g1, g2 = st.columns(2)
        with g1:
            fig = px.histogram(d, x="age", color="Outcome", nbins=30, barmode="overlay", opacity=0.65,
                               color_discrete_map={"Survived": GREEN, "Died": RED})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Age distribution")
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig = px.scatter(d, x="age", y="fare", color="Outcome", symbol="sex", hover_data=["pclass", "embarked"],
                             color_discrete_map={"Survived": GREEN, "Died": RED}, opacity=0.7)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Age vs fare")
            st.plotly_chart(fig, width="stretch")

        with st.expander("📄 View / download filtered data"):
            show = d[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
            st.dataframe(show, hide_index=True, width="stretch")
            st.download_button("⬇️ Download CSV", show.to_csv(index=False), "titanic_filtered.csv", "text/csv")

# ============================== 5. Model report ==============================
with tab_model:
    y_true, y_prob = test_set_scores()
    st.subheader("How good is the model? (20% hold-out test set)")
    t = st.slider("Threshold used for the metrics below", 0.10, 0.90, float(threshold), 0.05, key="metric_t")
    y_hat = (y_prob >= t).astype(int)

    a, b, c, e = st.columns(4)
    a.metric("Accuracy", f"{accuracy_score(y_true, y_hat):.1%}")
    b.metric("Precision", f"{precision_score(y_true, y_hat, zero_division=0):.1%}",
             help="Of those predicted to survive, how many did.")
    c.metric("Recall", f"{recall_score(y_true, y_hat):.1%}",
             help="Of those who survived, how many the model found.")
    e.metric("F1 score", f"{f1_score(y_true, y_hat):.2f}")

    p1, p2 = st.columns(2)
    with p1:
        cm = confusion_matrix(y_true, y_hat)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Died", "Survived"], y=["Died", "Survived"],
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(height=360, title="Confusion matrix", coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with p2:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc(fpr, tpr):.3f})",
                                 line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                 line=dict(dash="dash", color="grey")))
        tn, fp, fn, tp = cm.ravel()
        fig.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers", name="Your threshold",
                                 marker=dict(size=13, color="black", symbol="star")))
        fig.update_layout(height=360, title="ROC curve", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("Move the threshold slider to see the trade-off between precision and recall.")

# ============================== 6. Compare ==============================
with tab_cmp:
    st.subheader("Compare saved scenarios")
    sc = st.session_state["scenarios"]
    if not sc:
        st.info("No scenarios yet. Build a passenger, then click **💾 Save to Compare tab** on the Prediction tab.")
    else:
        sdf = pd.DataFrame(sc)
        fig = px.bar(sdf, x="Name", y="Probability", text=sdf["Probability"].map("{:.0%}".format),
                     color="Probability", color_continuous_scale="RdYlGn", range_color=[0, 1])
        fig.update_layout(height=360, yaxis_tickformat=".0%", yaxis_range=[0, 1], coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=10, b=10))
        fig.add_hline(y=threshold, line_dash="dash", annotation_text="threshold")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(sdf.style.format({"Probability": "{:.1%}", "fare": "${:.0f}"}), hide_index=True,
                     width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download scenarios (CSV)", sdf.to_csv(index=False), "scenarios.csv", "text/csv",
                           width="stretch")
        if d2.button("🗑️ Clear all", width="stretch"):
            st.session_state["scenarios"] = []
            st.rerun()

st.divider()
st.caption("Made with lots of ❤️ · Random Forest · Streamlit · Plotly")
