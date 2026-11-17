import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from pythermalcomfort.models import pmv_ppd_iso
    THERMAL_LIBRARY_READY = True
except ImportError:
    THERMAL_LIBRARY_READY = False

st.set_page_config(page_title="Office IEQ Comfort Assessment", page_icon="🏢", layout="wide")
st.title("Human-Centric Office IEQ Assessment")
st.caption("Occupant-specific comfort assessment for a human–robot office scenario")

DATA_FILE = Path(__file__).parent / "office_ieq_synthetic_training_data.csv"
FEATURES = [
    "air_temp_c", "mean_radiant_temp_c", "air_speed_ms",
    "relative_humidity_pct", "clothing_clo", "activity_met",
    "noise_dba", "illuminance_lux", "co2_ppm", "robot_distance_m",
    "cleaning_robot_active", "service_robot_active"
]

@st.cache_data
def load_training_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Training CSV is missing. Keep office_ieq_synthetic_training_data.csv "
            "in the same folder as app.py."
        )
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data["comfort_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250, min_samples_leaf=3, class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    return model, score, X_test, y_test

def calculate_pmv_ppd(person):
    if not THERMAL_LIBRARY_READY:
        return None, None, "Library is not installed."
    try:
        # ISO 7730 PMV/PPD: six thermal inputs are evaluated for each occupant.
        result = pmv_ppd_iso(
            tdb=float(person["air_temp_c"]),
            tr=float(person["mean_radiant_temp_c"]),
            vr=float(person["air_speed_ms"]),
            rh=float(person["relative_humidity_pct"]),
            met=float(person["activity_met"]),
            clo=float(person["clothing_clo"]),
            model="7730-2025",
            limit_inputs=True,
            round_output=False
        )
        pmv = float(result.pmv)
        ppd = float(result.ppd)
        if not np.isfinite(pmv) or not np.isfinite(ppd):
            return None, None, (
                "Inputs are outside the selected ISO model's applicability limits. "
                "Check temperature, air speed, activity and clothing values."
            )
        return pmv, ppd, None
    except Exception as exc:
        return None, None, f"Calculation error: {exc}"

def assess_reasons(p, pmv):
    reasons, actions = [], []
    # Thermal reasons: PMV is preferred when the calculation is valid.
    if pmv is not None:
        if pmv < -0.5:
            reasons.append(f"Thermal sensation is predicted on the cool side (PMV {pmv:.2f}).")
            actions.append("For this person, consider a small local warming adjustment, warmer clothing, or reducing a direct cold draft. Do not heat the whole office if other occupants are already warm.")
        elif pmv > 0.5:
            reasons.append(f"Thermal sensation is predicted on the warm side (PMV {pmv:.2f}).")
            actions.append("For this person, consider a small local cooling adjustment, a personal fan where appropriate, or reducing nearby heat gain. Avoid cooling the whole office without checking other occupants.")
    else:
        # Transparent screening only, not a standards-based comfort conclusion.
        if p["air_temp_c"] < 20:
            reasons.append("Air temperature is relatively low for this demo scenario.")
            actions.append("Check clothing and local drafts before considering a small heating adjustment.")
        elif p["air_temp_c"] > 26:
            reasons.append("Air temperature is relatively high for this demo scenario.")
            actions.append("Check solar/radiant heat before considering a small cooling adjustment.")

    if abs(p["mean_radiant_temp_c"] - p["air_temp_c"]) >= 2:
        reasons.append("Mean radiant temperature differs noticeably from air temperature; a hot window, cold window or warm surface may affect comfort.")
        actions.append("Inspect window/surface temperature, blinds and workstation location; consider moving the desk or using shading.")
    if p["air_speed_ms"] > 0.25:
        reasons.append("Air speed is relatively high and may create a draft for a seated person.")
        actions.append("Check air direction at the person's desk and redirect the diffuser or reduce local airflow if the occupant reports a draft.")
    if p["relative_humidity_pct"] < 30:
        reasons.append("Relative humidity is low in this scenario.")
        actions.append("Check the humidity sensor and ventilation/humidification strategy; avoid assuming humidity alone explains thermal discomfort.")
    elif p["relative_humidity_pct"] > 65:
        reasons.append("Relative humidity is high in this scenario.")
        actions.append("Check moisture sources and ventilation/dehumidification; verify readings before changing HVAC settings.")
    if p["noise_dba"] >= 55:
        reasons.append("Noise level is elevated in this demo scenario.")
        actions.append("Identify the noise source; if it is the cleaning robot, schedule cleaning at a quieter time or test a safe alternative route.")
    if p["illuminance_lux"] < 300:
        reasons.append("Desk illuminance is low for this assumed computer-work scenario.")
        actions.append("Check task lighting and desk position; add or adjust local task lighting.")
    elif p["illuminance_lux"] > 800:
        reasons.append("Desk illuminance is high and may contribute to glare, depending on screen reflections and daylight.")
        actions.append("Check glare and screen reflections; adjust blinds or task lighting rather than simply lowering all room lighting.")
    if p["co2_ppm"] > 1000:
        reasons.append("CO₂ is elevated relative to the demo's screening threshold.")
        actions.append("Verify the CO₂ sensor and occupancy; review ventilation operation and outdoor-air supply.")
    if p["cleaning_robot_active"] and p["robot_distance_m"] < 2.5:
        reasons.append("The cleaning robot is active close to the occupant; noise or proximity may be a concern.")
        actions.append("Check measured noise; consider a safe route with greater separation or cleaning at a less occupied time.")
    if p["service_robot_active"] and p["robot_distance_m"] < 1.5:
        reasons.append("The service robot is passing close to the occupant.")
        actions.append("Check the robot's safe passing clearance; consider a wider route if the navigation system permits.")
    if not reasons:
        reasons.append("No configured screening trigger was found in the entered values.")
        actions.append("Continue monitoring environmental conditions and check for unmeasured factors.")
    return reasons, list(dict.fromkeys(actions))

def make_person_input(prefix, clothing, clo, activity, met, env):
    p = dict(env)
    p["clothing_clo"] = float(clo)
    p["activity_met"] = float(met)
    p["profile"] = clothing
    p["person"] = prefix
    return p

def evaluate_person(model, p):
    row = pd.DataFrame([{k: p[k] for k in FEATURES}])
    pred = model.predict(row)[0]
    probs = model.predict_proba(row)[0]
    classes = list(model.classes_)
    p_uncomfortable = float(probs[classes.index("uncomfortable")])
    pmv, ppd, pmv_error = calculate_pmv_ppd(p)
    reasons, actions = assess_reasons(p, pmv)
    # Keep ML classification and PMV evidence distinct.
    if pmv is not None and abs(pmv) > 0.5:
        final_label = "Human Comfort Needs Review"
    elif pred == "uncomfortable":
        final_label = "Potential Human Discomfort"
    else:
        final_label = "No Strong Discomfort Signal"
    return pred, p_uncomfortable, pmv, ppd, pmv_error, reasons, actions, final_label

data = load_training_data()
model, test_accuracy, X_test, y_test = train_model(data)

st.sidebar.header("Shared office conditions")
env = {
    "air_temp_c": st.sidebar.slider("Air temperature (°C)", 16.0, 32.0, 24.0, 0.1),
    "mean_radiant_temp_c": st.sidebar.slider("Mean radiant temperature (°C)", 16.0, 32.0, 24.0, 0.1),
    "air_speed_ms": st.sidebar.slider("Air speed (m/s)", 0.05, 1.0, 0.10, 0.01),
    "relative_humidity_pct": float(st.sidebar.slider("Relative humidity (%)", 20, 80, 50)),
    "noise_dba": float(st.sidebar.slider("Noise (dBA)", 30, 85, 45)),
    "illuminance_lux": float(st.sidebar.slider("Desk illuminance (lux)", 100, 1000, 500, 10)),
    "co2_ppm": float(st.sidebar.slider("CO₂ (ppm)", 400, 1800, 650, 25)),
    "robot_distance_m": st.sidebar.slider("Nearest robot distance (m)", 0.3, 10.0, 4.0, 0.1),
    "cleaning_robot_active": int(st.sidebar.selectbox("Cleaning robot active?", ["No", "Yes"]) == "Yes"),
    "service_robot_active": int(st.sidebar.selectbox("Service robot active?", ["No", "Yes"]) == "Yes"),
}

st.subheader("Individual occupant profiles")
st.write("Both occupants share the same office conditions, but their clothing and activity levels may differ.")
a, b = st.columns(2)
clothing_options = {
    "Light summer clothing": 0.45,
    "Shirt and trousers": 0.65,
    "Shirt + sweater": 0.85,
    "Hoodie + trousers": 1.00,
    "Jacket / heavier winter clothing": 1.20
}
activity_options = {
    "Seated computer work": 1.15,
    "Seated quiet": 1.00,
    "Standing relaxed": 1.20,
    "Walking slowly": 1.70
}
with a:
    st.markdown("### Person A")
    a_cloth = st.selectbox("Clothing", list(clothing_options), index=1, key="a_cloth")
    a_clo = st.number_input("Clothing insulation (clo)", 0.30, 1.80, clothing_options[a_cloth], 0.05, key="a_clo")
    a_act = st.selectbox("Activity", list(activity_options), index=0, key="a_act")
    a_met = st.number_input("Activity level (met)", 0.80, 2.50, activity_options[a_act], 0.05, key="a_met")
with b:
    st.markdown("### Person B")
    b_cloth = st.selectbox("Clothing", list(clothing_options), index=3, key="b_cloth")
    b_clo = st.number_input("Clothing insulation (clo)", 0.30, 1.80, clothing_options[b_cloth], 0.05, key="b_clo")
    b_act = st.selectbox("Activity", list(activity_options), index=0, key="b_act")
    b_met = st.number_input("Activity level (met)", 0.80, 2.50, activity_options[b_act], 0.05, key="b_met")

if st.button("Assess both occupants", type="primary", use_container_width=True):
    people = [
        make_person_input("Person A", a_cloth, a_clo, a_act, a_met, env),
        make_person_input("Person B", b_cloth, b_clo, b_act, b_met, env)
    ]
    results = []
    cols = st.columns(2)
    for col, person in zip(cols, people):
        result = evaluate_person(model, person)
        results.append((person, result))
        pred, risk, pmv, ppd, pmv_error, reasons, actions, label = result
        with col:
            st.markdown(f"### {person['person']}")
            st.subheader(label)
            st.metric("ML demo class", pred.title())
            st.progress(risk, text=f"Model score for synthetic uncomfortable class: {risk:.0%}")
            st.write(f"Clothing: **{person['clothing_clo']:.2f} clo** · Activity: **{person['activity_met']:.2f} met**")
            if pmv is not None:
                st.metric("PMV (ISO 7730 model)", f"{pmv:+.2f}")
                st.metric("PPD (ISO 7730 model)", f"{ppd:.1f}%")
                st.caption("PMV -0.5 to +0.5 is a commonly used thermal acceptability band; PPD is a group estimate, not this person's measured vote.")
            else:
                st.warning("PMV/PPD not available: " + str(pmv_error))
            st.write("**Possible reasons based on entered values**")
            for reason in reasons:
                st.write("• " + reason)
            st.write("**Suggested actions to consider**")
            for action in actions:
                st.write("• " + action)

    if results[0][1][0] != results[1][1][0] or abs(results[0][1][2] or 0 - (results[1][1][2] or 0)) > 0.5:
        st.info("The two occupants may need different responses. Avoid changing the whole-room temperature based on one person's result. Consider occupant-specific local controls.")

st.divider()
st.subheader("Training data and model information")
m1, m2, m3 = st.columns(3)
m1.metric("Training records", f"{len(data):,}")
m2.metric("Synthetic hold-out accuracy", f"{test_accuracy:.1%}")
m3.metric("Input variables", len(FEATURES))
st.caption("Accuracy is measured against synthetic labels created by documented assumptions. It is not real-world human-comfort accuracy.")
with st.expander("View the actual training data used by this model"):
    st.write("This CSV file is loaded from the project folder and used to train the Random Forest classifier.")
    st.dataframe(data.head(30), use_container_width=True)
    st.download_button(
        "Download training data CSV",
        data=data.to_csv(index=False),
        file_name="office_ieq_synthetic_training_data.csv",
        mime="text/csv"
    )
with st.expander("Model validation details"):
    st.write("The model uses a stratified 75% training / 25% testing split. The test split is held out from fitting.")
    st.code(classification_report(y_test, model.predict(X_test), zero_division=0))

st.caption("Research prototype only. Synthetic data are not real office measurements. PMV/PPD estimate thermal sensation for a standardized model occupant and do not constitute a full ASHRAE 55 compliance assessment.")
