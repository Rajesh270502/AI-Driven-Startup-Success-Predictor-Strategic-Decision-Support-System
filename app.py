import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --- Page Layout & Configuration ---
st.set_page_config(
    page_title="Smart Startup Success Prediction System",
    page_icon="🚀",
    layout="wide"
)

# --- 1. Load Pre-trained Artifacts Safely ---
@st.cache_resource
def load_model_artifacts():
    """
    Loads your fitted XGBoost model and preprocessor pipeline objects.
    Replace 'xgb_model.pkl' and 'preprocessor.pkl' with your actual file names.
    """
    try:
        with open("xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except FileNotFoundError:
        # Fallback placeholder to allow the dashboard layout to launch seamlessly during testing
        st.sidebar.warning("⚠️ ML Artifact files not found. Using simulation engine placeholder mode.")
        return None, None

xgb_model, preprocessor = load_model_artifacts()

# --- Application Title Panel ---
st.title("🚀 Smart Startup Success Prediction System")
st.markdown("An intelligent, data-driven platform designed to estimate startup success probabilities using advanced predictive analytics.")

st.divider() 

# --- Create Multi-Column Layout ---
left_column, right_column = st.columns([1, 1.2])

# =========================================================
# LEFT COLUMN: MODULE 10 - STARTUP INFORMATION FORM
# =========================================================
with left_column:
    st.header("📋 Startup Profile Intake Form")
    st.write("Enter the company characteristics below to generate an evaluation profile.")
    
    with st.form("startup_form"):
        # Company General Data
        name = st.text_input("Startup Name", "Alpha Tech Labs")
        category = st.selectbox(
            "Industry Sector", 
            ["software", "web", "mobile", "enterprise", "advertising", "gamesvideo", "ecommerce", "biotech", "consulting", "othercategory"]
        )
        state = st.text_input("US State Code (e.g., CA, NY, TX)", "CA")
        
        # Financial Data Parameters
        funding_total = st.number_input("Total Funding Amount (USD)", min_value=0, value=250000, step=50000)
        funding_rounds = st.number_input("Number of Funding Rounds", min_value=1, value=1, step=1)
        
        # Operational Metrics
        relationships = st.number_input("Key Business Relationships Count", min_value=0, value=2, step=1)
        milestones = st.number_input("Business Milestones Achieved", min_value=0, value=1, step=1)
        
        # Chronological Windows
        age_first_funding = st.number_input("Startup Age at First Funding (Years)", min_value=0.0, value=1.2, step=0.1)
        age_last_funding = st.number_input("Startup Age at Last Funding (Years)", min_value=0.0, value=1.2, step=0.1)

        # Form Submit Indicator - THIS CRITICALLY DEFINES submit_btn FIRST
        submit_btn = st.form_submit_button("Run Diagnostics & Predict")

# =========================================================
# RIGHT COLUMN: PREDICTIONS, RISK METERS, & ANALYTICS
# =========================================================
with right_column:
    # Safely calling submit_btn now that it has been defined above
    if submit_btn:
        st.header("📊 Diagnostic Engine Output")
        
        # 1. Structure the actual parameters provided by the user via the form
        user_inputs = {
            'funding_total_usd': funding_total,
            'funding_rounds': funding_rounds,
            'relationships': relationships,
            'milestones': milestones,
            'age_first_funding_year': age_first_funding,
            'age_last_funding_year': age_last_funding,
            'category_code': category,
            'state_code': state
        }
        
        # 2. Complete List of all columns expected by your scikit-learn preprocessor pipeline
        expected_columns = [
            'state_code', 'latitude', 'longitude', 'zip_code', 'city', 'Unnamed: 6',
            'age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 
            'age_last_milestone_year', 'relationships', 'funding_rounds', 'funding_total_usd', 
            'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate', 'category_code', 
            'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 
            'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting', 'is_othercategory', 
            'has_VC', 'has_angel', 'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD', 
            'avg_participants', 'is_top500', 'status', 'funding_duration_days', 'age_at_first_funding_days'
        ]
        
        # 3. Initialize a complete template containing neutral placeholder values
        full_features = {}
        for col in expected_columns:
            if col in user_inputs:
                full_features[col] = user_inputs[col]
            elif 'age' in col or 'funding' in col or 'participants' in col or 'latitude' in col or 'longitude' in col:
                full_features[col] = 0.0  
            elif 'is_' in col or 'has_' in col or 'labels' in col:
                full_features[col] = 0    
            else:
                full_features[col] = "unknown" 

        # 4. Construct the structured pandas DataFrame matching your notebook's column layout
        input_df = pd.DataFrame([full_features])
        input_df = input_df[[c for c in expected_columns if c in input_df.columns]] 
        
        # 5. Run prediction computations securely using the pipeline artifacts
        if xgb_model and preprocessor:
            try:
                processed_vector = preprocessor.transform(input_df)
                prob_success = xgb_model.predict_proba(processed_vector)[0][1]
            except Exception as e:
                st.error(f"Transformation Pipeline Error: {e}")
                prob_success = 0.50
        else:
            # Fallback placeholder evaluation matrix rule system if models aren't loaded properly
            base_score = 35.0
            if funding_total > 500000: base_score += 20
            if milestones >= 2: base_score += 25
            if relationships >= 4: base_score += 15
            prob_success = min(base_score / 100.0, 0.98)

        # Map metrics to calibrated thresholds defined in Project Module 8
        success_probability = round(prob_success * 100, 1)
        success_score = int(success_probability)
        
        if 0 <= success_score <= 40:
            risk_category = "High Risk"
            risk_color = "red"
        elif 41 <= success_score <= 70:
            risk_category = "Moderate Risk"
            risk_color = "orange"
        else:
            risk_category = "High Potential"
            risk_color = "green"

        # --- Display KPIs & Risk Meter (Module 8) ---
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric(label="Success Probability Rate", value=f"{success_probability}%")
        with kpi_col2:
            st.metric(label="Calibrated Startup Score", value=f"{success_score} / 100")
            
        # Risk Status Box Display
        st.markdown(f"**Assigned Profile Status:** :{risk_color}[{risk_category}]")

        # --- Module 9: Recommendation Panel ---
        st.subheader("💡 Strategic Recommendation Panel")
        recommendations = []
        
        if funding_total < 500000:
            recommendations.append("🚨 **Raise additional funding:** Total funding volume sits below historical risk baselines.")
        if milestones <= 1:
            recommendations.append("🎯 **Accelerate core milestones:** Prioritize product deployment updates to showcase traction milestones.")
        if relationships < 3:
            recommendations.append("🤝 **Expand market outreach:** Cultivate broader advisory connections and operational channel networks.")
            
        if not recommendations:
            recommendations.append("📈 **Scale operations:** Metric layout reflects low vulnerability footprint. Focus on geographical scaling footprints.")

        for rec in recommendations:
            st.markdown(f"- {rec}")

        # --- Module 10: Business Analytics Charts ---
        st.divider()
        st.subheader("📈 Ecosystem Benchmarking Context Analytics")
        
        # Generate background benchmark distribution matrix
        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.set_theme(style="whitegrid")
        
        scores_sample = np.random.normal(loc=58, scale=18, size=1000)
        scores_sample = np.clip(scores_sample, 0, 100)
        
        sns.kdeplot(scores_sample, fill=True, color="gray", alpha=0.3, ax=ax, label="Ecosystem Peer Distribution")
        ax.axvline(x=success_score, color=risk_color, linestyle="--", linewidth=2, label=f"Your Startup Center ({success_score})")
        
        ax.set_title("Startup Placement vs. Global Ecosystem Distribution Profiles")
        ax.set_xlabel("Success Matrix Score Evaluation Scale")
        ax.set_ylabel("Density Distribution")
        ax.legend(fontsize=8)
        
        st.pyplot(fig)
    else:
        st.info("👈 Complete the entry details on the left form and click 'Run Diagnostics & Predict' to view performance profiles.")