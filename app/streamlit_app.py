import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bank Loan Analytics", layout="wide")

# # --- 1. LOAD DATA & MODELS (Cached for Speed) ---
# @st.cache_resource
# def load_data_and_models():
#     # Define Paths
#     base_dir = os.path.dirname(os.path.abspath(__file__)) # Current app folder
#     project_dir = os.path.dirname(base_dir) # One level up
    
#     model_path = os.path.join(project_dir, 'models')
#     data_path = os.path.join(project_dir, 'data', 'processed', 'risk_data_train_ready.csv')
    
#     # Load Models
#     try:
#         model_pd = joblib.load(os.path.join(model_path, 'pd_model.pkl'))
#         model_elig = joblib.load(os.path.join(model_path, 'propensity_model.pkl'))
#         cols_pd = joblib.load(os.path.join(model_path, 'pd_model_cols.pkl'))
#         cols_elig = joblib.load(os.path.join(model_path, 'propensity_model_cols.pkl'))
        
#         # Load Data (First 10k rows for speed in dashboard)
#         df = pd.read_csv(data_path, nrows=10000)
        
#         return df, model_pd, model_elig, cols_pd, cols_elig
#     except Exception as e:
#         st.error(f"Error loading files: {e}")
#         return None, None, None, None, None




# --- 1. LOAD DATA & MODELS (Debug Version) ---
@st.cache_resource
def load_data_and_models():
    # Define Paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_dir = os.path.dirname(base_dir) 
    
    model_path = os.path.join(project_dir, 'models')
    data_path = os.path.join(project_dir, 'data', 'processed', 'risk_data_train_ready.csv')
    
    # --- DEBUG PRINTS (These will show up on the web page) ---
    # st.write(f"📂 Debug: Base App Dir: `{base_dir}`")
    # st.write(f"📂 Debug: Project Root: `{project_dir}`")
    # st.write(f"🔍 Looking for Models in: `{model_path}`")
    # st.write(f"🔍 Looking for Data in: `{data_path}`")
    
    # Check if paths exist
    if not os.path.exists(model_path):
        st.error(f"❌ Error: The 'models' folder was not found at `{model_path}`")
    if not os.path.exists(data_path):
        st.error(f"❌ Error: The data file was not found at `{data_path}`")

    # Load Models
    try:
        model_pd = joblib.load(os.path.join(model_path, 'pd_model.pkl'))
        model_elig = joblib.load(os.path.join(model_path, 'propensity_model.pkl'))
        cols_pd = joblib.load(os.path.join(model_path, 'pd_model_cols.pkl'))
        cols_elig = joblib.load(os.path.join(model_path, 'propensity_model_cols.pkl'))
        
        # Load Data
        df = pd.read_csv(data_path, nrows=10000)
        
        return df, model_pd, model_elig, cols_pd, cols_elig
    except Exception as e:
        st.error(f"❌ Python Error Detail: {e}")
        return None, None, None, None, None







df, model_pd, model_elig, cols_pd, cols_elig = load_data_and_models()

# --- 2. SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Model Simulator"])

# --- 3. MAIN PAGE LOGIC ---
if page == "Project Overview":
    st.title("🏦 Retail Loan Analytics Dashboard")
    st.markdown("""
    This dashboard demonstrates an end-to-end analytics pipeline for:
    * **Propensity Modeling:** Identifying eligible customers.
    * **Credit Risk Modeling:** Predicting Probability of Default (PD).
    * **Profitability Analysis:** Optimizing offers based on expected profit.
    """)
    
    # Status Check
    if df is not None:
        st.success("✅ Models and Data Loaded Successfully!")
        st.write("### Sample Data Preview")
        st.dataframe(df.head())
    else:
        st.error("❌ Failed to load data. Check your file paths.")


elif page == "Model Simulator":
    st.title("🧪 Real-time Scoring Simulator")
    
    # --- 1. INPUT FORM (User enters customer details) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 15000)
        annual_inc = st.number_input("Annual Income ($)", 10000, 300000, 75000)
        term_months = st.selectbox("Term", [36, 60])
        
    with col2:
        fico_score = st.slider("FICO Credit Score", 300, 850, 700)
        dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 100.0, 15.0)
        grade = st.selectbox("Credit Grade (Proxy for Int. Rate)", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        
    with col3:
        emp_length = st.selectbox("Employment Length", ['< 1 year', '1 year', '2 years', '5 years', '10+ years'])
        home_ownership = st.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN'])
        purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])

    # --- 2. RUN PREDICTION BUTTON ---
    if st.button("🚀 Analyze Application"):
        
        # A. Preprocessing Inputs (Recreating Feature Engineering)
        
        # Map Grade to Number (A=0 ... G=6)
        grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
        grade_num = grade_map[grade]
        
        # Map Emp Length
        if '<' in emp_length: emp_num = 0
        elif '+' in emp_length: emp_num = 10
        else: emp_num = int(emp_length.split()[0])
        
        # Calculate Ratios
        loan_to_inc = loan_amnt / (annual_inc + 1)
        
        # Create Raw DataFrame (Single Row)
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': [term_months],
            'int_rate': [10.0 + grade_num*3], # Rough estimate: A=10%, G=28%
            'installment': [loan_amnt / term_months], # Approx
            'grade_num': [grade_num],
            'emp_length_num': [emp_num],
            'home_ownership': [home_ownership],
            'annual_inc': [annual_inc],
            'verification_status': ['Source Verified'], # Default assumption
            'purpose': [purpose],
            'dti': [dti],
            'loan_to_inc': [loan_to_inc],
            'revol_util': [50.0], # Default average
            'total_acc': [20],   # Default average
            'pub_rec_bankruptcies': [0.0],
            'addr_state': ['CA'], # Default
            'risk_score': [fico_score] # For eligibility model
        })

        # B. Align Columns with Model (The "Tricky" Part)
        # The model expects specific One-Hot Encoded columns (e.g., purpose_medical).
        # We use reindex to ensure our single row matches the training data structure exactly.
        
        # -- Prepare for PD Model --
        input_encoded_pd = pd.get_dummies(input_data)
        input_pd_final = input_encoded_pd.reindex(columns=cols_pd, fill_value=0)
        
        # -- Prepare for Eligibility Model --
        input_encoded_elig = pd.get_dummies(input_data)
        input_elig_final = input_encoded_elig.reindex(columns=cols_elig, fill_value=0)

        # C. Predict
        prob_default = model_pd.predict_proba(input_pd_final)[0][1] # Probability of Default (Class 1)
        prob_elig = model_elig.predict_proba(input_elig_final)[0][1] # Probability of Approval (Class 1)

        # D. Calculate Expected Profit
        # Profit = (Interest Income * (1 - PD)) - (Loss * PD) - Cost
        interest_income = loan_amnt * (input_data['int_rate'][0]/100) * (term_months/12) #* 0.5 # Approx net margin
        loss_given_default = loan_amnt * 0.6 # Assuming 60% loss if they default
        
        expected_loss = prob_default * loss_given_default
        expected_gain = (1 - prob_default) * interest_income
        
        expected_profit = expected_gain - expected_loss - 50 # Subtract $50 acquisition cost

        # --- 3. DISPLAY RESULTS ---
        st.divider()
        st.subheader("📊 Analysis Results")
        
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Eligibility Score", f"{prob_elig:.1%}", delta_color="normal")
        
        # Color Logic for Risk
        risk_color = "normal"
        if prob_default < 0.10: risk_color = "normal" # Green/Good
        else: risk_color = "inverse" # Red/Bad
            
        c2.metric("Default Risk (PD)", f"{prob_default:.1%}", delta="-Risk", delta_color=risk_color)
        
        c3.metric("Expected Profit", f"${expected_profit:,.0f}")
        
        # Final Recommendation
        if prob_elig < 0.5:
            st.error("🔴 **Recommendation: DECLINE** (Eligibility Score too low)")
        elif expected_profit < 0:
            st.warning("🟠 **Recommendation: REVIEW** (Customer Eligible, but expected profit is negative)")
        else:
            st.success("🟢 **Recommendation: APPROVE** (High Profit, Manageable Risk)")