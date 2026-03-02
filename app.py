import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from scipy.optimize import minimize
import google.generativeai as genai
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="Distillation Digital Twin", layout="wide", page_icon="🧪")
st.title("🧪 AI-Powered Distillation Column Digital Twin")

# --- SIDEBAR & API SETUP ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-1.5-pro-latest')
else:
    st.sidebar.warning("Please enter your Gemini API Key to enable AI insights.")

# --- SESSION STATE INITIALIZATION ---
if 'stitched_data' not in st.session_state:
    st.session_state.stitched_data = None
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
if 'chemistry_context' not in st.session_state:
    st.session_state.chemistry_context = ""

# --- HELPER FUNCTIONS ---
@st.cache_data
def process_quality_sheet(df):
    """Combines Date and Time columns into a Timestamp and sorts."""
    # Find Date and Time columns dynamically (case-insensitive)
    cols = [c.lower() for c in df.columns]
    date_col = df.columns[cols.index('date')] if 'date' in cols else None
    time_col = df.columns[cols.index('time')] if 'time' in cols else None
    
    if date_col and time_col:
        # Create Timestamp, handle different formats
        df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
        df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')
    return df

def generate_ai_context(feed_cols, top_cols, bot_cols):
    """Uses LLM to understand the chemistry based on GC columns."""
    if not api_key: return "API Key not provided."
    prompt = f"""
    You are an expert Chemical Engineer. Analyze the following GC composition columns from a distillation process.
    Feed contains: {feed_cols}
    Top product contains: {top_cols}
    Bottom product contains: {bot_cols}
    
    1. Identify the likely primary chemicals being separated.
    2. Briefly state the key thermodynamic principles/challenges for this specific separation (e.g., relative volatility, azeotropes).
    3. List 2 common safety or operational limits to watch out for (e.g., column flooding, temperature degradation).
    Keep it concise and technical.
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# --- MAIN APP TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Upload & Stitching", 
    "📊 Exploratory Data Analysis", 
    "🤖 AI Process Context", 
    "⚙️ Simulation (Digital Twin)", 
    "🚀 Process Optimization"
])

# ==========================================
# TAB 1: DATA UPLOAD & STITCHING
# ==========================================
with tab1:
    st.header("Upload Process & Quality Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Process Data (DCS/PLC)")
        process_file = st.file_uploader("Upload continuous process data (CSV)", type=['csv'])
    
    with col2:
        st.subheader("2. Quality Data (Lab GC)")
        quality_file = st.file_uploader("Upload Lab data (Excel with Feed, Top, Bottom sheets)", type=['xlsx', 'xls'])

    if process_file and quality_file:
        try:
            # Load Process Data
            df_process = pd.read_csv(process_file)
            # Find timestamp column dynamically
            ts_col = [c for c in df_process.columns if 'time' in c.lower() or 'date' in c.lower()][0]
            df_process[ts_col] = pd.to_datetime(df_process[ts_col], errors='coerce')
            df_process = df_process.dropna(subset=[ts_col]).sort_values(ts_col).rename(columns={ts_col: 'Timestamp'})
            
            # Load Quality Data Sheets
            xls = pd.ExcelFile(quality_file)
            sheet_names = [s.lower() for s in xls.sheet_names]
            
            feed_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('feed')]) if 'feed' in sheet_names else pd.DataFrame())
            top_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('top')]) if 'top' in sheet_names else pd.DataFrame())
            bot_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('bottom')]) if 'bottom' in sheet_names else pd.DataFrame())
            
            # Extract column names for AI Context
            feed_cols = list(feed_df.columns) if not feed_df.empty else []
            top_cols = list(top_df.columns) if not top_df.empty else []
            bot_cols = list(bot_df.columns) if not bot_df.empty else []

            st.success("Files loaded successfully. Stitching data...")
            
            # Merge logic: merge_asof matches the nearest timestamp backwards
            df_stitched = df_process.copy()
            if not feed_df.empty:
                df_stitched = pd.merge_asof(df_stitched, feed_df.drop(columns=['Date', 'Time'], errors='ignore'), on='Timestamp', direction='backward', suffixes=('', '_feed'))
            if not top_df.empty:
                df_stitched = pd.merge_asof(df_stitched, top_df.drop(columns=['Date', 'Time'], errors='ignore'), on='Timestamp', direction='backward', suffixes=('', '_top'))
            if not bot_df.empty:
                df_stitched = pd.merge_asof(df_stitched, bot_df.drop(columns=['Date', 'Time'], errors='ignore'), on='Timestamp', direction='backward', suffixes=('', '_bot'))
            
            # Forward fill lab data to simulate "holding" the last known quality until the next sample
            df_stitched = df_stitched.ffill().dropna() 
            
            st.session_state.stitched_data = df_stitched
            st.dataframe(df_stitched.head())
            
            # Generate AI Context in the background
            if not st.session_state.chemistry_context:
                with st.spinner("Analyzing process chemistry..."):
                    st.session_state.chemistry_context = generate_ai_context(feed_cols, top_cols, bot_cols)
                    
        except Exception as e:
            st.error(f"Error processing files: {e}")

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    if st.session_state.stitched_data is not None:
        df = st.session_state.stitched_data
        all_cols = [c for c in df.columns if c != 'Timestamp']
        
        # 2. Data Filtering
        st.subheader("Data Filtering")
        col_date1, col_date2 = st.columns(2)
        min_date, max_date = df['Timestamp'].min(), df['Timestamp'].max()
        start_date = col_date1.date_input("Start Date", min_date)
        end_date = col_date2.date_input("End Date", max_date)
        
        mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)
        df_filtered = df.loc[mask]
        
        # 3. Summary Stats
        with st.expander("Summary Statistics"):
            st.dataframe(df_filtered.describe())

        # 1. Trends over time
        st.subheader("1. Time Series Trends")
        trend_cols = st.multiselect("Select parameters to plot over time", all_cols, default=all_cols[:2] if len(all_cols)>1 else all_cols)
        if trend_cols:
            fig = px.line(df_filtered, x='Timestamp', y=trend_cols, title="Process & Quality Trends")
            st.plotly_chart(fig, use_container_width=True)
            
        colA, colB = st.columns(2)
        
        with colA:
            # 4. Distributions (Histograms & Boxplots)
            st.subheader("2. Distributions")
            dist_col = st.selectbox("Select parameter for Distribution", all_cols)
            fig_hist = px.histogram(df_filtered, x=dist_col, marginal="box", title=f"Distribution of {dist_col}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with colB:
            # 5. Scatterplot
            st.subheader("3. Scatter Plot")
            scat_x = st.selectbox("X-axis", all_cols, index=0)
            scat_y = st.selectbox("Y-axis", all_cols, index=1 if len(all_cols)>1 else 0)
            fig_scat = px.scatter(df_filtered, x=scat_x, y=scat_y, trendline="ols", title=f"{scat_y} vs {scat_x}")
            st.plotly_chart(fig_scat, use_container_width=True)
            
        # 6. Correlation Heatmap
        st.subheader("4. Correlation Heatmap")
        corr_cols = st.multiselect("Select parameters for Correlation", all_cols, default=all_cols[:6] if len(all_cols)>5 else all_cols)
        if len(corr_cols) > 1:
            corr = df_filtered[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Please upload and stitch data in Tab 1 first.")

# ==========================================
# TAB 3: AI PROCESS CONTEXT
# ==========================================
with tab3:
    st.header("🧠 Process Chemistry & Physics Context")
    if st.session_state.chemistry_context:
        st.markdown(st.session_state.chemistry_context)
        st.info("This context is actively used by the Digital Twin during Simulation and Optimization to ensure recommendations are physically viable and safe.")
    else:
        st.info("Upload data and provide API key to generate context.")

# ==========================================
# TAB 4: SIMULATION (DIGITAL TWIN)
# ==========================================
with tab4:
    if st.session_state.stitched_data is not None:
        st.header("⚙️ Train Digital Twin & Simulate")
        df = st.session_state.stitched_data
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Variable (e.g., Top Quality, Steam Norm)", all_numeric)
        with col2:
            features = st.multiselect("Select Process Parameters (Features)", [c for c in all_numeric if c != target])
            
        if st.button("Train Digital Twin Model") and features:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Chronological split
            
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            
            st.session_state.ml_model = model
            st.session_state.features = features
            st.session_state.target = target
            
            st.success(f"Model Trained Successfully! R² Score on Test Data: {r2:.3f}")
            
        st.divider()
        
        if st.session_state.ml_model is not None:
            st.subheader("🎛️ Process Simulator")
            st.write("Adjust the process parameters below to simulate the impact on the target variable.")
            
            input_data = {}
            sim_cols = st.columns(3)
            # Create sliders for each feature based on historical min/max/mean
            for i, feat in enumerate(st.session_state.features):
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                mean_val = float(df[feat].mean())
                with sim_cols[i % 3]:
                    input_data[feat] = st.slider(feat, min_value=min_val, max_value=max_val, value=mean_val)
                    
            if st.button("Run Simulation"):
                input_df = pd.DataFrame([input_data])
                prediction = st.session_state.ml_model.predict(input_df)[0]
                
                st.metric(label=f"Predicted {st.session_state.target}", value=f"{prediction:.4f}")
                
                # AI Validation
                if api_key:
                    with st.spinner("AI checking engineering constraints..."):
                        sim_prompt = f"""
                        Context: {st.session_state.chemistry_context}
                        
                        The operator simulated the following setpoints:
                        {input_data}
                        The ML model predicted a {st.session_state.target} of {prediction}.
                        
                        Based on chemical engineering principles, does this prediction seem thermodynamically sound? 
                        Are there any risks of flooding, weeping, or off-spec material with these setpoints? Provide a 3-sentence technical advisory.
                        """
                        response = llm_model.generate_content(sim_prompt)
                        st.info(f"**AI Process Engineer Advisory:**\n{response.text}")
    else:
        st.info("Please upload data in Tab 1.")

# ==========================================
# TAB 5: PROCESS OPTIMIZATION
# ==========================================
with tab5:
    if st.session_state.ml_model is not None:
        st.header("🚀 AI-Driven Process Optimization")
        st.write("Find the best operating regime to optimize your target variable.")
        
        opt_goal = st.radio("Optimization Goal", ["Maximize Target", "Minimize Target"])
        
        st.subheader("Set Constraint Bounds")
        bounds = []
        bound_inputs = {}
        df = st.session_state.stitched_data
        
        # Let user define safe operating envelopes for the optimizer
        for feat in st.session_state.features:
            col1, col2 = st.columns(2)
            min_hist = float(df[feat].min())
            max_hist = float(df[feat].max())
            with col1:
                lower = st.number_input(f"{feat} Lower Bound", value=min_hist)
            with col2:
                upper = st.number_input(f"{feat} Upper Bound", value=max_hist)
            bounds.append((lower, upper))
            bound_inputs[feat] = (lower, upper)
            
        if st.button("Run Optimizer"):
            with st.spinner("Finding optimal operating regime..."):
                # Objective function for scipy
                def objective(x):
                    pred = st.session_state.ml_model.predict(pd.DataFrame([x], columns=st.session_state.features))[0]
                    # Minimize returns the lowest value. If we want to maximize, we return negative prediction.
                    return -pred if opt_goal == "Maximize Target" else pred
                
                # Initial guess (means of bounds)
                x0 = [(b[0] + b[1]) / 2 for b in bounds]
                
                # Run L-BFGS-B optimization
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
                
                if res.success:
                    st.success("Optimization Converged!")
                    opt_pred = -res.fun if opt_goal == "Maximize Target" else res.fun
                    
                    st.metric(label=f"Optimized {st.session_state.target}", value=f"{opt_pred:.4f}")
                    
                    st.subheader("Recommended Setpoints")
                    rec_data = {feat: val for feat, val in zip(st.session_state.features, res.x)}
                    st.json(rec_data)
                    
                    # LLM Sanity Check of Optimized Results
                    if api_key:
                        opt_prompt = f"""
                        Context: {st.session_state.chemistry_context}
                        
                        An algorithm optimized the distillation process to {opt_goal} {st.session_state.target}.
                        Recommended setpoints are: {rec_data}
                        Predicted outcome: {opt_pred}
                        
                        As a Senior Chemical Engineer, review these setpoints. Do they push the equipment too close to the edge of the operating envelope? Would you approve taking these setpoints to the actual plant DCS? Explain why or why not.
                        """
                        opt_response = llm_model.generate_content(opt_prompt)
                        st.warning(f"**Safety & Engineering Review:**\n{opt_response.text}")
                else:
                    st.error("Optimization failed to converge. Try relaxing the bounds.")
    else:
        st.info("Please train the Digital Twin model in Tab 4 first.")
