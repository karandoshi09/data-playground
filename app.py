import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
from scipy.optimize import minimize
import shap

# --- PAGE SETUP ---
st.set_page_config(page_title="Distillation Digital Twin (PoC)", layout="wide", page_icon="🧪")
st.title("🧪 Distillation Column Digital Twin (PoC)")
st.caption("Secure, offline Data Science and Simulation environment for Process Engineering.")

# --- SESSION STATE INITIALIZATION ---
for key in ['stitched_data', 'filtered_data', 'ml_model', 'features', 'target', 'X_train', 'X_test']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'features' else []

# --- HELPER FUNCTIONS ---
@st.cache_data
def process_quality_sheet(df, suffix):
    cols = [c.lower() for c in df.columns]
    date_col = df.columns[cols.index('date')] if 'date' in cols else None
    time_col = df.columns[cols.index('time')] if 'time' in cols else None
    
    if date_col and time_col:
        # 1. Clean the string representations
        date_series = df[date_col].astype(str).str.replace('.', '/', regex=False)
        time_series = df[time_col].astype(str)
        
        # 2. Force strict datetime parsing. We use dayfirst=True, but we also specify 
        # format='%d/%m/%Y %H:%M:%S' as a fallback if dayfirst gets confused.
        datetime_str = date_series + ' ' + time_series
        
        try:
             # Try explicit format first if you know it's strictly DD/MM/YYYY
             df['Timestamp'] = pd.to_datetime(datetime_str, format='%d/%m/%Y %H:%M:%S', errors='coerce')
        except:
             # Fallback to dayfirst if the time format is irregular (e.g., missing seconds)
             df['Timestamp'] = pd.to_datetime(datetime_str, dayfirst=True, errors='coerce')
             
        df = df.dropna(subset=['Timestamp']).drop(columns=[date_col, time_col])
        
        num_cols = df.columns.drop('Timestamp')
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        
        df['Timestamp'] = df['Timestamp'].dt.round('30min')
        df = df.groupby('Timestamp').mean(numeric_only=True).reset_index()
        df = df.rename(columns={c: f"{c}_{suffix}" for c in df.columns if c != 'Timestamp'})
    return df

# --- MAIN APP TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Upload & Engineering", 
    "📊 EDA & Filtering", 
    "📈 Quartile Analysis", 
    "⚙️ Simulation & Interpretability", 
    "🚀 Optimization"
])


# ==========================================
# TAB 1: DATA UPLOAD & ENGINEERING
# ==========================================
with tab1:
    st.header("1. Upload Process & Quality Data")
    col1, col2 = st.columns(2)
    
    with col1: process_file = st.file_uploader("Upload Process Data (CSV/XLSX)", type=['csv', 'xlsx', 'xls'])
    with col2: quality_file = st.file_uploader("Upload Quality Lab Data (Excel)", type=['xlsx', 'xls'])

    if process_file and quality_file:
        try:
            with st.spinner("Processing & Stitching Data..."):
                # --- 1. Process Data Ingestion ---
                if process_file.name.endswith('.csv'): df_process = pd.read_csv(process_file)
                else: df_process = pd.read_excel(process_file)
                    
                ts_col = [c for c in df_process.columns if 'time' in c.lower() or 'date' in c.lower()][0]
                ts_series = df_process[ts_col].astype(str).str.replace('.', '/', regex=False)
                df_process['Timestamp'] = pd.to_datetime(ts_series, dayfirst=True, errors='coerce')
                
                if ts_col != 'Timestamp': df_process = df_process.drop(columns=[ts_col])
                df_process = df_process.dropna(subset=['Timestamp'])
                
                proc_num_cols = df_process.columns.drop('Timestamp')
                df_process[proc_num_cols] = df_process[proc_num_cols].apply(pd.to_numeric, errors='coerce')
                df_process = df_process.set_index('Timestamp').resample('30min').mean(numeric_only=True).reset_index()
                
                # --- 2. Quality Data Ingestion ---
                xls = pd.ExcelFile(quality_file)
                sheet_names = [s.lower() for s in xls.sheet_names]
                
                feed_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('feed')]) if 'feed' in sheet_names else pd.DataFrame(), 'feed')
                top_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('top')]) if 'top' in sheet_names else pd.DataFrame(), 'top')
                bot_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('bottom')]) if 'bottom' in sheet_names else pd.DataFrame(), 'bot')
                
                # --- 3. Merging ---
                df_stitched = df_process.copy()
                if not feed_df.empty: df_stitched = pd.merge(df_stitched, feed_df, on='Timestamp', how='left')
                if not top_df.empty: df_stitched = pd.merge(df_stitched, top_df, on='Timestamp', how='left')
                if not bot_df.empty: df_stitched = pd.merge(df_stitched, bot_df, on='Timestamp', how='left')
                
                # --- 4. Interpolation ---
                qual_cols = [c for c in df_stitched.columns if c.endswith('_feed') or c.endswith('_top') or c.endswith('_bot')]
                df_stitched[qual_cols] = df_stitched[qual_cols].interpolate(method='linear', limit=4, limit_direction='forward')
                df_stitched = df_stitched.dropna(subset=qual_cols, how='all')

            st.success("✅ Data Cleaned, Stitched, and Interpolated Successfully!")
            st.divider()

            # --- DISPLAY DATA TABLES AND STATS ---
            st.subheader("Data Overview (Post-Stitching)")
            
            # Format timestamp for better Streamlit display
            display_df = df_stitched.copy()
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            c_top, c_bot = st.columns(2)
            with c_top:
                st.write("**Top 10 Rows:**")
                st.dataframe(display_df.head(10))
            with c_bot:
                st.write("**Bottom 10 Rows:**")
                st.dataframe(display_df.tail(10))
                
            st.write("**Summary Statistics:**")
            st.dataframe(df_stitched.describe())
            
            st.divider()
            
            # --- Feature Engineering UI ---
            st.subheader("2. Engineer Physics Features")
            
            all_cols = ['None'] + list(df_stitched.columns)
            c1, c2, c3, c4 = st.columns(4)
            feed_c = c1.selectbox("Feed Flow", all_cols)
            top_c = c2.selectbox("Distillate (Top) Flow", all_cols)
            steam_c = c3.selectbox("Steam Flow", all_cols)
            reflux_c = c4.selectbox("Reflux Flow", all_cols)
            
            c5, c6, c7, c8 = st.columns(4)
            t1_c = c5.selectbox("1st Bed Temp", all_cols)
            t2_c = c6.selectbox("Last Bed Temp", all_cols)
            p1_c = c7.selectbox("1st Bed Pressure", all_cols)
            p2_c = c8.selectbox("Last Bed Pressure", all_cols)
            
            if st.button("Generate Features"):
                if top_c != 'None' and feed_c != 'None': df_stitched['D/F_Ratio'] = df_stitched[top_c] / (df_stitched[feed_c] + 1e-9)
                if steam_c != 'None' and feed_c != 'None': df_stitched['Steam/Feed_Ratio'] = df_stitched[steam_c] / (df_stitched[feed_c] + 1e-9)
                if steam_c != 'None' and top_c != 'None': df_stitched['Steam/Top_Ratio'] = df_stitched[steam_c] / (df_stitched[top_c] + 1e-9)
                if reflux_c != 'None' and top_c != 'None': df_stitched['Reflux_Ratio'] = df_stitched[reflux_c] / (df_stitched[top_c] + 1e-9)
                if t1_c != 'None' and t2_c != 'None': df_stitched['Delta_T'] = df_stitched[t1_c] - df_stitched[t2_c]
                if p1_c != 'None' and p2_c != 'None': df_stitched['Delta_P'] = df_stitched[p1_c] - df_stitched[p2_c]
                
                st.session_state.stitched_data = df_stitched
                st.session_state.filtered_data = df_stitched.copy()
                st.success("Features Generated! Go to Tab 2.")
                
        except Exception as e:
            st.error(f"Error processing files: {e}")

# ==========================================
# TAB 2: EDA & FILTERING
# ==========================================
with tab2:
    if st.session_state.stitched_data is not None:
        df = st.session_state.stitched_data
        st.subheader("🧹 Outlier Filtering")
        filter_cols = st.multiselect("Select columns to filter:", df.columns.drop('Timestamp'))
        
        filters = {}
        if filter_cols:
            f_cols = st.columns(min(len(filter_cols), 3))
            for i, col in enumerate(filter_cols):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                with f_cols[i % 3]: filters[col] = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
        
        df_filtered = df.copy()
        for col, (f_min, f_max) in filters.items():
            df_filtered = df_filtered[(df_filtered[col] >= f_min) & (df_filtered[col] <= f_max)]
        
        st.session_state.filtered_data = df_filtered
        st.write(f"Data points remaining: **{len(df_filtered)}** out of **{len(df)}**")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            trend_cols = st.multiselect("Time Series Trends", df_filtered.columns.drop('Timestamp').tolist(), default=df_filtered.columns.drop('Timestamp').tolist()[:1])
            if trend_cols: st.plotly_chart(px.line(df_filtered, x='Timestamp', y=trend_cols), use_container_width=True)
        with c2:
            scat_x = st.selectbox("Scatter X-axis", df_filtered.columns.drop('Timestamp').tolist(), index=0)
            scat_y = st.selectbox("Scatter Y-axis", df_filtered.columns.drop('Timestamp').tolist(), index=1)
            st.plotly_chart(px.scatter(df_filtered, x=scat_x, y=scat_y, trendline="ols"), use_container_width=True)
    else:
        st.info("Upload and process data in Tab 1 first.")

# ==========================================
# TAB 3: QUARTILE ANALYSIS (Bivariate)
# ==========================================
with tab3:
    if st.session_state.filtered_data is not None:
        df_q = st.session_state.filtered_data.copy()
        numeric_cols = df_q.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1: q_target = st.selectbox("Select Target Variable to Bin", numeric_cols)
        with col2: n_bins = st.slider("Number of Bins", min_value=2, max_value=10, value=4)
        q_features = st.multiselect("Select Parameters to Analyze", [c for c in numeric_cols if c != q_target])
        
        if q_target and q_features:
            try:
                df_q['Target_Bins'] = pd.qcut(df_q[q_target], q=n_bins, duplicates='drop')
                df_q['Bin_Label'] = df_q['Target_Bins'].astype(str)
                df_q = df_q.sort_values('Target_Bins')
                
                for feature in q_features:
                    fig = px.box(df_q, x='Bin_Label', y=feature, color='Bin_Label', title=f"Directionality of {feature} relative to {q_target}")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                summary_stat = st.radio("Select Statistic to Display", ["Median", "Mean"])
                summary_df = df_q.groupby('Bin_Label')[q_features].median().reset_index() if summary_stat == "Median" else df_q.groupby('Bin_Label')[q_features].mean().reset_index()
                st.dataframe(summary_df.style.background_gradient(cmap='Blues', axis=0))
            except ValueError as e:
                st.warning(f"Could not create bins for {q_target}. Error: {e}")
    else:
        st.info("Upload and filter data first.")

# ==========================================
# TAB 4: SIMULATION & INTERPRETABILITY
# ==========================================
with tab4:
    if st.session_state.filtered_data is not None:
        st.header("⚙️ Train Digital Twin & Simulate")
        df_train = st.session_state.filtered_data
        all_numeric = df_train.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1: target = st.selectbox("Select Target Variable for ML Model", all_numeric)
        with col2: features = st.multiselect("Select Process Parameters (Features)", [c for c in all_numeric if c != target])

        if st.button("Train Digital Twin Model") and features:
            with st.spinner("Training XGBoost and calculating SHAP values..."):
                
                # --- THE FIX: Drop rows where the Target variable is NaN ---
                df_ml = df_train.dropna(subset=[target])
                
                # Safety check: ensure we still have enough data to train
                if len(df_ml) < 20:
                    st.error(f"Not enough valid data points ({len(df_ml)}) for target '{target}'. Your lab data might be too sparse. Try a different target or increase the interpolation limit in Tab 1.")
                else:
                    X = df_ml[features]
                    y = df_ml[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    
                    # Initialize and train the model
                    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                    model.fit(X_train, y_train)
                    
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    
                    # Save to session state
                    st.session_state.ml_model = model
                    st.session_state.features = features
                    st.session_state.target = target
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    
                    st.success(f"Model Trained! Test Data R² Score: {r2:.3f}")
            
        if st.session_state.ml_model is not None:
            st.divider()
            
            # --- SHAP & PDP SECTION ---
            st.subheader("📊 Model Interpretability")
            
            # 1. Calculate SHAP
            explainer = shap.TreeExplainer(st.session_state.ml_model)
            shap_values = explainer.shap_values(st.session_state.X_test)
            
            col_shap, col_pdp = st.columns(2)
            
            with col_shap:
                st.markdown("**Feature Importance (SHAP Summary)**")
                st.caption("Shows which parameters drive the target and in which direction.")
                fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
                shap.summary_plot(shap_values, st.session_state.X_test, show=False, plot_size=(6,4))
                st.pyplot(fig_shap)
                plt.clf()
            
            with col_pdp:
                st.markdown("**Partial Dependence Plots (Top 3 Parameters)**")
                st.caption("Shows the isolated marginal effect of the top parameters on the target.")
                
                # Extract Top 3 Features based on mean absolute SHAP value
                mean_shap_vals = np.abs(shap_values).mean(axis=0)
                shap_importance = pd.DataFrame(list(zip(st.session_state.features, mean_shap_vals)), columns=['feature', 'importance'])
                shap_importance = shap_importance.sort_values(by='importance', ascending=False)
                top_3_features = shap_importance['feature'].head(3).tolist()
                
                if top_3_features:
                    fig_pdp, ax_pdp = plt.subplots(figsize=(6, 4))
                    # Adjust layout based on number of features (up to 3)
                    n_cols = min(len(top_3_features), 3)
                    PartialDependenceDisplay.from_estimator(
                        st.session_state.ml_model, 
                        st.session_state.X_train, 
                        top_3_features, 
                        ax=ax_pdp, 
                        n_cols=n_cols,
                        grid_resolution=20
                    )
                    fig_pdp.tight_layout()
                    st.pyplot(fig_pdp)
                    plt.clf()
            
            st.divider()
            
            # --- SIMULATOR SECTION ---
            st.subheader("🎛️ Process Simulator")
            input_data = {}
            sim_cols = st.columns(3)
            for i, feat in enumerate(st.session_state.features):
                min_val = float(st.session_state.X_train[feat].min())
                max_val = float(st.session_state.X_train[feat].max())
                mean_val = float(st.session_state.X_train[feat].mean())
                with sim_cols[i % 3]: input_data[feat] = st.slider(feat, min_val, max_val, mean_val)
                    
            if st.button("Run Simulation"):
                input_df = pd.DataFrame([input_data])
                prediction = st.session_state.ml_model.predict(input_df)[0]
                st.metric(label=f"Predicted {st.session_state.target}", value=f"{prediction:.4f}")
    else:
        st.info("Please upload data in Tab 1.")

# ==========================================
# TAB 5: OPTIMIZATION
# ==========================================
with tab5:
    if st.session_state.ml_model is not None:
        st.header("🚀 Process Optimization")
        opt_goal = st.radio("Optimization Goal", ["Maximize Target", "Minimize Target"])
        st.subheader("Set Constraint Bounds")
        
        bounds = []
        df_train = st.session_state.filtered_data
        
        for feat in st.session_state.features:
            col1, col2 = st.columns(2)
            min_hist = float(df_train[feat].min())
            max_hist = float(df_train[feat].max())
            with col1: lower = st.number_input(f"{feat} Lower", value=min_hist)
            with col2: upper = st.number_input(f"{feat} Upper", value=max_hist)
            bounds.append((lower, upper))
            
        if st.button("Run Optimizer"):
            with st.spinner("Finding optimal regime..."):
                def objective(x):
                    pred = st.session_state.ml_model.predict(pd.DataFrame([x], columns=st.session_state.features))[0]
                    return -pred if opt_goal == "Maximize Target" else pred
                
                x0 = [(b[0] + b[1]) / 2 for b in bounds]
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
                
                if res.success:
                    st.success("Optimization Converged!")
                    opt_pred = -res.fun if opt_goal == "Maximize Target" else res.fun
                    st.metric(label=f"Optimized {st.session_state.target}", value=f"{opt_pred:.4f}")
                    st.subheader("Recommended Setpoints")
                    st.json({feat: round(val, 2) for feat, val in zip(st.session_state.features, res.x)})
                else:
                    st.error("Optimization failed to converge. Relax constraints.")
    else:
        st.info("Train the Digital Twin model in Tab 4 first.")
