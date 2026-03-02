import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
from scipy.optimize import minimize
import google.generativeai as genai

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
for key in ['stitched_data', 'filtered_data', 'ml_model', 'features', 'target', 'chemistry_context', 'data_insights']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'features' else []

# --- HELPER FUNCTIONS ---
@st.cache_data
def process_quality_sheet(df, suffix):
    """Combines Date and Time, applies dayfirst, adds suffixes, and rounds to 30min."""
    cols = [c.lower() for c in df.columns]
    date_col = df.columns[cols.index('date')] if 'date' in cols else None
    time_col = df.columns[cols.index('time')] if 'time' in cols else None
    
    if date_col and time_col:
        # Create Timestamp with dayfirst=True
        df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Timestamp']).drop(columns=[date_col, time_col])
        
        # Round to nearest 30 mins to align with process data
        df['Timestamp'] = df['Timestamp'].dt.round('30min')
        
        # Group by timestamp in case of duplicates in the same 30 min window
        df = df.groupby('Timestamp').mean(numeric_only=True).reset_index()
        
        # Add suffixes to composition columns
        df = df.rename(columns={c: f"{c}_{suffix}" for c in df.columns if c != 'Timestamp'})
    return df

def generate_ai_insights(df):
    """Uses LLM to find 5 key data patterns (trends, outliers, stats)."""
    if not api_key: return "API Key required for insights."
    
    # Create a lightweight summary to send to the LLM
    stats = df.describe().to_string()
    # Calculate monthly averages to spot trends
    monthly_trends = df.resample('ME', on='Timestamp').mean(numeric_only=True).to_string()
    
    prompt = f"""
    You are an expert Data Scientist in a chemical plant. Analyze this flat-file dataset summary.
    
    Summary Statistics:
    {stats}
    
    Monthly Averages (to identify trends over time):
    {monthly_trends}
    
    Identify exactly 5 key observations/patterns from this data. Look for:
    - Long-term trends (e.g., "Parameter X is in an increasing trend from Month Y").
    - Operational ranges (e.g., "Feed Flow mostly operates at X but maxes at Y").
    - Quality variations (e.g., "Top quality median is X with a minimum of Y").
    Format as a clean, bulleted list.
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# --- MAIN APP TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Upload & Engineering", 
    "📊 EDA & Filtering", 
    "🧠 AI Process Context", 
    "⚙️ Simulation", 
    "🚀 Optimization"
])

# ==========================================
# TAB 1: DATA UPLOAD & ENGINEERING
# ==========================================
with tab1:
    st.header("1. Upload Process & Quality Data")
    col1, col2 = st.columns(2)
    
    with col1:
        process_file = st.file_uploader("Upload Process Data (CSV/XLSX)", type=['csv', 'xlsx', 'xls'])
    with col2:
        quality_file = st.file_uploader("Upload Quality Lab Data (Excel)", type=['xlsx', 'xls'])

    if process_file and quality_file:
        try:
            with st.spinner("Processing & Stitching Data..."):
                # 1. Load & Resample Process Data
                if process_file.name.endswith('.csv'):
                    df_process = pd.read_csv(process_file)
                else:
                    df_process = pd.read_excel(process_file)
                    
                ts_col = [c for c in df_process.columns if 'time' in c.lower() or 'date' in c.lower()][0]
                df_process['Timestamp'] = pd.to_datetime(df_process[ts_col], dayfirst=True, errors='coerce')
                df_process = df_process.dropna(subset=['Timestamp'])
                
                # Resample Process Data to 30 Min Average
                df_process = df_process.set_index('Timestamp').resample('30min').mean(numeric_only=True).reset_index()
                
                # 2. Load Quality Data
                xls = pd.ExcelFile(quality_file)
                sheet_names = [s.lower() for s in xls.sheet_names]
                
                feed_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('feed')]) if 'feed' in sheet_names else pd.DataFrame(), 'feed')
                top_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('top')]) if 'top' in sheet_names else pd.DataFrame(), 'top')
                bot_df = process_quality_sheet(pd.read_excel(xls, xls.sheet_names[sheet_names.index('bottom')]) if 'bottom' in sheet_names else pd.DataFrame(), 'bot')
                
                # 3. Stitch Data (Merge on exact 30-min rounded Timestamp)
                df_stitched = df_process.copy()
                if not feed_df.empty: df_stitched = pd.merge(df_stitched, feed_df, on='Timestamp', how='left')
                if not top_df.empty: df_stitched = pd.merge(df_stitched, top_df, on='Timestamp', how='left')
                if not bot_df.empty: df_stitched = pd.merge(df_stitched, bot_df, on='Timestamp', how='left')
                
                # 4. Interpolate Quality Data (Linear, forward limit = 4, approx 2 hours)
                qual_cols = [c for c in df_stitched.columns if c.endswith('_feed') or c.endswith('_top') or c.endswith('_bot')]
                df_stitched[qual_cols] = df_stitched[qual_cols].interpolate(method='linear', limit=4, limit_direction='forward')
                
                # Drop rows where we still have no quality data (optional, but good for ML)
                df_stitched = df_stitched.dropna(subset=qual_cols, how='all')

            st.success("✅ Data Stitched and Interpolated Successfully!")
            
            # --- Feature Engineering UI ---
            st.divider()
            st.subheader("2. Engineer Physics Features")
            st.write("Map your columns to generate advanced thermodynamic features.")
            
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
                st.session_state.filtered_data = df_stitched # Init filtered data
                st.success("Features Generated! Go to Tab 2.")
                
                # Generate AI Insights in background
                with st.spinner("AI is analyzing data patterns..."):
                    st.session_state.data_insights = generate_ai_insights(df_stitched)
                    
            st.dataframe(df_stitched.head())

        except Exception as e:
            st.error(f"Error processing files: {e}")

# ==========================================
# TAB 2: EDA & FILTERING (Outlier Removal)
# ==========================================
with tab2:
    if st.session_state.stitched_data is not None:
        df = st.session_state.stitched_data
        
        st.subheader("🧹 Outlier Filtering")
        st.write("Select parameters to filter their ranges and remove outliers.")
        filter_cols = st.multiselect("Select columns to filter:", df.columns.drop('Timestamp'))
        
        # Dynamic filtering dictionary
        filters = {}
        if filter_cols:
            f_cols = st.columns(min(len(filter_cols), 3))
            for i, col in enumerate(filter_cols):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                with f_cols[i % 3]:
                    filters[col] = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
        
        # Apply filters
        df_filtered = df.copy()
        for col, (f_min, f_max) in filters.items():
            df_filtered = df_filtered[(df_filtered[col] >= f_min) & (df_filtered[col] <= f_max)]
        
        st.session_state.filtered_data = df_filtered
        st.write(f"Data points remaining: **{len(df_filtered)}** out of **{len(df)}**")
        
        st.divider()
        st.subheader("📈 Exploratory Data Analysis")
        all_cols = df_filtered.columns.drop('Timestamp').tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            trend_cols = st.multiselect("Time Series Trends", all_cols, default=all_cols[:1])
            if trend_cols:
                st.plotly_chart(px.line(df_filtered, x='Timestamp', y=trend_cols), use_container_width=True)
        with c2:
            scat_x = st.selectbox("Scatter X-axis", all_cols, index=0)
            scat_y = st.selectbox("Scatter Y-axis", all_cols, index=1 if len(all_cols)>1 else 0)
            st.plotly_chart(px.scatter(df_filtered, x=scat_x, y=scat_y, trendline="ols"), use_container_width=True)

    else:
        st.info("Upload and process data in Tab 1 first.")

# ==========================================
# TAB 3: AI PROCESS CONTEXT
# ==========================================
with tab3:
    st.header("🧠 AI Process & Data Intelligence")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Key Data Observations")
        if st.session_state.data_insights:
            st.markdown(st.session_state.data_insights)
        else:
            st.info("Data insights will appear here after Feature Engineering.")
            
    with col2:
        st.subheader("🧪 Chemistry Context")
        # Ensure context generation handles the suffixes properly
        # (Assuming you add the chemistry prompt logic here similarly to the previous version)
        st.markdown("*LLM Chemistry context generates based on GC column names...*")

# ==========================================
# TAB 4 & 5: SIMULATION & OPTIMIZATION
# ==========================================
# (These tabs remain exactly the same as the previous version, 
# just ensure they use `st.session_state.filtered_data` instead of `stitched_data` for model training)
with tab4:
    if st.session_state.filtered_data is not None:
        st.header("⚙️ Train Digital Twin & Simulate")
        df_train = st.session_state.filtered_data
        # ... [Rest of Tab 4 training logic using df_train] ...
        st.info("Simulation engine ready. (Insert Tab 4 logic here from previous version)")

with tab5:
    st.header("🚀 AI-Driven Process Optimization")
    st.info("Optimization engine ready. (Insert Tab 5 logic here from previous version)")
