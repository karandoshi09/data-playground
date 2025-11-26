import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Data Playground", layout="wide", page_icon="📊")

# Title and Introduction
st.title("📊 The Data Playground")
st.markdown("""
Welcome to your personal data analysis tool. Upload a dataset to get started.
You can clean data, calculate statistics, and create interactive visualizations.
""")

# --- Sidebar for File Upload ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Caching the data loading function for performance
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Main App Logic
if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)

    if df is not None:
        # Create Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["👀 Data Overview", "🧹 Data Cleaning", "🧮 Statistics", "📈 Visualization"])

        # --- TAB 1: Data Overview ---
        with tab1:
            st.header("Data Overview")
            st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
            
            # Show first few rows
            st.subheader("First 5 Rows")
            st.dataframe(df.head())

            # Column Data Types
            with st.expander("View Column Data Types"):
                st.json(df.dtypes.astype(str).to_dict())

            # Basic Summary
            with st.expander("View Statistical Summary"):
                st.dataframe(df.describe())

        # --- TAB 2: Data Cleaning & Filtering ---
        with tab2:
            st.header("Data Cleaning")
            
            # Make a copy for cleaning to avoid modifying original session state incorrectly
            df_clean = df.copy()

            col1, col2 = st.columns(2)

            # 1. Drop Columns
            with col1:
                st.subheader("Remove Columns")
                columns_to_drop = st.multiselect("Select columns to drop", df_clean.columns)
                if columns_to_drop:
                    df_clean = df_clean.drop(columns=columns_to_drop)
                    st.success(f"Dropped: {', '.join(columns_to_drop)}")

            # 2. Handle Missing Values
            with col2:
                st.subheader("Handle Missing Values")
                missing_action = st.radio("Choose action:", ["None", "Drop Rows with Missing Values", "Fill with 0"])
                
                if missing_action == "Drop Rows with Missing Values":
                    df_clean = df_clean.dropna()
                    st.info("Rows with missing values dropped.")
                elif missing_action == "Fill with 0":
                    df_clean = df_clean.fillna(0)
                    st.info("Missing values filled with 0.")

            # 3. Filter Data
            st.subheader("Filter Data")
            filter_col = st.selectbox("Select column to filter by", ["None"] + list(df_clean.columns))
            
            if filter_col != "None":
                # Check if column is numeric or categorical
                if pd.api.types.is_numeric_dtype(df_clean[filter_col]):
                    min_val = float(df_clean[filter_col].min())
                    max_val = float(df_clean[filter_col].max())
                    rng = st.slider(f"Select range for {filter_col}", min_val, max_val, (min_val, max_val))
                    df_clean = df_clean[(df_clean[filter_col] >= rng[0]) & (df_clean[filter_col] <= rng[1])]
                else:
                    unique_vals = df_clean[filter_col].unique()
                    selected_vals = st.multiselect(f"Select values for {filter_col}", unique_vals, default=unique_vals)
                    df_clean = df_clean[df_clean[filter_col].isin(selected_vals)]

            st.write("### Resulting Dataframe")
            st.dataframe(df_clean)

        # --- TAB 3: Statistics ---
        with tab3:
            st.header("Statistical Calculations")
            
            # Filter only numeric columns
            numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if numeric_cols:
                selected_col_stat = st.selectbox("Select a numeric column", numeric_cols)
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.metric("Mean", f"{df_clean[selected_col_stat].mean():.2f}")
                with c2:
                    st.metric("Median", f"{df_clean[selected_col_stat].median():.2f}")
                with c3:
                    st.metric("Std Dev", f"{df_clean[selected_col_stat].std():.2f}")
                
                st.subheader("Custom Percentile")
                percentile = st.slider("Select Percentile", 0, 100, 50)
                p_value = df_clean[selected_col_stat].quantile(percentile / 100)
                st.write(f"The **{percentile}th percentile** of **{selected_col_stat}** is: **{p_value:.2f}**")
            else:
                st.warning("No numeric columns found in the dataset.")

        # --- TAB 4: Visualization ---
        with tab4:
            st.header("Visualize Data")
            
            chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot"])
            
            # General Axis Selection
            all_cols = df_clean.columns
            
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox("X Axis", all_cols)
            with c2:
                # Y Axis is not needed for Histograms sometimes, but usually good to have
                y_axis = st.selectbox("Y Axis", all_cols, index=1 if len(all_cols) > 1 else 0)
            with c3:
                color_encode = st.selectbox("Color (Group By)", ["None"] + list(all_cols))

            # Logic to plot
            if color_encode == "None":
                color = None
            else:
                color = color_encode

            try:
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df_clean, x=x_axis, y=y_axis, color=color, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Line Chart":
                    fig = px.line(df_clean, x=x_axis, y=y_axis, color=color, title=f"{y_axis} over {x_axis}")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df_clean, x=x_axis, y=y_axis, color=color, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Histogram":
                    fig = px.histogram(df_clean, x=x_axis, color=color, title=f"Distribution of {x_axis}")
                elif chart_type == "Box Plot":
                    fig = px.box(df_clean, x=x_axis, y=y_axis, color=color, title=f"Box Plot of {y_axis} by {x_axis}")

                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Could not create chart: {e}. Please ensure you selected appropriate columns (e.g., Numeric vs Categorical).")

else:
    st.info("Please upload a file from the sidebar to begin.")