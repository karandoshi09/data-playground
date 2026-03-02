# 🧪 AI-Powered Distillation Column Digital Twin

A comprehensive Streamlit application that acts as a Digital Twin for a chemical distillation column. It combines Machine Learning (XGBoost) for predictive simulation and Large Language Models (Gemini) for thermodynamic validation and process chemistry insights.

## Features
1. **Intelligent Data Stitching:** Merges high-frequency DCS/PLC process data with irregular lab quality data (Feed, Top, Bottom GC analysis).
2. **AI Chemistry Profiling:** Uses Gemini to infer the process chemistry and thermodynamic properties based on uploaded GC compositions.
3. **Exploratory Data Analysis:** Interactive Plotly dashboards for trends, distributions, scatterplots, and correlation heatmaps.
4. **Predictive Simulation:** Train a dynamic XGBoost model to simulate how changing temperatures, pressures, or flows impacts product quality.
5. **AI-Guided Optimization:** Uses Scipy optimization to find the best operating setpoints, cross-validated by the LLM for safety (e.g., flooding/weeping limits).

## Deployment on Streamlit Community Cloud
1. Fork or clone this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Deploy a new app, selecting this repository and `app.py` as the main file.
4. **Important:** Add your Google Gemini API Key in the Streamlit Cloud Secrets setting as `GEMINI_API_KEY`.

## Usage
1. Upload your Process Data (CSV) and Quality Data (Excel with `Feed`, `Top`, and `Bottom` sheets).
2. Enter your Gemini API key in the sidebar (if running locally).
3. Navigate through the tabs to analyze, simulate, and optimize your process.
