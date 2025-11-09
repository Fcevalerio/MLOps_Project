import os
from io import BytesIO
import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from datetime import datetime
from utils.app_helpers import list_weather_blobs
from azure.storage.blob import BlobServiceClient

# ─────────────────────────────────────────────
# 0. Auxiliary Functions
# ─────────────────────────────────────────────
def highlight_temp(val):
    color = 'red' if val > 30 else 'blue' if val < 10 else ''
    return f'background-color: {color}'

# ─────────────────────────────────────────────
# 1. Website Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Forecast Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
DAILY_DATA_PATH = "daily-weather-data"
HISTORICAL_DATA_PATH = "weather-data"

blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

# ─────────────────────────────────────────────
# 2. Website Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main {
        background-color: #f7faff;
    }
    h1, h2, h3, h4 {
        color: #003366;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=80)
with col2:
    st.title("🌤️ Weather Forecast & Model Dashboard")
    st.write("Visualizing 14-day forecasts for Madrid, ML model performance, and weather insights powered by Azure + MLflow.")

st.markdown("---")

# ─────────────────────────────────────────────
# 3. Laod Data & Model
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    # Connect to Azure Blob and get latest daily data
    daily_data = blob_service_client.get_container_client(DAILY_DATA_PATH)
    # List all weather blobs sorted by date in filename
    last_daily_blob = list_weather_blobs(daily_data)[-1]
    print(f"Downloading latest daily data blob: {last_daily_blob}")
    # Connection to blob and download
    daily_client = daily_data.get_blob_client(last_daily_blob)
    daily_df = pd.read_csv(BytesIO(daily_client.download_blob().readall()))
    return daily_df

@st.cache_data
def load_historical_data():
    # Connect to Azure Blob and get latest daily data
    historical_data = blob_service_client.get_container_client(HISTORICAL_DATA_PATH)
    # List all weather blobs sorted by date in filename
    last_historical_blob = list_weather_blobs(historical_data)[-1]
    print(f"Downloading latest daily data blob: {last_historical_blob}")
    # Connection to blob and download
    historical_client = historical_data.get_blob_client(last_historical_blob)
    historical_df = pd.read_csv(BytesIO(historical_client.download_blob().readall()))
    return historical_df

# Load data
data = load_data()
historical = load_historical_data()

# Clean column names
data.drop(columns=["temperature_2m_mean"], inplace=True)
data.columns = [col.replace("_", " ").title() for col in data.columns]
historical.columns = [col.replace("_", " ").title() for col in historical.columns]

# Reordering columns
col_order = ['Date', 'Weather Prediction'] + [col for col in data.columns if col not in ('Date','Weather Prediction')]
data = data[col_order]

# ─────────────────────────────────────────────
# Section 1: 14-Day Temperature Forecast
# ─────────────────────────────────────────────
st.header("14-Day Temperature Forecast using ML Model!")

col1, col2, col3 = st.columns(3)
col1.metric("Average Temp (°C)", f"{data['Weather Prediction'].mean():.1f}")
col2.metric("Max Temp (°C)", f"{data['Weather Prediction'].max():.1f}")
col3.metric("Min Temp (°C)", f"{data['Weather Prediction'].min():.1f}")

fig_forecast = px.line(
    data,
    x="Date",
    y="Weather Prediction",
    title="Predicted Temperature Trend (Next 14 Days)",
    markers=True,
    line_shape="spline"
)
fig_forecast.update_layout(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(color="#003366"),
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(
        showline=True,
        linecolor='black',
        tickfont=dict(color='black'),
        title_font=dict(color='black') ),
    yaxis=dict(
        showline=True,
        linecolor='black',
        tickfont=dict(color='black'),
        title_font=dict(color='black') )
)

st.plotly_chart(fig_forecast, width='stretch')

# ─────────────────────────────────────────────
# Section 2: Forecast Data Table
# ─────────────────────────────────────────────
st.header("Forecast Data Table")

# Formating date column
data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

# Display table in Streamlit with highlighted temps
st.dataframe(
    data.style.applymap(highlight_temp, subset=["Weather Prediction"]),
    use_container_width=True,
    height=400
)

# Show summary stats below the table
st.subheader("Summary Statistics")
st.table(data.describe().round(2))

# ─────────────────────────────────────────────
# Section 3: MLflow Experiment Tracking
# ─────────────────────────────────────────────
st.header("MLflow Experiment Tracking")

try:
    # Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiments = client.search_experiments()

    if not experiments:
        st.warning("No experiments found in MLflow.")
        st.stop()

    # Let user pick an experiment
    exp_names = [exp.name for exp in experiments]
    chosen_exp = st.selectbox("Select Experiment:", exp_names)

    # Retrieve selected experiment details
    exp = next(e for e in experiments if e.name == chosen_exp)
    runs = client.search_runs(experiment_ids=[exp.experiment_id])

    if not runs:
        st.warning(f"No runs found for experiment '{chosen_exp}'.")
        st.stop()

    # Convert MLflow Run objects into a DataFrame
    runs_df = pd.DataFrame([
        {
            "run_id": r.info.run_id,
            "start_time": datetime.fromtimestamp(r.info.start_time / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                          if r.info.start_time else None,
            **{f"param_{k}": v for k, v in r.data.params.items()},
            **{f"metric_{k}": v for k, v in r.data.metrics.items()},
        }
        for r in runs
    ])

    # Display in Streamlit
    st.subheader(f"Experiment: {chosen_exp}")

    # Show only a few key columns if they exist
    cols_to_show = [col for col in ["run_id", "metric_rmse", "metric_mae", "param_C", "start_time"] if col in runs_df.columns]
    st.dataframe(runs_df[cols_to_show] if cols_to_show else runs_df)

    st.success("Connected to MLflow successfully!")

except Exception as e:
    st.warning("Could not connect to MLflow.")
    st.exception(e)

# ─────────────────────────────────────────────
# Section 4: Historical Data Trends
# ─────────────────────────────────────────────
st.header("Historical Data Trends")

fig_hist = px.histogram(
    historical,
    x="Temperature 2M Mean",
    nbins=10,
    title="Temperature Distribution",
    color_discrete_sequence=["#6699cc"]
)
st.plotly_chart(fig_hist, width='stretch')

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <center>
    <p style='color:gray'>Powered by Azure ML, MLflow, Streamlit and Group 5 Friendship</p>
    <img src='https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg' width='120'>
    </center>
    """,
    unsafe_allow_html=True
)
