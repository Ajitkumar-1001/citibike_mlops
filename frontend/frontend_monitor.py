import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions, load_metrics_from_registry
from src.experiment_utils import set_mlflow_tracking

# Set page configuration
st.set_page_config(page_title="Citi Bike Model Monitoring", layout="wide")

# Title and description
st.title("Citi Bike Model Monitoring App")
st.markdown("This app monitors the performance of the Citi Bike demand prediction model.")

# Sidebar for user input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=12,
    step=1,
)

# Section 1: MAE Over Time (from Hopsworks data)
st.header("Mean Absolute Error (MAE) by Pickup Hour")

# Fetch data
st.write("Fetching data for the past", past_hours, "hours...")
try:
    df1 = fetch_hourly_rides(past_hours)
    df2 = fetch_predictions(past_hours)

    # Merge the DataFrames on 'pickup_location_id' and 'pickup_hour'
    merged_df = pd.merge(df1, df2, on=["pickup_location_id", "pickup_hour"])

    # Calculate the absolute error
    merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

    # Group by 'pickup_hour' and calculate the mean absolute error (MAE)
    mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
    mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

    # Create a Plotly plot for MAE over time
    fig = px.line(
        mae_by_hour,
        x="pickup_hour",
        y="MAE",
        title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
        labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
        markers=True,
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    st.write(f'Average MAE: {mae_by_hour["MAE"].mean():.2f}')
except Exception as e:
    st.error(f"Error fetching data or calculating MAE: {str(e)}")

# Section 2: MLflow Metrics
st.header("Model Metrics from MLflow")

# Set up MLflow tracking
try:
    mlflow = set_mlflow_tracking()
    st.write("Connected to MLflow tracking server.")
except Exception as e:
    st.error(f"Error connecting to MLflow: {str(e)}")
    mlflow = None

# Fetch metrics from the model registry
if mlflow:
    try:
        metrics = load_metrics_from_registry()
        st.subheader("Model Training Metrics")

        # Display metrics in a table
        if metrics:
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
            st.write(metrics_df)

            # If MAE and baseline improvement metrics are available, visualize them
            if "MAE" in metrics:
                st.metric("Training MAE", f"{metrics['MAE']:.2f}")
            if "baseline_improvement" in metrics:
                st.metric("Improvement Over Baseline", f"{metrics['baseline_improvement']:.2f}%")
        else:
            st.info("No metrics found in the model registry.")
    except Exception as e:
        st.error(f"Error fetching MLflow metrics: {str(e)}")