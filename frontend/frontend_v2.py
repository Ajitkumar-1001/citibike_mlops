import sys
from pathlib import Path

# Add the parent directory to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
# from src.config import DATA_DIR

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import plotly.express as px

from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store, get_model_predictions, load_model_from_registry
from src.plot_utils import plot_prediction

# Set page configuration
st.set_page_config(page_title="Citi Bike Demand Prediction", layout="wide")

# Title and description
st.title("Citi Bike Demand Prediction App")
st.markdown("This app predicts the hourly demand for Citi Bike at various locations in NYC.")

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")

# Fetch available locations
with st.spinner("Fetching available data..."):
    # Load a batch of features to get available locations (using a recent date)
    current_date = datetime.now(ZoneInfo("America/New_York"))
    features_df = load_batch_of_features_from_store(current_date)
    available_locations = sorted(features_df["pickup_location_id"].unique())

# User selects a pickup location
selected_location = st.sidebar.selectbox(
    "Select Pickup Location ID",
    options=available_locations,
    help="Choose a location to see demand predictions."
)

# User selects the prediction time (default to next hour)
next_hour = (current_date + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
prediction_time = st.sidebar.date_input(
    "Prediction Date",
    value=next_hour.date(),
    min_value=next_hour.date(),
    max_value=(next_hour + timedelta(days=7)).date()
)
prediction_hour = st.sidebar.slider(
    "Prediction Hour",
    min_value=0,
    max_value=23,
    value=next_hour.hour
)
selected_time = datetime.combine(prediction_time, datetime.min.time()).replace(
    hour=prediction_hour, tzinfo=ZoneInfo("America/New_York")
)

# Load the model
with st.spinner("Loading model..."):
    model = load_model_from_registry()

# Main content
st.header(f"Demand Prediction for Location {selected_location}")

# Fetch historical features for the selected location
historical_features = features_df[features_df["pickup_location_id"] == selected_location].tail(1)

if historical_features.empty:
    st.error(f"No historical data available for Location {selected_location} at {selected_time}.")
else:
    # Get predictions
    with st.spinner("Generating predictions..."):
        predictions = get_model_predictions(model, historical_features)

    # Display the prediction
    st.subheader("Predicted Demand")
    predicted_demand = int(predictions["predicted_demand"].iloc[0])
    st.metric("Predicted Rides", predicted_demand)

    # Plot historical data and prediction
    st.subheader("Historical Data and Prediction")
    fig = plot_prediction(historical_features, predictions)
    st.plotly_chart(fig, use_container_width=True)

# Fetch and display stored predictions (if available)
st.subheader("Stored Predictions for Next Hour")
try:
    stored_predictions = fetch_next_hour_predictions()
    if not stored_predictions.empty:
        stored_for_location = stored_predictions[stored_predictions["pickup_location_id"] == selected_location]
        if not stored_for_location.empty:
            st.write(stored_for_location[["pickup_hour", "predicted_demand"]])
        else:
            st.info(f"No stored predictions for Location {selected_location} for the next hour.")
    else:
        st.info("No stored predictions available for the next hour.")
except Exception as e:
    st.error(f"Error fetching stored predictions: {str(e)}")