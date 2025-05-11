import sys
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions, fetch_days_data

# Set page configuration
st.set_page_config(page_title="Citi Bike Model Monitoring", layout="wide")

st.title("Mean Absolute Error (MAE) by Pickup Hour")

# Sidebar for user input
st.sidebar.header("Settings")
past_days = st.sidebar.slider(
    "Number of Past Days to Plot",
    min_value=1,
    max_value=28,
    value=7,
    step=1,
)

# Convert days to hours for fetch_predictions
past_hours = past_days * 24

# Fetch data
st.write("Fetching data for the past", past_days, "days (", past_hours, "hours)...")
try:
    # Fetch actual rides using fetch_days_data (simulated data from 52 weeks ago)
    df1 = fetch_days_data(past_days)
    df2 = fetch_predictions(past_hours)

    # Debug: Inspect the DataFrames
    st.write("Actual Rides Data (df1):", df1.head())
    st.write("Predictions Data (df2):", df2.head())

    # Ensure pickup_hour is in the same timezone
    df1["pickup_hour"] = df1["pickup_hour"].dt.tz_convert("America/New_York")
    df2["pickup_hour"] = df2["pickup_hour"].dt.tz_convert("America/New_York")

    # Merge the DataFrames on 'pickup_location_id' and 'pickup_hour'
    merged_df = pd.merge(df1, df2, on=["pickup_location_id", "pickup_hour"], how="inner")

    # Debug: Inspect the merged DataFrame
    st.write("Merged DataFrame:", merged_df.head())

    if merged_df.empty:
        st.warning("No matching data found between actual rides and predictions for the selected time range.")
    else:
        # Calculate the absolute error
        merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

        # Group by 'pickup_hour' and calculate the mean absolute error (MAE)
        mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
        mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

        # Debug: Inspect MAE DataFrame
        st.write("MAE by Hour:", mae_by_hour.head())

        if mae_by_hour.empty:
            st.warning("No MAE data to plot after grouping.")
        else:
            # Create a Plotly plot
            fig = px.line(
                mae_by_hour,
                x="pickup_hour",
                y="MAE",
                title=f"Mean Absolute Error (MAE) for the Past {past_days} Days",
                labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
                markers=True,
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            avg_mae = mae_by_hour["MAE"].mean()
            st.write(f'Average MAE: {avg_mae:.2f}' if not pd.isna(avg_mae) else "Average MAE: nan")
except Exception as e:
    st.error(f"Error fetching data or calculating MAE: {str(e)}")