# Crop Recommendation and Yield Prediction and Selling Prediction Streamlit Application

import pandas as pd
import plotly.express as px
import streamlit as st

def render_all_states_map_page(sell_csv_path="Crop_Selling_Final.csv"):
    st.markdown("## 🗺️ LIVE REAL APMC MARKET MAP")
    st.caption("Home / Market Price Trends / Full Real Map")

    # Load dataset
    try:
        sell_df = pd.read_csv(sell_csv_path)
        sell_df["Price_per_Quintal"] = sell_df["Price per kg (₹)"] * 100
    except Exception as e:
        st.error(f"Error loading {sell_csv_path}: {e}")
        if st.button("← Back to Market Trends"):
            st.session_state["page_nav"] = "Market Trends"
            st.rerun()
        return

    # Filter Controls
    col1, col2 = st.columns(2)
    with col1:
        all_crops = ["All Crops"] + sorted(sell_df["Crop"].unique().tolist())
        selected_crop = st.selectbox("Select Crop", all_crops, key="map_page_crop")
    with col2:
        all_seasons = ["All Seasons"] + sorted(sell_df["Season"].unique().tolist())
        selected_season = st.selectbox("Select Season", all_seasons, key="map_page_season")

    filtered_df = sell_df.copy()
    if selected_crop != "All Crops":
        filtered_df = filtered_df[filtered_df["Crop"] == selected_crop]
    if selected_season != "All Seasons":
        filtered_df = filtered_df[filtered_df["Season"] == selected_season]

    if filtered_df.empty:
        st.warning("No records found.")
        return

    # Aggregate
    state_agg = (
        filtered_df.groupby("State")
        .agg(
            Avg_Price=("Price_per_Quintal", "mean"),
            Min_Price=("Price_per_Quintal", "min"),
            Max_Price=("Price_per_Quintal", "max"),
            Total_Kg=("Quantity (kg)", "sum"),
            Total_Value=("Total Value (₹)", "sum"),
            Total_Records=("Crop", "count"),
        )
        .reset_index()
    )

    # Geographic coordinates for dataset states
    geo_locations = {
        "Gujarat": {"lat": 22.2587, "lon": 71.1924, "APMC": "Ahmedabad / Rajkot APMC"},
        "Karnataka": {"lat": 15.3173, "lon": 75.7139, "APMC": "Bengaluru APMC"},
        "Maharashtra": {"lat": 19.7515, "lon": 75.7139, "APMC": "Mumbai APMC"},
        "Punjab": {"lat": 31.1471, "lon": 75.3412, "APMC": "Ludhiana APMC"},
        "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569, "APMC": "Chennai APMC"},
    }

    state_agg["lat"] = state_agg["State"].map(lambda x: geo_locations.get(x, {}).get("lat", 20.5937))
    state_agg["lon"] = state_agg["State"].map(lambda x: geo_locations.get(x, {}).get("lon", 78.9629))
    state_agg["APMC_Hub"] = state_agg["State"].map(lambda x: geo_locations.get(x, {}).get("APMC", "State Hub"))

    state_agg["Avg Price (₹/Quintal)"] = state_agg["Avg_Price"].map(lambda x: f"₹ {x:,.2f}")
    state_agg["Total Volume (Tons)"] = (state_agg["Total_Kg"] / 1000).round(2)

    # Map visualization
    st.markdown("### 📍 Real OpenStreetMap Geographic View")
    fig = px.scatter_mapbox(
        state_agg,
        lat="lat", lon="lon",
        size="Total Volume (Tons)",
        color="Avg_Price",
        hover_name="State",
        hover_data={
            "APMC_Hub": True,
            "Avg Price (₹/Quintal)": True,
            "Total Volume (Tons)": True,
            "Total_Records": True,
            "lat": False, "lon": False,
        },
        color_continuous_scale="Plasma",
        zoom=4.3,
        center={"lat": 22, "lon": 77},
        mapbox_style="open-street-map",
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Aggregation table
    st.markdown("### 📊 Market Aggregation Details")
    st.dataframe(
        state_agg[["State","APMC_Hub","Avg Price (₹/Quintal)","Total Volume (Tons)","Total_Records"]],
        use_container_width=True,
        hide_index=True,
    )

    # Back navigation
    if st.button("← Back to Market Trends"):
        st.session_state["page_nav"] = "Market Trends"
        st.rerun()
