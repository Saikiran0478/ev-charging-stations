import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from transformers import pipeline # Example for a text-based model
import joblib # For loading scikit-learn models, if applicable

# --- 1. Load your AI Model (Placeholder) ---
# This is where you'd integrate your actual Hugging Face model.
# Option A: Using a pre-trained Hugging Face model (e.g., for sentiment analysis on reviews, or a custom text-based recommender)
# Try to find a model on Hugging Face that *might* be relevant, even if it's not a perfect "EV charge locator."
# For a real EV charge locator, you'd likely need to train your own model or use a specialized API.
# For demonstration, let's pretend we have a simple "ranking" model.
try:
    # If you have a custom model hosted on Hugging Face or locally:
    # Example: a model that takes location context and returns a score for a hypothetical charging station.
    # This is highly speculative and would need to be replaced with a real model.
    # For a real application, you'd train a model that takes features like location,
    # time of day, desired amenities, and outputs a recommendation score or probability of availability.

    # Example of a *dummy* Hugging Face model for demonstration:
    # Let's imagine a model that predicts "user happiness score" for a station based on text input.
    # This is NOT an EV locator, but demonstrates the integration.
    # For a real EV locator, you'd need a dataset of stations and train a model on features.
    # As a placeholder, let's use a sentiment analysis model to "score" a fictional description.
    # Replace 'distilbert-base-uncased-finetuned-sst-2-english' with your actual model if you have one.
    st.cache_resource
    def load_model():
        # This is a placeholder. You'll need to train/find a suitable model.
        # For a true EV locator, you'd likely use a model that takes location data and other factors.
        # This sentiment pipeline is just to show the *Hugging Face integration* concept.
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_analyzer = load_model()

    # If you have a scikit-learn model saved as a .pkl file (e.g., for predicting optimal locations)
    # try:
    #     ev_locator_model = joblib.load("path/to/your/ev_locator_model.pkl")
    # except FileNotFoundError:
    #     st.error("EV locator model not found. Please ensure 'ev_locator_model.pkl' is in the correct directory.")
    #     ev_locator_model = None # Handle case where model isn't loaded

except Exception as e:
    st.error(f"Error loading Hugging Face model: {e}")
    st.info("Ensure you have `transformers` installed (`pip install transformers`) and check model name.")
    sentiment_analyzer = None

# --- 2. Sample Data for EV Charging Stations (Realistic Placeholder) ---
# In a real app, this would come from a database or API.
# For demonstration, we'll create some mock data.
ev_stations_data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Green Charge Hub', 'EcoVolt Station', 'PowerUp Plaza', 'E-Charge Point', 'Quick Spark'],
    'latitude': [17.4399, 17.4450, 17.4300, 17.4520, 17.4250], # Hyderabad coordinates for example
    'longitude': [78.4983, 78.5050, 78.4890, 78.4900, 78.5100],
    'charger_types': [['CCS', 'Type 2'], ['CHAdeMO', 'Type 2'], ['CCS'], ['Type 2'], ['CCS', 'CHAdeMO']],
    'charging_speed_kw': [50, 150, 22, 11, 100],
    'availability': ['Available', 'Busy', 'Available', 'Available', 'Busy'],
    'rating': [4.5, 3.8, 4.2, 3.5, 4.0],
    'description': [
        "Well-maintained station with fast charging. Good for a quick top-up.",
        "Often busy, but offers high-speed charging. Near a cafe.",
        "Standard charging, usually available. Residential area.",
        "Slower charging, good for overnight. Close to a park.",
        "Very fast charging, but frequently occupied. Shopping complex nearby."
    ]
})

# --- 3. Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="EV Charge Locator")

st.title("⚡ EV Charge Locator with AI Recommendations")

st.write("Find the best EV charging stations near you with smart recommendations!")

# User Input
st.sidebar.header("Your Preferences")
user_latitude = st.sidebar.number_input("Your Latitude", value=17.4367, format="%.4f")
user_longitude = st.sidebar.number_input("Your Longitude", value=78.4988, format="%.4f")

preferred_charger_type = st.sidebar.multiselect(
    "Preferred Charger Type",
    options=['CCS', 'Type 2', 'CHAdeMO'],
    default=['CCS', 'Type 2']
)

min_charging_speed = st.sidebar.slider(
    "Minimum Charging Speed (kW)",
    min_value=0, max_value=150, value=20, step=10
)

show_available_only = st.sidebar.checkbox("Show Available Stations Only", value=True)

# --- 4. Filter and Process Data ---
filtered_stations = ev_stations_data.copy()

# Apply availability filter
if show_available_only:
    filtered_stations = filtered_stations[filtered_stations['availability'] == 'Available']

# Apply charger type filter
if preferred_charger_type:
    filtered_stations = filtered_stations[
        filtered_stations['charger_types'].apply(lambda types: any(ct in preferred_charger_type for ct in types))
    ]

# Apply charging speed filter
filtered_stations = filtered_stations[filtered_stations['charging_speed_kw'] >= min_charging_speed]

# Calculate distance from user to each station
# Using Haversine formula for distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

filtered_stations['distance_km'] = filtered_stations.apply(
    lambda row: haversine(user_latitude, user_longitude, row['latitude'], row['longitude']), axis=1
)

# Sort by distance
filtered_stations = filtered_stations.sort_values(by='distance_km').reset_index(drop=True)

# --- 5. AI Model Integration (Demonstration with Sentiment Analysis) ---
st.subheader("AI-Powered Recommendations (Conceptual)")

if sentiment_analyzer and not filtered_stations.empty:
    st.write("Using a conceptual AI model to 'score' station descriptions for recommendation.")
    # This is a *highly simplified* example. A real AI model would be trained
    # on features relevant to EV charging station selection.
    # Here, we're just running sentiment analysis on the description as a placeholder.
    # You would replace this with your actual model's prediction logic.

    # Batch prediction for efficiency
    descriptions = filtered_stations['description'].tolist()
    sentiment_results = sentiment_analyzer(descriptions)

    # Extract score (e.g., positive sentiment score)
    scores = []
    for res in sentiment_results:
        # Assuming 'POSITIVE' label for better stations, adjust based on your model's output
        if res['label'] == 'POSITIVE':
            scores.append(res['score'])
        else:
            scores.append(1 - res['score']) # Invert score for negative sentiment
    filtered_stations['ai_score'] = scores

    # Sort by AI score (descending) and then by distance
    filtered_stations = filtered_stations.sort_values(by=['ai_score', 'distance_km'], ascending=[False, True])
    st.dataframe(filtered_stations[['name', 'distance_km', 'rating', 'availability', 'ai_score', 'description']])

elif not filtered_stations.empty:
    st.warning("AI model not loaded. Displaying stations based on filters only.")
    st.dataframe(filtered_stations[['name', 'distance_km', 'rating', 'availability', 'description']])
else:
    st.info("No charging stations found matching your criteria.")


# --- 6. Display Results on a Map ---
st.subheader("Map of Charging Stations")

if not filtered_stations.empty:
    # Prepare data for map (Altair requires specific column names for lat/lon)
    map_data = filtered_stations[['latitude', 'longitude', 'name', 'availability', 'distance_km']].copy()
    map_data.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    # Add user's location to the map data
    user_location_df = pd.DataFrame({
        'lat': [user_latitude],
        'lon': [user_longitude],
        'name': ['Your Location'],
        'availability': ['N/A'],
        'distance_km': [0]
    })
    map_data = pd.concat([map_data, user_location_df], ignore_index=True)

    # Create the Altair chart
    chart = alt.Chart(map_data).mark_circle().encode(
        latitude='lat',
        longitude='lon',
        size=alt.Size('distance_km', scale=alt.Scale(range=[100, 1000]), legend=alt.Legend(title="Distance (km)")),
        color=alt.Color('availability', legend=alt.Legend(title="Availability")),
        tooltip=['name', 'distance_km', 'availability', 'rating']
    ).properties(
        title="EV Charging Stations Near You"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

else:
    st.info("No stations to display on the map.")

st.markdown("""
---
**Note on AI Model:** The AI model integration in this example uses a simple sentiment analysis model as a placeholder to demonstrate how a Hugging Face model can be incorporated. For a real-world EV charge locator, you would need to train or fine-tune a model specifically for this task, likely using datasets containing features like station popularity, historical availability, pricing, amenities, and user reviews.
""")
