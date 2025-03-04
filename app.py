import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import xgboost as xgb
from datetime import timedelta
import requests
import logging
import plotly.express as px


st.set_page_config(page_title="AQI App", page_icon="🌍", layout="wide")
# ------------------------------------------------------------------
#                           LOGGING
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("aqi_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#                     DARK THEME & CUSTOM CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(45deg, #1e1e1e, #333333, #1e1e1e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .section {
        transition: all 1s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
#                   TITLE & BASIC DESCRIPTION
# ------------------------------------------------------------------
st.markdown(
    """
    <h1 style="text-align: center;">Lahore Air Quality Forecasting App</h1>
    <h3 style="text-align: center;">
        This app Fetches Real-Time pollutants & AQI data 24 times/day from OpenWeather API & forecasts the next three days Air Quality Index (AQI)
        for Lahore using a trained XGBoost model.
    </h3>
    """,
    unsafe_allow_html=True
)
# Refresh button to clear cache and re-run the app
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()
    
OPENWEATHERMAP_API_KEY = st.secrets["OPENWEATHERMAP_API_KEY"]
HOPSWORKS_API_KEY = st.secrets["HOPSWORKS_API_KEY"]
# ------------------------------------------------------------------
#                      HOPSWORKS CONNECTION
# ------------------------------------------------------------------
@st.cache_resource
def connect_to_hopsworks():
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        logger.info("Connected to Hopsworks successfully.")
        return fs, mr
    except Exception as e:
        logger.error(f"Failed to connect to Hopsworks: {e}")
        st.error("Failed to connect to Hopsworks. Please check your API key and connection.")
        return None, None

fs, mr = connect_to_hopsworks()

# ------------------------------------------------------------------
#                   FETCH CURRENT AQI FROM FEATURE STORE
# ------------------------------------------------------------------
@st.cache_data(ttl=300)  # Cache expires after 5 min
def fetch_current_aqi_record():
    try:
        features_fg = fs.get_feature_group("lahore_air_quality_features", version=1)
        targets_fg = fs.get_feature_group("lahore_air_quality_targets", version=1)
        
        features_df = features_fg.read()
        targets_df = targets_fg.read()
        
        df = pd.merge(features_df, targets_df, on="timestamp", how="inner")
        if df.empty:
            st.error("No data found in the Feature Store.")
            return None
        if "aqi" not in df.columns:
            st.error("Column 'aqi' not found in the merged DataFrame.")
            return None
        
        latest_record = df.sort_values("timestamp", ascending=False).iloc[0]
        return latest_record
    except Exception as e:
        logger.error(f"Error fetching current AQI: {e}")
        st.error("Failed to fetch current AQI. Please check the Feature Store data.")
        return None

# ------------------------------------------------------------------
#                  LOAD TRAINED MODEL FROM MODEL REGISTRY
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Loads the model with the highest version number named 'lahore_aqi_model' from Hopsworks.
    """
    try:
        # Fetch all models with the given name
        models = mr.get_models("lahore_aqi_model")
        if not models:
            raise ValueError("No models found with the name 'lahore_aqi_model'.")

        # Sort the models by version in descending order (highest version first)
        models_sorted = sorted(models, key=lambda m: m.version, reverse=True)
        latest_model = models_sorted[0]  # The model with the highest version

        # Download and load the XGBoost model
        model_dir = latest_model.download()
        xgb_model = xgb.Booster()
        xgb_model.load_model(f"{model_dir}/xgb_model.json")

        logger.info(f"Loaded model 'lahore_aqi_model' version {latest_model.version} successfully.")
        return xgb_model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Failed to load the trained model. Please check the Model Registry.")
        return None


# ------------------------------------------------------------------
#               HELPER FUNCTIONS FOR AQI LABEL/COLOR
# ------------------------------------------------------------------
def get_aqi_color(aqi_value):
    """Returns a hex color based on the AQI range."""
    if aqi_value <= 50:
        return "#00cc44"  # Good
    elif aqi_value <= 100:
        return "#C8A600"  # Moderate
    elif aqi_value <= 150:
        return "#CC5500"  # Unhealthy for Sensitive Groups
    elif aqi_value <= 200:
        return "#cd5c5c"  # Unhealthy
    elif aqi_value <= 300:
        return "#9900cc"  # Very Unhealthy
    else:
        return "#4E0068"  # Hazardous

def get_aqi_label(aqi_value):
    """Returns a textual label for the AQI category."""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_aqi_message(aqi_value):
    """Returns a brief recommendation based on AQI."""
    if aqi_value <= 50:
        return "Enjoy outdoor activities"
    elif aqi_value <= 100:
        return "Sensitive individuals should limit prolonged exertion"
    elif aqi_value <= 150:
        return "Sensitive groups may experience health effects"
    elif aqi_value <= 200:
        return "Everyone may begin to experience health effects"
    elif aqi_value <= 300:
        return "Significant health effects for everyone"
    else:
        return "Serious health effects for everyone"

def get_aqi_face(aqi_value):
    """Returns an emoji face based on AQI."""
    if aqi_value <= 50:
        return "😀"
    elif aqi_value <= 100:
        return "😐"
    elif aqi_value <= 150:
        return "🤧"  
    else:
        return "😷"

# ------------------------------------------------------------------
#                   FORECAST NEXT 3 DAYS AQI
# ------------------------------------------------------------------
def forecast_next_days(model, last_record, pollutant_cols, days=3, max_lag=3):
    try:
        aqi_predictions = []
        pm25_values = []
        pm10_values = []
        record = last_record.copy()
        
        for _ in range(days):
            required_features = [
                "pm25", "pm10", "no2", "so2", "co", "o3",
                "pm25_lag1", "pm25_lag2", "pm25_lag3",
                "pm10_lag1", "pm10_lag2", "pm10_lag3",
                "no2_lag1", "no2_lag2", "no2_lag3",
                "so2_lag1", "so2_lag2", "so2_lag3",
                "co_lag1", "co_lag2", "co_lag3",
                "o3_lag1", "o3_lag2", "o3_lag3",
                "aqi_lag1", "aqi_lag2", "aqi_lag3"
            ]
            X_last = record[required_features].to_frame().T.astype(float)
            dmatrix = xgb.DMatrix(X_last, feature_names=required_features)
            pred_aqi = int(round(model.predict(dmatrix)[0]))
            aqi_predictions.append(pred_aqi)
            pm25_values.append(record["pm25"])
            pm10_values.append(record["pm10"])
            
            # Advance record by 1 day, shift lags
            record["timestamp"] = pd.to_datetime(record["timestamp"]) + timedelta(days=1)
            record["aqi"] = pred_aqi
            for col in pollutant_cols + ["aqi"]:
                for lag in reversed(range(1, max_lag)):
                    record[f"{col}_lag{lag+1}"] = record[f"{col}_lag{lag}"]
                record[f"{col}_lag1"] = pred_aqi if col == "aqi" else record[col]
        
        return aqi_predictions, pm25_values, pm10_values
    except Exception as e:
        logger.error(f"Error forecasting AQI: {e}")
        st.error("Failed to forecast AQI. Please check the model and input data.")
        return None

# ------------------------------------------------------------------
#                FETCH LAST RECORD FOR FORECASTING
# ------------------------------------------------------------------
@st.cache_data(ttl=300)  # Cache expires after 5 min
def fetch_last_record():
    try:
        features_fg = fs.get_feature_group("lahore_air_quality_features", version=1)
        targets_fg = fs.get_feature_group("lahore_air_quality_targets", version=1)
        features_df = features_fg.read()
        targets_df = targets_fg.read()
        df = pd.merge(features_df, targets_df, on="timestamp", how="inner")
        
        pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]
        max_lag = 3
        for col in pollutant_cols + ["aqi"]:
            for lag in range(1, max_lag + 1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        
        df = df.dropna().reset_index(drop=True)
        last_record = df.sort_values("timestamp", ascending=False).iloc[0]
        return last_record
    except Exception as e:
        logger.error(f"Error fetching last record: {e}")
        st.error("Failed to fetch last record. Please check the Feature Store data.")
        return None

# ------------------------------------------------------------------
#               MAIN LOGIC & DATA FETCHING
# ------------------------------------------------------------------
current_aqi = None
next_three_days_pred = None

if fs and mr:
    last_record_for_current = fetch_current_aqi_record()
    if last_record_for_current is not None:
        current_aqi = last_record_for_current["aqi"]
    
    model = load_model()
    if model is not None:
        last_record_for_forecast = fetch_last_record()
        if last_record_for_forecast is not None:
            pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]
            forecast_results = forecast_next_days(model, last_record_for_forecast, pollutant_cols, days=3, max_lag=3)
            if forecast_results is not None:
                aqi_preds, pm25_vals, pm10_vals = forecast_results
                next_three_days_pred = {
                    "aqi": aqi_preds,
                    "pm25": pm25_vals,
                    "pm10": pm10_vals
                }

# ------------------------------------------------------------------
#                       DISPLAY CURRENT AQI
# ------------------------------------------------------------------
st.write("")
st.write("")

if current_aqi is not None and last_record_for_current is not None:
    current_pm25 = last_record_for_current["pm25"]
    current_pm10 = last_record_for_current["pm10"]

    aqi_label = get_aqi_label(current_aqi)
    current_aqi_bg_color = get_aqi_color(current_aqi)
    aqi_message = get_aqi_message(current_aqi)
    aqi_face = get_aqi_face(current_aqi)

    st.markdown(
        f"""
        <div style="
            background-color: {current_aqi_bg_color};
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            color: #fff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        " class="section">
            <div style="flex: 1;">
                <h2 style="margin: 0;">Today's AQI: {current_aqi} - {aqi_label}</h2>
                <p style="margin: 0; font-size: 1.5rem;"><strong>{aqi_message}</strong></p>
            </div>
            <div style="text-align: right;">
                <p style="font-size: 4rem; margin: 0;">{aqi_face}</p>
                <p style="margin: 0; font-size: 1.2rem;"><strong>Main Pollutant: PM2.5</strong></p>
                <p style="margin: 0; font-size: 1.2rem;">
                    <strong>PM2.5: {current_pm25} µg/m³ | PM10: {current_pm10} µg/m³</strong>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Unable to fetch current AQI. Please check the Feature Store data.")

# ------------------------------------------------------------------
#              POLLUTANT BREAKDOWN CHART 
# ------------------------------------------------------------------
st.write("")
st.write("")
if last_record_for_current is not None:
    st.subheader("Real Time Pollutants Breakdown")
    pollutant_data = {
        "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
        "Value": [
            last_record_for_current.get("pm25", 0),
            last_record_for_current.get("pm10", 0),
            last_record_for_current.get("no2", 0),
            last_record_for_current.get("so2", 0),
            last_record_for_current.get("co", 0),
            last_record_for_current.get("o3", 0)
        ]
    }
    df_pollutant = pd.DataFrame(pollutant_data)

    # A horizontal bar chart with distinct colors
    fig_breakdown = px.bar(
        df_pollutant,
        y="Pollutant",
        x="Value",
        color="Pollutant",
        orientation="h",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_breakdown.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font_color="#fff",
        hovermode="y unified"
    )
    fig_breakdown.update_traces(
        hovertemplate="<b>%{y}</b>: %{x:.2f} µg/m³<extra></extra>"
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

# ------------------------------------------------------------------
#                   DISPLAY NEXT 3 DAYS FORECAST
# ------------------------------------------------------------------
st.write("")
st.write("")
st.write("")
st.subheader("Next 3 Days AQI Forecast")
st.write("")

if next_three_days_pred is not None:
    aqi_preds = next_three_days_pred["aqi"]
    pm25_preds = next_three_days_pred["pm25"]
    pm10_preds = next_three_days_pred["pm10"]

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for i in range(len(aqi_preds)):
        day_aqi = aqi_preds[i]
        day_pm25 = pm25_preds[i]
        day_pm10 = pm10_preds[i]
        day_label = get_aqi_label(day_aqi)
        day_color = get_aqi_color(day_aqi)
        day_face = get_aqi_face(day_aqi)

        with columns[i]:
            st.markdown(
                f"""
                <div style="
                    background-color: {day_color};
                    border-radius: 10px;
                    padding: 1rem;
                    color: #fff;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                " class="section">
                    <div style="flex: 1;">
                        <h3 style="margin-top: 0;">Day {i+1}</h3>
                        <p style="font-size: 1.5rem; margin: 0;"><strong>AQI: {day_aqi}</strong></p>
                        <p style="margin: 0; font-size: 1.2rem;"><strong>{day_label}</strong></p>
                        <p style="margin: 0; font-size: 1.1rem;">
                           <strong>PM2.5: {day_pm25:.2f} µg/m³ | PM10: {day_pm10:.2f} µg/m³</strong>
                        </p>
                    </div>
                    <div style="margin-left: 1rem; font-size: 3rem;">
                        {day_face}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.warning("Unable to predict AQI for the next 3 days. Please check the model and data.")
# ------------------------------------------------------------------
#              DISPLAY: LAHORE LAST 2 YEARS HISTORICAL DATA
#              (Bar Chart with Custom Tooltip)
# ------------------------------------------------------------------
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.subheader("Lahore Historical Air Quality Data")
st.write("")
@st.cache_data
def fetch_historical_data():
    try:
        features_fg = fs.get_feature_group("lahore_air_quality_features", version=1)
        targets_fg = fs.get_feature_group("lahore_air_quality_targets", version=1)
        
        features_df = features_fg.read()
        targets_df = targets_fg.read()
        
        df = pd.merge(features_df, targets_df, on="timestamp", how="inner")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = pd.to_datetime("today") - pd.Timedelta(days=730)
        df = df[df['timestamp'] >= cutoff]

        # Group by date to get daily averages
        df['date'] = df['timestamp'].dt.date
        df = df.groupby('date').mean().reset_index()

        # Create label & formatted date strings for custom hover
        df["aqi_label"] = df["aqi"].apply(get_aqi_label)
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%a, %b %d")

        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        st.error("Failed to fetch historical data. Please check the Feature Store data.")
        return None

historical_data = fetch_historical_data()
if historical_data is not None and not historical_data.empty:
    aqi_color_map = {
        "Good": "#00cc44",
        "Moderate": "#C8A600",
        "Unhealthy for Sensitive Groups": "#CC5500",
        "Unhealthy": "#cd5c5c",
        "Very Unhealthy": "#9900cc",
        "Hazardous": "#4E0068"
    }

    fig_hist = px.bar(
        historical_data,
        x="date",
        y="aqi",
        color="aqi_label",
        color_discrete_map=aqi_color_map,
        template="plotly_dark",
        labels={"aqi": "AQI", "date": "Date", "aqi_label": "Category"},
        custom_data=["aqi", "aqi_label", "date_str"]
    )
    fig_hist.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='#fff',
        hovermode="x unified",
        legend=dict(itemclick='toggleothers', itemdoubleclick='toggle')
    )
    # Hovertemplate: "191 AQI US Unhealthy\nTue, Feb 04"
    fig_hist.update_traces(
        hovertemplate="<b>%{customdata[0]} AQI</b> %{customdata[1]}<br>%{customdata[2]}<extra></extra>"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.warning("No historical data found for the last 2 years.")


# ------------------------------------------------------------------
#                       FOOTER WITH SOCIAL MEDIA LINKS
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <h4 style="text-align: center;">Developed by Muhammad Talha</h4>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <!-- 1) Load Font Awesome CSS -->
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <!-- 2) Add your icons wrapped in links -->
    <div style='text-align: center; padding: 10px;'>
      <a href='https://github.com/Muhammad-Talha4k' target='_blank'
         style='color: #fff; margin: 10px; text-decoration: none;'>
        <i class="fab fa-github" style="font-size: 32px;"></i>
      </a>
      <a href='https://www.linkedin.com/in/muhammad-talha-sheikh-' target='_blank'
         style='color: #fff; margin: 10px; text-decoration: none;'>
        <i class="fab fa-linkedin" style="font-size: 32px;"></i>
      </a>
    </div>
    """,
    unsafe_allow_html=True
)
