import requests
import hopsworks
import pandas as pd
import time
import os
import logging
from datetime import datetime, timedelta
from hsfs.feature import Feature
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Read API keys from environment variables
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Validate API keys
if not API_KEY:
    raise ValueError("OPENWEATHERMAP_API_KEY environment variable is not set.")
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY environment variable is not set.")

# Constants
LATITUDE = 31.5204  # Latitude for Lahore
LONGITUDE = 74.3587  # Longitude for Lahore
BASE_URL = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}"

# Rate limiting variables
REQUEST_LIMIT = 60  # 60 requests per minute
REQUEST_WINDOW = 60  # 60 seconds
request_count = 0
last_request_time = time.time()

# Function to calculate US EPA AQI
def calculate_us_aqi(pm25, pm10):
    # AQI breakpoints for PM2.5 (µg/m³)
    pm25_breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    # AQI breakpoints for PM10 (µg/m³)
    pm10_breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]

    def calculate_aqi(concentration, breakpoints):
        for (c_low, c_high, i_low, i_high) in breakpoints:
            if c_low <= concentration <= c_high:
                return ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
        return -1  # Invalid concentration

    aqi_pm25 = calculate_aqi(pm25, pm25_breakpoints)
    aqi_pm10 = calculate_aqi(pm10, pm10_breakpoints)
    # Return the maximum AQI, rounded to the nearest integer.
    return int(round(max(aqi_pm25, aqi_pm10)))

# Fetch historical data for a specific day (using Unix timestamps)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_historical_data(start_timestamp, end_timestamp):
    global request_count, last_request_time

    # Check rate limit
    current_time = time.time()
    if current_time - last_request_time > REQUEST_WINDOW:
        # Reset request count if the window has passed
        request_count = 0
        last_request_time = current_time
    if request_count >= REQUEST_LIMIT:
        # Wait until the next window
        time_to_wait = REQUEST_WINDOW - (current_time - last_request_time)
        logger.warning(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
        time.sleep(time_to_wait)
        request_count = 0
        last_request_time = time.time()

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={LATITUDE}&lon={LONGITUDE}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    )
    response = requests.get(url)
    request_count += 1

    if response.status_code == 429:
        logger.warning("Rate limit hit. Waiting for 60 seconds...")
        time.sleep(60)  # Wait 60 seconds before retrying
        return fetch_historical_data(start_timestamp, end_timestamp)  # Retry request

    if response.status_code == 200:
        return response.json()

    raise Exception(f"Failed to fetch historical data: {response.status_code}")

# Fetch latest data
def fetch_latest_data():
    global request_count, last_request_time

    # Check rate limit
    current_time = time.time()
    if current_time - last_request_time > REQUEST_WINDOW:
        # Reset request count if the window has passed
        request_count = 0
        last_request_time = current_time
    if request_count >= REQUEST_LIMIT:
        # Wait until the next window
        time_to_wait = REQUEST_WINDOW - (current_time - last_request_time)
        logger.warning(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
        time.sleep(time_to_wait)
        request_count = 0
        last_request_time = time.time()

    response = requests.get(BASE_URL)
    request_count += 1

    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to fetch latest data: {response.status_code}")

# Process historical data
def process_historical_data(raw_data):
    if "list" not in raw_data or not raw_data["list"]:
        return []
    processed_data = []
    for entry in raw_data["list"]:
        timestamp = entry.get("dt", None)
        if not timestamp:
            continue
        components = entry.get("components", {})

        # Extract pollutant values
        pm25 = components.get("pm2_5", -1)
        pm10 = components.get("pm10", -1)
        no2 = components.get("no2", -1)
        so2 = components.get("so2", -1)
        co = components.get("co", -1)
        o3 = components.get("o3", -1)

        # Handle negative values for each feature
        if pm25 < 0:
            pm25 = 0  # Replace negative PM2.5 with 0
        if pm10 < 0:
            pm10 = 0  # Replace negative PM10 with 0
        if no2 < 0:
            no2 = 0  # Replace negative NO2 with 0
        if so2 < 0:
            so2 = 0  # Replace negative SO2 with 0
        if co < 0:
            co = 0  # Replace negative CO with 0
        if o3 < 0:
            o3 = 0  # Replace negative O3 with 0

        # Create features dictionary
        features = {
            "timestamp": pd.to_datetime(timestamp, unit="s").strftime("%Y-%m-%d"),
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "co": co,
            "o3": o3,
        }

        # Calculate US EPA AQI
        us_aqi = calculate_us_aqi(pm25, pm10)
        target = {
            "timestamp": pd.to_datetime(timestamp, unit="s").strftime("%Y-%m-%d"),
            "aqi": int(us_aqi),
        }
        processed_data.append((features, target))
    return processed_data

# Process latest data
def process_latest_data(raw_data):
    if "list" not in raw_data or not raw_data["list"]:
        return []
    processed_data = []
    for entry in raw_data["list"]:
        timestamp = entry.get("dt", None)
        if not timestamp:
            continue
        components = entry.get("components", {})

        # Extract pollutant values
        pm25 = components.get("pm2_5", -1)
        pm10 = components.get("pm10", -1)
        no2 = components.get("no2", -1)
        so2 = components.get("so2", -1)
        co = components.get("co", -1)
        o3 = components.get("o3", -1)

        # Handle negative values for each feature
        if pm25 < 0:
            pm25 = 0  # Replace negative PM2.5 with 0
        if pm10 < 0:
            pm10 = 0  # Replace negative PM10 with 0
        if no2 < 0:
            no2 = 0  # Replace negative NO2 with 0
        if so2 < 0:
            so2 = 0  # Replace negative SO2 with 0
        if co < 0:
            co = 0  # Replace negative CO with 0
        if o3 < 0:
            o3 = 0  # Replace negative O3 with 0

        # Create features dictionary
        features = {
            "timestamp": pd.to_datetime(timestamp, unit="s").strftime("%Y-%m-%d"),
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "co": co,
            "o3": o3,
        }

        # Calculate US EPA AQI
        us_aqi = calculate_us_aqi(pm25, pm10)
        target = {
            "timestamp": pd.to_datetime(timestamp, unit="s").strftime("%Y-%m-%d"),
            "aqi": int(us_aqi),
        }
        processed_data.append((features, target))
    return processed_data

# Aggregate daily data
def aggregate_daily_data(processed_data, day_date):
    # Filter out entries where AQI is -1
    valid_entries = [(feat, targ) for feat, targ in processed_data if targ["aqi"] != -1]
    if not valid_entries:
        return None, None
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    aggregated_features = {}
    for pollutant in pollutants:
        vals = [float(feat[pollutant]) for feat, _ in valid_entries]
        aggregated_features[pollutant] = round(sum(vals) / len(vals), 4)
    aggregated_features["timestamp"] = day_date.strftime("%Y-%m-%d")
    aqi_values = [targ["aqi"] for _, targ in valid_entries]
    aggregated_target = {
        "timestamp": day_date.strftime("%Y-%m-%d"),
        "aqi": int(round(sum(aqi_values) / len(aqi_values))),
    }
    return aggregated_features, aggregated_target

# Backfill historical data
def backfill_historical_data():
    # Define the period: last 728 days up to yesterday.
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=728)
    aggregated_features_list = []
    aggregated_target_list = []
    current_date = start_date
    while current_date <= end_date:
        day_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Processing data for {day_str} ...")
        # Define the day's time range.
        start_timestamp = int(current_date.replace(hour=0, minute=0, second=0).timestamp())
        end_timestamp = int(current_date.replace(hour=23, minute=59, second=59).timestamp())
        try:
            raw_data = fetch_historical_data(start_timestamp, end_timestamp)
            processed_data = process_historical_data(raw_data)
            day_features, day_target = aggregate_daily_data(processed_data, current_date)
            if day_features is not None and day_target is not None:
                aggregated_features_list.append(day_features)
                aggregated_target_list.append(day_target)
                logger.info(f"  ✓ Valid data found for {day_str}")
            else:
                logger.warning(f"  ⨯ No valid data for {day_str}, skipping.")
        except Exception as e:
            logger.error(f"  ⚠ Error for {day_str}: {e}")
        current_date += timedelta(days=1)

    if aggregated_features_list and aggregated_target_list:
        try:
            project = hopsworks.login()
            fs = project.get_feature_store()
            # Ensure the feature groups exist (or create them)
            try:
                feature_group = fs.get_feature_group("lahore_air_quality_features", version=1)
                logger.info("Feature group 'lahore_air_quality_features' already exists.")
            except:
                logger.info("Feature group 'lahore_air_quality_features' does not exist. Creating it...")
                feature_group = fs.create_feature_group(
                    name="lahore_air_quality_features",
                    version=1,
                    primary_key=["timestamp"],
                    description="Daily aggregated air quality data for Lahore from OpenWeatherMap",
                    features=[
                        Feature("timestamp", type="string"),
                        Feature("pm25", type="double"),
                        Feature("pm10", type="double"),
                        Feature("no2", type="double"),
                        Feature("so2", type="double"),
                        Feature("co", type="double"),
                        Feature("o3", type="double"),
                    ],
                    online_enabled=True,
                )
                time.sleep(30)  # Wait for 30 seconds after creating the Feature Group
            try:
                target_group = fs.get_feature_group("lahore_air_quality_targets", version=1)
                logger.info("Feature group 'lahore_air_quality_targets' already exists.")
            except:
                logger.info("Feature group 'lahore_air_quality_targets' does not exist. Creating it...")
                target_group = fs.create_feature_group(
                    name="lahore_air_quality_targets",
                    version=1,
                    primary_key=["timestamp"],
                    description="Daily aggregated Air Quality Index (AQI) target values for Lahore",
                    features=[
                        Feature("timestamp", type="string"),
                        Feature("aqi", type="bigint"),
                    ],
                    online_enabled=True,
                )
                time.sleep(30)  # Wait for 30 seconds after creating the Feature Group
            # Build DataFrames from the aggregated lists
            features_df = pd.DataFrame(aggregated_features_list)
            target_df = pd.DataFrame(aggregated_target_list)
            features_df = features_df.astype({
                "timestamp": "string",
                "pm25": "float64",
                "pm10": "float64",
                "no2": "float64",
                "so2": "float64",
                "co": "float64",
                "o3": "float64",
            })
            target_df = target_df.astype({
                "timestamp": "string",
                "aqi": "int64",
            })
            # Print DataFrame schema
            logger.info("Features DataFrame Schema:")
            logger.info(features_df.dtypes)
            logger.info("Targets DataFrame Schema:")
            logger.info(target_df.dtypes)
            # Print Feature Group schema
            logger.info("Feature Group Schema:")
            logger.info(feature_group.schema)
            logger.info("Target Group Schema:")
            logger.info(target_group.schema)
            # Print the first few rows of the DataFrames
            logger.info("Features DataFrame:")
            logger.info(features_df.head())
            logger.info("Targets DataFrame:")
            logger.info(target_df.head())
            # Check if the feature group is empty
            try:
                existing_features_df = feature_group.read()
                existing_targets_df = target_group.read()
                if existing_features_df.empty or existing_targets_df.empty:
                    logger.info("Feature groups are empty. Inserting new data directly.")
                    updated_features_df = features_df
                    updated_targets_df = target_df
                else:
                    # Append new data to existing data (avoid duplicates)
                    updated_features_df = pd.concat([existing_features_df, features_df]).drop_duplicates(subset=["timestamp"])
                    updated_targets_df = pd.concat([existing_targets_df, target_df]).drop_duplicates(subset=["timestamp"])
            except Exception as e:
                logger.error(f"Error reading from feature groups. Inserting new data directly. Error: {e}")
                updated_features_df = features_df
                updated_targets_df = target_df
            # Insert updated data into the Feature Store (without overwriting)
            feature_group.insert(updated_features_df, overwrite=False)  # Set overwrite=False
            target_group.insert(updated_targets_df, overwrite=False)    # Set overwrite=False
            logger.info("Incremental update complete.")
        except Exception as e:
            logger.error(f"Failed to update Hopsworks: {e}")
            raise
    else:
        logger.warning("No aggregated data to store.")

# Fetch and process latest data
def fetch_and_process_latest_data():
    # Fetch latest data
    raw_data = fetch_latest_data()
    processed_data = process_latest_data(raw_data)
    if not processed_data:
        logger.warning("No latest data to process.")
        return
    # Aggregate latest data
    latest_date = datetime.now().strftime("%Y-%m-%d")
    latest_features, latest_target = aggregate_daily_data(processed_data, datetime.now())
    if latest_features is None or latest_target is None:
        logger.warning("No valid latest data to store.")
        return
    # Insert latest data into Hopsworks
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        try:
            feature_group = fs.get_feature_group("lahore_air_quality_features", version=1)
            target_group = fs.get_feature_group("lahore_air_quality_targets", version=1)
        except Exception as e:
            logger.error(f"Error fetching feature groups: {e}")
            return
        # Create DataFrames for latest data
        latest_features_df = pd.DataFrame([latest_features])
        latest_target_df = pd.DataFrame([latest_target])
        # Append latest data to existing data (avoid duplicates)
        try:
            existing_features_df = feature_group.read()
            existing_targets_df = target_group.read()
            updated_features_df = pd.concat([existing_features_df, latest_features_df]).drop_duplicates(subset=["timestamp"])
            updated_targets_df = pd.concat([existing_targets_df, latest_target_df]).drop_duplicates(subset=["timestamp"])
        except Exception as e:
            logger.error(f"Error reading from feature groups. Inserting latest data directly. Error: {e}")
            updated_features_df = latest_features_df
            updated_targets_df = latest_target_df
        # Insert updated data into the Feature Store (without overwriting)
        feature_group.insert(updated_features_df, overwrite=False)  # Set overwrite=False
        target_group.insert(updated_targets_df, overwrite=False)    # Set overwrite=False
        logger.info("Latest data update complete.")
    except Exception as e:
        logger.error(f"Failed to update Hopsworks: {e}")
        raise

# Main function
def main():
    try:
        # Check if historical data has already been backfilled
        project = hopsworks.login()
        fs = project.get_feature_store()
        try:
            feature_group = fs.get_feature_group("lahore_air_quality_features", version=1)
            target_group = fs.get_feature_group("lahore_air_quality_targets", version=1)
            existing_features_df = feature_group.read()
            existing_targets_df = target_group.read()
            if existing_features_df.empty or existing_targets_df.empty:
                logger.info("Historical data not found. Backfilling historical data...")
                backfill_historical_data()
            else:
                logger.info("Historical data already exists. Fetching latest data...")
                fetch_and_process_latest_data()
        except Exception as e:
            logger.error(f"Error checking feature groups: {e}")
            logger.info("Backfilling historical data...")
            backfill_historical_data()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main()
    logger.info("✅ Finished updating data in Hopsworks!")