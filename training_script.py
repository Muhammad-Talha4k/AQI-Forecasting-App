import hopsworks
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("aqi_model_training.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

def create_lag_features(df, pollutant_cols, max_lag=3):
    """
    Create lag features for each pollutant column and 'aqi'.
    For example, for lag=1, we create pm25_lag1, pm10_lag1, ..., aqi_lag1.
    """
    logger.info("Creating lag features...")
    # Sort by timestamp before creating lags
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in pollutant_cols + ["aqi"]:
        for lag in range(1, max_lag + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Drop rows with NaN caused by shifting (first 'max_lag' rows become NaN)
    df = df.dropna().reset_index(drop=True)
    logger.info("Lag features created successfully.")
    return df

def forecast_next_days(model, last_record, pollutant_cols, days=3, max_lag=3):
    """
    Forecast the next 'days' days starting from a single 'last_record'.
    This function updates lag features for each pollutant and 'aqi'.
    """
    logger.info("Forecasting next days...")
    predictions = []

    for _ in range(days):
        # Prepare the feature vector (exclude 'timestamp' and 'aqi')
        drop_cols = ["timestamp", "aqi"]
        X_cols = [c for c in last_record.index if c not in drop_cols]
        X_last = last_record[X_cols].values.reshape(1, -1)

        # Predict the next day's AQI
        pred_aqi = model.predict(X_last)[0]
        pred_aqi = int(round(pred_aqi))
        predictions.append(pred_aqi)

        # Advance the timestamp by one day
        last_record["timestamp"] = pd.to_datetime(last_record["timestamp"]) + timedelta(days=1)
        # Update the 'aqi' with the new prediction
        last_record["aqi"] = pred_aqi if col == "aqi" else last_record[col]

        # Now shift lag features for each pollutant and for 'aqi'
        # Example: aqi_lag1 <- aqi, aqi_lag2 <- aqi_lag1, ...
        for col in pollutant_cols + ["aqi"]:
            # shift the lags from oldest to newest
            for lag in reversed(range(1, max_lag)):
                last_record[f"{col}_lag{lag+1}"] = last_record[f"{col}_lag{lag}"]
            # The most recent lag1 becomes today's predicted (or known) value
            last_record[f"{col}_lag1"] = pred_aqi if col == "aqi" else last_record[col]

    logger.info("Forecasting completed.")
    return predictions

def main():
    try:
        logger.info("Starting AQI model training pipeline...")

        # 1. Connect to Hopsworks and fetch data
        logger.info("Connecting to Hopsworks...")
        project = hopsworks.login()
        fs = project.get_feature_store()

        # Fetch your features and targets
        logger.info("Fetching feature groups...")
        features_fg = fs.get_feature_group("lahore_air_quality_features", version=1)
        targets_fg = fs.get_feature_group("lahore_air_quality_targets", version=1)

        # Read from offline store
        logger.info("Reading feature and target data...")
        features_df = features_fg.read()
        targets_df = targets_fg.read()

        # Print columns of both DataFrames
        logger.info("Features DataFrame Columns: %s", features_df.columns.tolist())
        logger.info("Targets DataFrame Columns: %s", targets_df.columns.tolist())

        # Merge on "timestamp" to get a single DataFrame
        logger.info("Merging features and targets...")
        df = pd.merge(features_df, targets_df, on="timestamp", how="inner")

        # Print columns of the merged DataFrame
        logger.info("Merged DataFrame Columns: %s", df.columns.tolist())

        # Check for missing data in features_df
        logger.info("Checking for missing data...")
        logger.info("Missing data in features_df:\n%s", features_df.isnull().sum())

        # Set display options for full view
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        # List pollutant columns that are numeric and may have zero values
        pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]

        # Filter rows where any of the pollutant columns have a zero value
        rows_with_zero = df[(df[pollutant_cols] == 0).any(axis=1)]
        logger.info("Rows with zero values in any pollutant column:\n%s", tabulate(rows_with_zero, headers='keys', tablefmt='grid', showindex=False))

        # Display count of zero values for each pollutant column
        logger.info("Count of zero values in each pollutant column:")
        for col in pollutant_cols:
            count_zero = (df[col] == 0).sum()
            logger.info("  - %s: %d rows with zero", col, count_zero)

        # Define numeric pollutant columns
        numeric_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]

        # Filter out rows with any zero in the numeric pollutant columns
        df_filtered = df[~(df[numeric_cols] == 0).any(axis=1)]
        logger.info("Shape of DataFrame after dropping rows with zeros: %s", df_filtered.shape)

        # Filter rows where any of the pollutant columns have a zero value
        rows_with_zero = df_filtered[(df_filtered[pollutant_cols] == 0).any(axis=1)]
        logger.info("Rows with zero values in any pollutant column:\n%s", tabulate(rows_with_zero, headers='keys', tablefmt='grid', showindex=False))

        # Display count of zero values for each pollutant column
        logger.info("Count of zero values in each pollutant column:")
        for col in pollutant_cols:
            count_zero = (df_filtered[col] == 0).sum()
            logger.info("  - %s: %d rows with zero", col, count_zero)

        df = df_filtered.copy()

        # Sort by time
        df = df.sort_values("timestamp").reset_index(drop=True)

        ############################
        # 3. Create Lag Features
        ############################
        logger.info("Creating lag features...")
        df_lagged = create_lag_features(df, pollutant_cols, max_lag=3)

        ############################
        # 4. Train/Test Split
        ############################
        logger.info("Splitting data into train and test sets...")
        split_index = int(len(df_lagged) * 0.8)
        train_df = df_lagged.iloc[:split_index].copy()
        test_df = df_lagged.iloc[split_index:].copy()

        X_train = train_df.drop(["timestamp", "aqi"], axis=1)
        y_train = train_df["aqi"]

        ############################
        # 5. GridSearchCV
        ############################
        logger.info("Starting GridSearchCV for hyperparameter tuning...")
        xgb_model = xgb.XGBRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=20,
            verbose=3,
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)

        logger.info("Best Hyperparameters: %s", grid_search.best_params_)
        logger.info("Best CV Score (neg MSE): %s", grid_search.best_score_)

        best_model = grid_search.best_estimator_

        ############################
        # 6. Rolling Forecast
        ############################
        logger.info("Starting rolling forecast...")
        rolling_predictions = []
        rolling_actuals = []
        current_train = train_df.copy()

        # We'll walk through each day in the test set, one by one
        for idx, test_row in test_df.iterrows():
            # Retrain a new model on the expanded training set
            model_rolling = xgb.XGBRegressor(random_state=42, **grid_search.best_params_)

            X_current_train = current_train.drop(["timestamp", "aqi"], axis=1)
            y_current_train = current_train["aqi"]
            model_rolling.fit(X_current_train, y_current_train)

            # Predict for this test_row
            X_test_instance = test_row.drop(["timestamp", "aqi"]).values.reshape(1, -1)
            pred = model_rolling.predict(X_test_instance)[0]
            pred = int(round(pred))
            rolling_predictions.append(pred)
            rolling_actuals.append(test_row["aqi"])

            # Append the actual test_row to the training set
            current_train = pd.concat([current_train, pd.DataFrame([test_row])], ignore_index=True)

        # Evaluate rolling forecast
        mse = mean_squared_error(rolling_actuals, rolling_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(rolling_actuals, rolling_predictions)
        logger.info("Rolling Forecast Performance on Test Set:")
        logger.info("  - RMSE: %.2f", rmse)
        logger.info("  - R2:   %.2f", r2)

        ############################
        # 7. Store Model in Registry
        ############################
        logger.info("Saving model to Hopsworks Model Registry...")
        mr = project.get_model_registry()
        local_model_dir = "aqi_model"
        os.makedirs(local_model_dir, exist_ok=True)
        best_model.save_model(f"{local_model_dir}/xgb_model.json")

        model_meta = mr.python.create_model(
            name="lahore_aqi_model",
            metrics={"rmse": rmse, "r2": r2},
            description="XGBoost with lag features + rolling forecast"
        )
        model_meta.save(local_model_dir)
        logger.info("✅ Best model successfully saved to Hopsworks Model Registry.")

    except Exception as e:
        logger.error("An error occurred during the pipeline execution: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
    logger.info("✅ Finished updating data in Hopsworks!")
