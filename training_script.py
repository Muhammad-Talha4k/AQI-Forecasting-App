import hopsworks
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import logging
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. Connect to Hopsworks and fetch data
        project = hopsworks.login()
        fs = project.get_feature_store()

        # Fetch your features and targets
        features_fg = fs.get_feature_group("lahore_air_quality_features", version=1)
        targets_fg = fs.get_feature_group("lahore_air_quality_targets", version=1)

        # Read from offline store
        features_df = features_fg.read()
        targets_df = targets_fg.read()

        # Print columns of both DataFrames
        logger.info("Features DataFrame Columns: %s", features_df.columns.tolist())
        logger.info("Targets DataFrame Columns: %s", targets_df.columns.tolist())

        # Merge on "timestamp" to get a single DataFrame
        df = pd.merge(features_df, targets_df, on="timestamp", how="inner")

        # Print columns of the merged DataFrame
        logger.info("Merged DataFrame Columns: %s", df.columns.tolist())

        # Check for missing data in features_df
        logger.info("Missing data in features_df:")
        logger.info(features_df.isnull().sum())

        # Set display options for full view
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        # Display merged DataFrame using tabulate
        logger.info("\nMerged DataFrame:")
        logger.info(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        # List pollutant columns that are numeric and may have zero values
        pollutant_columns = ["pm25", "pm10", "no2", "so2", "co", "o3"]

        # Filter rows where any of the pollutant columns have a zero value
        rows_with_zero = df[(df[pollutant_columns] == 0).any(axis=1)]
        logger.info("\nRows with zero values in any pollutant column:")
        logger.info(tabulate(rows_with_zero, headers='keys', tablefmt='grid', showindex=False))

        # Display count of zero values for each pollutant column
        logger.info("\nCount of zero values in each pollutant column:")
        for col in pollutant_columns:
            count_zero = (df[col] == 0).sum()
            logger.info(f"  - {col}: {count_zero} rows with zero")

        # Define numeric pollutant columns
        numeric_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]

        # Filter out rows with any zero in the numeric pollutant columns
        df_filtered = df[~(df[numeric_cols] == 0).any(axis=1)]
        logger.info("Shape of DataFrame after dropping rows with zeros: %s", df_filtered.shape)

        # Filter rows where any of the pollutant columns have a zero value
        rows_with_zero = df_filtered[(df_filtered[pollutant_columns] == 0).any(axis=1)]
        logger.info("\nRows with zero values in any pollutant column:")
        logger.info(tabulate(rows_with_zero, headers='keys', tablefmt='grid', showindex=False))

        # Display count of zero values for each pollutant column
        logger.info("\nCount of zero values in each pollutant column:")
        for col in pollutant_columns:
            count_zero = (df_filtered[col] == 0).sum()
            logger.info(f"  - {col}: {count_zero} rows with zero")

        df = df_filtered.copy()
        # Prepare features (X) and target (y)
        X = df.drop(["timestamp", "aqi"], axis=1)
        y = df["aqi"]

        # 2. Train-test split (we’ll do CV only on the training set)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("X shape: %s", X.shape)
        logger.info("y shape: %s", y.shape)

        # 3. Define an XGBoost model and a parameter grid for GridSearchCV
        xgb_model = xgb.XGBRegressor(random_state=42)

        # Define the parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1, 0.2]
        }

        # 4. Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=10,
            verbose=3,
            n_jobs=1
        )

        # Fit GridSearchCV on the training data
        grid_search.fit(X_train, y_train)

        # Print out the best parameters found by GridSearchCV
        logger.info("Best Hyperparameters: %s", grid_search.best_params_)
        logger.info("Best CV Score (neg MSE): %s", grid_search.best_score_)

        # 5. Evaluate the best model on the test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        logger.info("\nTest Set Performance of Best Model:")
        logger.info(f"  - RMSE: {rmse:.2f}")
        logger.info(f"  - R2:   {r2:.2f}")

        # 6. Store the best model in the Hopsworks Model Registry
        mr = project.get_model_registry()

        # Create a local directory to save the model artifacts
        local_model_dir = "aqi_model"
        os.makedirs(local_model_dir, exist_ok=True)

        # Save the best model
        best_model.save_model(f"{local_model_dir}/xgb_model.json")

        # Create a new model entry in the registry
        model_meta = mr.python.create_model(
            name="lahore_aqi_model",
            metrics={"rmse": rmse, "r2": r2},
            description="XGBoost model predicting Lahore AQI with hyperparameter tuning via GridSearchCV"
        )

        # Upload model artifacts to the Model Registry
        model_meta.save(local_model_dir)
        logger.info("✅ Best model successfully saved to Hopsworks Model Registry.")

    except Exception as e:
        logger.error("Script failed: %s", e)
        sys.exit(1)  # Exit with a non-zero status code to indicate failure

if __name__ == "__main__":
    main()
    logger.info("✅ Finished updating data in Hopsworks!")