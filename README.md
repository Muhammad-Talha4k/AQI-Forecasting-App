[![AQI Prediction Pipeline](https://github.com/Muhammad-Talha4k/AQI-Forecasting-App/actions/workflows/pipeline.yml/badge.svg?branch=main)](https://github.com/Muhammad-Talha4k/AQI-Forecasting-App/actions/workflows/pipeline.yml)
# Air Quality Index (AQI) Forecasting App

## Overview

This project is a Machine Learning Operations (MLOps) application that predicts the Air Quality Index (AQI) for Lahore, Pakistan, using a 100% serverless stack. The app fetches real-time air quality data from the OpenWeather API, processes it, and forecasts the AQI for the next three days using a trained XGBoost model. The entire pipeline is automated using GitHub Actions, and the app is deployed using Streamlit.

## Live app 
- Check out the deployed app here: [AQI Forecasting App](https://aqi-forecasting-app.streamlit.app/)

## Features

- **Daily AQI Prediction**: Fetches real-time air quality data and forecasts the AQI for the next three days.
- **Serverless Architecture**: Built using a serverless stack, including GitHub Actions for automation and Hopsworks for feature storage and model registry.
- **Interactive Web App**: A Streamlit-based web app that displays real-time AQI, pollutant breakdown, and historical data.
- **Automated Pipelines**: Automated feature extraction, model training, and prediction pipelines using GitHub Actions.

## Project Structure
```
├── .devcontainer/
│ └── devcontainer.json
├── .github/workflows/
│ └── Pipeline.yaml
├── README.md
├── app.py
├── features_script.py
├── training_script.py
└── requirements.txt
```

## Workflow

The workflow of this project is divided into these steps:

- `Feature Pipeline`: Fetches raw weather and pollutant data from an external API, computes features, and stores them in the Hopsworks Feature Store.

- `Backfill Historical Data`: Runs the feature script for a range of past dates to generate training data for the ML model.
  
- `Fill Realtime Data`: If historical data is already filled then fetches real time data from API and updates in hopsworks feature store.

- `Training Pipeline`: Fetches historical data from the Feature Store, trains an XGBoost model, and stores the trained model in the Hopsworks Model Registry.

- `Automate Pipeline Runs`: Uses GitHub Actions to automatically run the feature script every hour and the training script every week.

- `Web App`: A Streamlit app that loads the model and features from the Feature Store, computes predictions, and displays them on a user-friendly interface.

- `Model Deployment`: The trained model is deployed and used to make real-time predictions in the web app.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Muhammad-Talha4k/AQI-Forecasting-App.git
   ```
2. **Prerequisites**:
- `Python 3.8+`
- `Open Wheather or any other API key`
- `Hopsworks API key`

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
Once the app is running, you can:

- **View Daily AQI** for Lahore.
- **View Daily pollutants breakdown** (PM2.5, PM10, NO2, SO2, CO, O3).
- **Get a 3-day AQI forecast**.
- **Explore historical AQI data** for the past two years.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.
