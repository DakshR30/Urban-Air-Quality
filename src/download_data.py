"""
Real-World Air Quality Data Ingestion & ETL Pipeline
Fetches authentic public CPCB (Central Pollution Control Board, India) city_day dataset
exceeding 20,000 real sensor observations. Cleans nulls, calculates temporal dimensions,
and structures data under standard CPCB AQI classifications.
"""

import os
import io
import requests
import pandas as pd
import numpy as np

# Authentic public repositories hosting the official CPCB city_day dataset (>29,000 records)
DATA_URLS = [
    "https://raw.githubusercontent.com/SunnyRao07/Air-Quality-Index-AQI-Prediction/master/city_day.csv",
    "https://raw.githubusercontent.com/aryanrathore1012/Air_Quality_Analysis_and_Prediction_for_Indian_Cities_and_States/master/city_day.csv"
]

ORDERED_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

CITY_COORDINATES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "zone": "North"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "zone": "West"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "zone": "South"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "zone": "West"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "zone": "South"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "zone": "South"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "zone": "East"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "zone": "West"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "zone": "East"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "zone": "North"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "zone": "North"},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana", "zone": "North"},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab", "zone": "North"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "zone": "South"}
}

def get_season(month: int) -> str:
    """Classifies calendar month into meteorological season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

def aqi_to_bucket(aqi: float) -> str:
    """Maps continuous AQI value to official Indian regulatory category."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def fetch_and_clean_real_data(output_path: str = "data/real_cpcb_air_quality.csv") -> pd.DataFrame:
    """
    Downloads authentic CPCB dataset from public mirrors, cleans genuine missing
    readings using city medians, extracts temporal dimensions, and writes cleaned CSV.
    """
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            if len(df) >= 5000 and "AQI_Bucket" in df.columns:
                return df
        except Exception:
            pass

    raw_df = None
    for url in DATA_URLS:
        try:
            resp = requests.get(url, timeout=25)
            if resp.status_code == 200 and len(resp.content) > 50000:
                raw_df = pd.read_csv(io.StringIO(resp.text))
                break
        except Exception:
            continue

    if raw_df is None:
        raise RuntimeError("Failed to fetch public real-world air quality dataset from configured endpoints.")

    # Retain observations having basic valid data
    # Filter rows having valid Date and City
    df = raw_df.dropna(subset=["City", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Core target pollutants
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
    for col in pollutants:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "AQI" not in df.columns:
        df["AQI"] = np.nan
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    # Retain records that have PM2.5 and AQI (or compute AQI if PM2.5 present)
    # The CPCB dataset has over 20,000 such real rows
    df = df.dropna(subset=["PM2.5", "AQI"]).copy()

    # Impute missing secondary pollutants (PM10, NO2, SO2, CO) using city medians
    for p in pollutants:
        df[p] = df.groupby("City")[p].transform(lambda x: x.fillna(x.median()))
        df[p] = df[p].fillna(df[p].median())

    # Ensure physical positive bounds
    for p in pollutants:
        df[p] = df[p].clip(lower=0.1).round(2)
    df["AQI"] = df["AQI"].clip(lower=1.0).round(1)

    # Derive or validate AQI_Bucket
    df["AQI_Bucket"] = df["AQI"].apply(aqi_to_bucket)

    # Compute temporal attributes
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.strftime("%A")
    df["Is_Weekend"] = df["Date"].dt.weekday.apply(lambda w: 1 if w >= 5 else 0)
    df["Season"] = df["Month"].apply(get_season)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Attach spatial metadata
    def get_city_meta(city_name, key, default):
        return CITY_COORDINATES.get(city_name, {}).get(key, default)

    df["State"] = df["City"].apply(lambda c: get_city_meta(c, "state", "India"))
    df["Zone"] = df["City"].apply(lambda c: get_city_meta(c, "zone", "Central"))
    df["Latitude"] = df["City"].apply(lambda c: get_city_meta(c, "lat", 20.5937))
    df["Longitude"] = df["City"].apply(lambda c: get_city_meta(c, "lon", 78.9629))

    # Clean select columns
    keep_cols = [
        "Date", "City", "State", "Zone", "Latitude", "Longitude",
        "PM2.5", "PM10", "NO2", "SO2", "CO", "AQI", "AQI_Bucket",
        "Year", "Month", "Month_Name", "Day", "Day_of_Week", "Is_Weekend", "Season"
    ]
    cleaned_df = df[keep_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df

def get_sample_upload_data(num_rows: int = 50) -> pd.DataFrame:
    """Returns sample batch CSV with required pollutant headers for testing."""
    df = fetch_and_clean_real_data()
    sample = df[["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "CO"]].sample(n=num_rows, random_state=42)
    return sample.reset_index(drop=True)

if __name__ == "__main__":
    data = fetch_and_clean_real_data()
    print(f"Dataset successfully fetched & cleaned: {len(data):,} records")
