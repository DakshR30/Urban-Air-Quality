"""
Star Schema Warehouse Modeling & OLAP Operations Engine
Constructs Fact_AirQuality, Dim_Location, Dim_Date relational views
and implements Slice, Dice, Roll-up / Drill-down, and Pivot operations.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

def build_star_schema(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Decomposes the flattened dataset into normalized Star Schema entities:
    Fact_AirQuality, Dim_Location, Dim_Date.
    """
    # 1. Dim_Location
    loc_cols = ["City", "State", "Zone", "Latitude", "Longitude"]
    dim_loc = df[loc_cols].drop_duplicates().reset_index(drop=True)
    dim_loc["LocationKey"] = [f"LOC_{i+1:03d}" for i in range(len(dim_loc))]

    # 2. Dim_Date
    date_cols = ["Date", "Year", "Month", "Month_Name", "Day", "Day_of_Week", "Is_Weekend", "Season"]
    dim_date = df[date_cols].drop_duplicates().reset_index(drop=True)
    dim_date["DateKey"] = pd.to_datetime(dim_date["Date"]).dt.strftime("%Y%m%d").astype(int)

    # 3. Fact_AirQuality
    fact_df = df.copy()
    fact_df["DateKey"] = pd.to_datetime(fact_df["Date"]).dt.strftime("%Y%m%d").astype(int)
    fact_df = fact_df.merge(dim_loc[["City", "LocationKey"]], on="City", how="left")
    fact_df["FactID"] = [f"FACT_{i+1:06d}" for i in range(len(fact_df))]

    fact_cols = ["FactID", "DateKey", "LocationKey", "PM2.5", "PM10", "NO2", "SO2", "CO", "AQI", "AQI_Bucket"]
    fact_table = fact_df[fact_cols].reset_index(drop=True)

    return {
        "Fact_AirQuality": fact_table,
        "Dim_Location": dim_loc,
        "Dim_Date": dim_date
    }

def slice_cube(df: pd.DataFrame, dimension: str, value: Any) -> pd.DataFrame:
    """
    OLAP Slice:
    Filters the multidimensional cube along a single dimension attribute.
    """
    if dimension not in df.columns:
        raise ValueError(f"Dimension '{dimension}' does not exist.")
    return df[df[dimension] == value].copy().reset_index(drop=True)

def dice_cube(
    df: pd.DataFrame,
    cities: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    aqi_range: Optional[tuple] = None
) -> pd.DataFrame:
    """
    OLAP Dice:
    Extracts a sub-cube by filtering across multiple dimensions simultaneously.
    """
    sub = df.copy()
    if cities:
        sub = sub[sub["City"].isin(cities)]
    if seasons:
        sub = sub[sub["Season"].isin(seasons)]
    if years:
        sub = sub[sub["Year"].isin(years)]
    if aqi_range:
        sub = sub[(sub["AQI"] >= aqi_range[0]) & (sub["AQI"] <= aqi_range[1])]
    return sub.reset_index(drop=True)

def rollup_drilldown(
    df: pd.DataFrame,
    level: str = "Monthly",
    agg_func: str = "Mean",
    pollutants: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    OLAP Roll-up & Drill-down:
    Transitions across temporal hierarchy levels:
    Drill-down: Yearly -> Seasonal -> Monthly -> Daily
    Roll-up: Daily -> Monthly -> Seasonal -> Yearly
    """
    if pollutants is None:
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "AQI"]

    fn_map = {"Mean": "mean", "Median": "median", "Max": "max", "Min": "min"}
    fn = fn_map.get(agg_func, "mean")

    if level == "Yearly":
        group_keys = ["City", "Year"]
    elif level == "Seasonal":
        group_keys = ["City", "Year", "Season"]
    elif level == "Monthly":
        group_keys = ["City", "Year", "Month_Name"]
    elif level == "Day_of_Week":
        group_keys = ["City", "Day_of_Week"]
    else:  # Daily
        group_keys = ["City", "Date"]

    agg_df = df.groupby(group_keys)[pollutants].agg(fn).reset_index()
    for p in pollutants:
        agg_df[p] = agg_df[p].round(2)
    return agg_df

def pivot_cube(
    df: pd.DataFrame,
    row_dim: str = "City",
    col_dim: str = "Season",
    metric: str = "AQI",
    agg_func: str = "Mean"
) -> pd.DataFrame:
    """
    OLAP Pivot:
    Rotates cube axes to generate cross-tabulated aggregate matrix.
    """
    fn_map = {"Mean": "mean", "Median": "median", "Max": "max", "Min": "min"}
    fn = fn_map.get(agg_func, "mean")
    matrix = pd.pivot_table(df, index=row_dim, columns=col_dim, values=metric, aggfunc=fn)
    return matrix.round(2)
