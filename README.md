# Urban Air Quality & Pollution Forecasting System
### Multidimensional Data Warehousing & Data Mining (DWM) Application

A production-grade, standalone **Streamlit** application engineered for atmospheric data warehousing, multidimensional OLAP analytics, and machine learning air quality classification utilizing **authentic public Central Pollution Control Board (CPCB, India) environmental sensor records (>20,000 observations)**.

---

## 🏛️ System Architecture

### 1. Data Ingestion & Star Schema Warehouse
- **Authentic Data Source:** Official daily ambient monitoring station measurements from major Indian metropolitan centers (Delhi, Ahmedabad, Bengaluru, Mumbai, Chennai, Hyderabad, Kolkata, etc.).
- **Features Ingested:** `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `AQI`, and `AQI_Bucket`.
- **Relational Star Schema:**
  - `Fact_AirQuality`: Stores daily observations, pollutant measures, computed index, and surrogate keys (`FactID`, `DateKey`, `LocationKey`).
  - `Dim_Location`: Spatial dimension with `LocationKey`, `City`, `State`, `Zone`, `Latitude`, `Longitude`.
  - `Dim_Date`: Temporal dimension with `DateKey`, `Date`, `Year`, `Month`, `Season`, `Day_of_Week`.

```
                        +----------------------+
                        |       Dim_Date       |
                        +----------------------+
                        | DateKey (PK)         |
                        | Date                 |
                        | Year, Month, Day     |
                        | Season, Day_of_Week  |
                        +----------+-----------+
                                   | 1
                                   |
                                   | *
+----------------------+ *         |         * +----------------------+
|     Dim_Location     +-----------+-----------+   Fact_AirQuality    |
+----------------------+ 1                     +----------------------+
| LocationKey (PK)     |                       | FactID (PK)          |
| City                 |                       | DateKey (FK)         |
| State, Zone          |                       | LocationKey (FK)     |
| Latitude, Longitude  |                       | PM2.5, PM10          |
+----------------------+                       | NO2, SO2, CO         |
                                               | AQI, AQI_Bucket      |
                                               +----------------------+
```

### 2. Multidimensional OLAP Cube Analytics
- **Slice:** Filter across a single dimension (e.g., `Season = 'Winter'`).
- **Dice:** Multi-dimensional sub-cube extraction (e.g., specific cities, seasons, years, and AQI bounds).
- **Roll-up & Drill-down:** Temporal aggregation transitions (`Daily ➔ Monthly ➔ Seasonal ➔ Yearly`) with flexible aggregate operators (`Mean`, `Median`, `Max`, `Min`).
- **Pivot:** Axis reorientation producing interactive cross-tabulated matrices and 2D Plotly heatmaps.

### 3. Machine Learning Mining Pipeline
Four distinct classification models trained on standardized pollutant features:
1. **Linear Regression (Threshold Classifier):** Continuous AQI estimator mapped to discrete severity buckets via standard CPCB cutoffs (`[0, 50] ➔ Good`, `(50, 100] ➔ Satisfactory`, etc.).
2. **Multinomial Logistic Regression:** Softmax multiclass classification with L-BFGS optimizer.
3. **J48 / C4.5 Decision Tree:** Ross Quinlan's information gain / entropy criterion (`DecisionTreeClassifier(criterion='entropy')`).
4. **Random Forest Classifier:** Bagged ensemble of 100 entropy decision trees.

### 4. Dedicated Visualizations & Batch Inference
- Consolidated evaluation table reporting **Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-Score**.
- **Four Dedicated, Separate Interactive Plotly Graphs:**
  - Graph 1: Comparative Accuracy Across Models
  - Graph 2: Comparative Precision Across Models
  - Graph 3: Comparative Recall Across Models
  - Graph 4: Comparative F1-Score Across Models
- Interactive multi-class confusion matrix heatmaps.
- **Batch CSV Ingestion Engine:** Validates required pollutant columns (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`), dynamic model selection, and instant one-click CSV export (`st.download_button`).

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📜 Regulatory Standards (CPCB India)

| AQI Range | Bucket | Health Implications |
| :---: | :---: | :--- |
| **0 – 50** | **Good** | Minimal health impact |
| **51 – 100** | **Satisfactory** | Minor breathing discomfort to sensitive people |
| **101 – 200** | **Moderate** | Breathing discomfort to people with asthma and heart disease |
| **201 – 300** | **Poor** | Breathing discomfort to most people on prolonged exposure |
| **301 – 400** | **Very Poor** | Respiratory illness on prolonged exposure |
| **401 – 500+** | **Severe** | Affects healthy people and severely impacts those with existing disease |
