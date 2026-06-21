# King County Housing Price Prediction

An end-to-end machine learning pipeline for predicting residential housing prices in King County, Washington. The project uses Snowflake as the cloud data warehouse, dbt for data transformation and cleaning, and scikit-learn for model training and evaluation.

---

## Table of Contents

1. [Results and Findings](#1-results-and-findings)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Data Pipeline and Architecture](#3-data-pipeline-and-architecture)
4. [Project Structure](#4-project-structure)
5. [Setup and Installation](#5-setup-and-installation)
6. [Configuration](#6-configuration)
7. [Running the Project](#7-running-the-project)
8. [Tech Stack](#8-tech-stack)

---

## 1. Results and Findings

### Model Performance

Three regression models were trained and compared. The dataset was split into 80% training and 20% test sets. All input features were normalized using `StandardScaler` before training.

| Model | R2 Score | RMSE (USD) |
| --- | --- | --- |
| Random Forest Regressor | 0.7648 | 152,923 |
| Linear Regression | 0.7301 | 170,824 |
| Decision Tree Regressor | 0.6751 | 197,520 |

**Best Model: Random Forest Regressor** with `max_depth=6` and `random_state=42`.

The Random Forest model achieved the highest R2 score of 0.7648, meaning it explains approximately 76.5% of the variance in housing prices. It also produced the lowest prediction error at roughly $152,923 per home. Linear Regression performed competitively at 73%, suggesting that strong linear relationships exist between the selected features and price. The Decision Tree showed the weakest performance at 67.5%, likely due to underfitting caused by the shallow depth constraint.

---

### Feature Correlation with Price

The table below ranks all features by their Pearson correlation with the target variable `PRICE`. Features with a higher absolute correlation value have a stronger relationship with price.

| Feature | Correlation | Interpretation |
| --- | --- | --- |
| SQFT_LIVING | +0.702 | The interior living area is the single strongest predictor of price. Larger homes consistently command higher prices. |
| ZIPCODE | +0.638 | Encoded as the average sale price per zip code, this feature captures neighborhood-level demand and location quality. |
| SQFT_ABOVE | +0.605 | Above-ground square footage reinforces the effect of interior size. |
| SQFT_LIVING15 | +0.585 | The average living area of the 15 nearest neighbors reflects the quality of the surrounding neighborhood. |
| BATHROOMS | +0.475 | Bathroom count serves as a reliable proxy for overall home quality and size tier. |
| VIEW | +0.397 | Homes with better view ratings carry a measurable price premium. |
| SQFT_BASEMENT | +0.324 | Additional basement square footage contributes positively to price. |
| BEDROOMS | +0.308 | Bedroom count has a moderate but limited effect. By itself, it is a weak predictor. |
| GRADE | +0.292 | The construction and design quality grade assigned by King County assessors has a meaningful impact on price. |
| distance_to_seattle_km | -0.287 | Homes farther from downtown Seattle sell for less. Proximity to the city center commands a consistent price premium. |
| WATERFRONT | +0.266 | Waterfront properties are priced significantly higher than non-waterfront homes. |
| FLOORS | +0.258 | Multi-story homes tend to be priced slightly higher, though the relationship is moderate. |
| YEAR_RENOVATED | +0.126 | Recent renovation has a mild positive effect on price. |

---

### Features Removed from the Model

The following features were excluded from training based on their low correlation with price or their redundancy with engineered features.

| Feature | Correlation | Reason for Removal |
| --- | --- | --- |
| CONDITION | 0.036 | Near-zero correlation. The condition rating as defined in this dataset does not differentiate price tiers in a statistically meaningful way. |
| YEAR_BUILT | 0.054 | Weak correlation. A home's construction year alone does not determine its value. A 1950 home and a 2010 home can sell for the same price. This is superseded by the `effective_age` feature. |
| YEAR_RENOVATED | 0.126 | Replaced by the engineered `effective_age` feature, which accounts for both construction year and renovation year in a single value. |
| SQFT_LOT | 0.090 | Lot size has almost no impact on price in King County. The most expensive homes are on small urban lots, while large lots are typically found in cheaper rural areas. Buyers pay for interior space, not yard size. |
| SQFT_LOT15 | 0.082 | Same reasoning as SQFT_LOT. Neighboring lot sizes do not predict price. |
| LAT / LONG | — | Raw coordinates were replaced by the `distance_to_seattle_km` engineered feature, which is more interpretable and less redundant. |
| DATE | — | The listing date of a sale is not a predictive factor for home price. |

---

## 2. Exploratory Data Analysis

### 2.1 Dataset Overview

The source dataset `kc_house_data.csv` contains **21,613 residential home sale records** from King County, Washington, covering transactions between 2014 and 2015. The raw data has 21 columns. After cleaning in dbt and feature selection in the notebook, the final model is trained on 13 features.

```
Total Records  : 21,613
Original Columns: 21
Final Features : 13
Target Variable: PRICE (integer, USD)
```

No records were removed during cleaning. All missing or invalid values were imputed using statistical defaults to preserve the full dataset size.

---

### 2.2 Correlation Heatmap

A Seaborn heatmap was generated to visualize pairwise correlations across all numeric features. This helped identify:

- **Multicollinearity**: `SQFT_LIVING`, `SQFT_ABOVE`, and `SQFT_LIVING15` are all highly correlated with each other because they all measure variations of dwelling size. This is expected and was handled through deliberate feature selection rather than dimensionality reduction.
- **Weak predictors**: `CONDITION` and `YEAR_BUILT` showed near-zero correlation with `PRICE`, confirming their removal before training.
- **Grade and size relationship**: `GRADE` and `SQFT_LIVING` are correlated with each other, indicating that larger homes tend to receive higher construction quality ratings.

---

### 2.3 Feature Engineering

Two new features were created to improve model interpretability and predictive power.

#### Distance to Downtown Seattle

Raw latitude and longitude values were replaced with a single distance metric calculated using the **Haversine formula**. This measures the straight-line distance in kilometers from each property to downtown Seattle (47.6062° N, 122.3321° W).

```python
import numpy as np

seattle_lat  = 47.6062
seattle_long = -122.3321

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df['distance_to_seattle_km'] = haversine_distance(
    df['LAT'], df['LONG'], seattle_lat, seattle_long
)
```

The resulting feature has a correlation of **-0.287** with price. The negative sign confirms the expected pattern: as distance from the city center increases, home prices decrease. This is consistent with urban real estate pricing theory, where proximity to employment centers and amenities drives demand.

Using this single feature instead of raw coordinates also eliminates the redundancy between `LAT` and `LONG` and makes the geographic signal easier to interpret.

#### Effective Age (computed in dbt)

Rather than using raw `YEAR_BUILT` and `YEAR_RENOVATED` values, the dbt staging model computes two derived features that more accurately represent how "current" a property is.

**Effective Age** is the number of years since the property was either renovated or originally built, whichever is more recent. Properties that have never been renovated use their construction year.

```sql
(2015 - COALESCE(NULLIF(yr_renovated, 0), yr_built)) AS effective_age
```

**Effective Age Status** groups properties into four categorical tiers based on their effective age:

```sql
CASE
    WHEN effective_age >= 51 THEN 4  -- Old       (50 or more years)
    WHEN effective_age >= 16 THEN 3  -- Mature    (16 to 50 years)
    WHEN effective_age >= 6  THEN 2  -- Recent    (6 to 15 years)
    WHEN effective_age >= 0  THEN 1  -- New       (0 to 5 years)
END AS effective_age_status
```

#### Zipcode as Neighborhood Price Proxy (computed in dbt)

Raw zipcode integers carry no ordinal meaning on their own. A zipcode of 98004 is not inherently "greater than" 98001. To make this feature useful, each zipcode was replaced with the **average sale price of all homes in that same zipcode**, computed using a SQL window function.

```sql
AVG(price) OVER (PARTITION BY zipcode) AS zipcode
```

This transforms a nominal identifier into a continuous, price-informative neighborhood signal. The result is a correlation of **+0.638** with price — making it the second strongest predictor in the model.

---

### 2.4 Data Cleaning Strategy (dbt)

All null handling and data validation is enforced in the dbt staging layer before data is passed to the notebook. The table below describes how each column was treated.

| Column | Null Handling | Validation Rule |
| --- | --- | --- |
| price | Imputed with median | — |
| bedrooms | Imputed with mode | — |
| bathrooms | Defaulted to 0 | — |
| sqft_living | Imputed with mode | — |
| sqft_lot | Imputed with mode | — |
| floors | Imputed with mode | — |
| waterfront | Binarized: any value greater than 0 becomes 1, otherwise 0 | — |
| view | Imputed with median | — |
| condition | Defaulted to 3 (mid-range) | Must be between 1 and 5 |
| grade | Defaulted to 5 (average) | Must be between 1 and 10 |
| sqft_above | Imputed with median | — |
| sqft_basement | Imputed with median | — |
| yr_built | Imputed with median | Must be between 1900 and 2015 |
| yr_renovated | Defaulted to 0 (not renovated) | — |
| lat, long | Imputed with median | — |
| sqft_living15 | Imputed with median | — |
| sqft_lot15 | Imputed with median | — |

---

## 3. Data Pipeline and Architecture

The pipeline follows a linear flow from raw data ingestion to model output.

```
kc_house_data.csv
        |
        v
Snowflake — RAW schema (housing_raw table)
        |
        v
dbt — housing_staging.sql
  - Null imputation
  - Data validation
  - Feature engineering (effective_age, zipcode encoding)
        |
        v
Snowflake — STAGING schema (housing_staging table)
        |
        v
snowflake_info.py — fetches cleaned data into a Pandas DataFrame
        |
        v
model.ipynb
  - EDA and correlation analysis
  - Additional feature engineering (distance_to_seattle_km)
  - Feature selection (drop low-correlation columns)
  - Model training and evaluation
        |
        v
Model Results (R2, RMSE per model)
```

### Snowflake Resources

| Resource | Value |
| --- | --- |
| Account Region | ap-southeast-7.aws |
| Database | housing_db |
| Raw Schema | raw |
| Staging Schema | staging |
| Warehouse | housing_wh |
| Role | data_engineer |

### dbt Models

| Layer | Model Name | Output |
| --- | --- | --- |
| Source | housing_raw | Raw table in the `raw` schema |
| Staging | housing_staging | Cleaned and engineered table in the `staging` schema |

---

## 4. Project Structure

```
boston_housing_prediction/
|
|-- kc_house_data.csv          # Raw source dataset (King County, 21,613 records)
|-- model.ipynb                # Notebook for EDA, feature selection, and model training
|-- snowflake_info.py          # Connects to Snowflake and loads staging data into Pandas
|-- .env                       # Snowflake credentials (excluded from version control)
|-- .gitignore
|
└── dbt_housing/               # dbt project directory
    |-- dbt_project.yml        # Project settings and model materialization config
    |-- profiles.yml           # Snowflake connection profile
    |
    └── models/
        └── staging/
            |-- src_sources.yml       # Declares the housing_raw source table
            └── housing_staging.sql   # Cleaning, validation, and feature engineering logic
```

---

## 5. Setup and Installation

### Prerequisites

- Python 3.9 or higher
- A Snowflake account with the `data_engineer` role and access to `housing_db`
- dbt Core with the Snowflake adapter installed

### Step 1 — Clone the Repository

```bash
git clone <repository-url>
cd boston_housing_prediction
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Python Dependencies

```bash
pip install snowflake-connector-python[pandas] pandas scikit-learn seaborn matplotlib numpy python-dotenv
```

### Step 4 — Install dbt

```bash
pip install dbt-snowflake
```

---

## 6. Configuration

### Environment Variables

Create a `.env` file in the project root with your Snowflake credentials:

```env
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_WAREHOUSE=housing_wh
SNOWFLAKE_DATABASE=housing_db
SNOWFLAKE_SCHEMA=staging
```

> **Security Note:** Never commit the `.env` file or `profiles.yml` containing real credentials to version control. Both files are listed in `.gitignore` by default.

### dbt Profile

Update `dbt_housing/profiles.yml` with your Snowflake connection details:

```yaml
dbt_housing:
  outputs:
    dev:
      type: snowflake
      account: <your_account_identifier>
      user: <your_username>
      password: <your_password>
      role: data_engineer
      database: housing_db
      schema: raw
      warehouse: housing_wh
      threads: 1
  target: dev
```

---

## 7. Running the Project

Follow these steps in order. Each step depends on the one before it.

### Step 1 — Load Raw Data into Snowflake

Upload `kc_house_data.csv` into the `housing_db.raw.housing_raw` table in Snowflake. You can do this using the Snowflake web UI (Snowsight), SnowSQL, or a bulk load stage.

### Step 2 — Run dbt to Clean and Transform the Data

Navigate into the dbt project directory and run the staging model:

```bash
cd dbt_housing

# Verify that dbt can connect to Snowflake
dbt debug

# Run the staging transformation
dbt run --select staging.housing_staging

# Optional: run data quality tests
dbt test
```

After this step, the cleaned and feature-engineered data will be available in `housing_db.staging.housing_staging`.

### Step 3 — Run the Notebook

Return to the project root and open the notebook:

```bash
cd ..
jupyter notebook model.ipynb
```

Run all cells from top to bottom. The notebook will:

1. Connect to Snowflake via `snowflake_info.py` and load the staging table into a Pandas DataFrame
2. Generate a correlation heatmap across all features
3. Print the correlation of each feature with `PRICE`
4. Compute the `distance_to_seattle_km` feature using the Haversine formula
5. Drop low-signal features based on correlation analysis
6. Train Linear Regression, Random Forest, and Decision Tree models
7. Print the R2 score and RMSE for each model

---

## 8. Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Cloud Data Warehouse | Snowflake | Stores raw and staged data; executes SQL transformations |
| Data Transformation | dbt Core (dbt-snowflake adapter) | Enforces data cleaning, null imputation, and feature engineering in SQL |
| Python-Snowflake Bridge | snowflake-connector-python | Fetches query results from Snowflake into a Pandas DataFrame |
| Analysis and Modeling | Jupyter Notebook | Interactive environment for EDA, feature selection, and model evaluation |
| Machine Learning | scikit-learn | Provides regression models, train-test split, scaling, and evaluation metrics |
| Visualization | Seaborn, Matplotlib | Generates the correlation heatmap and other exploratory plots |
| Numerical Computing | NumPy, Pandas | Data manipulation, Haversine distance computation, and DataFrame operations |
| Credential Management | python-dotenv | Loads Snowflake credentials from the `.env` file at runtime |

---

*Dataset: King County House Sales — sourced from Kaggle. Region: King County, Washington, USA. Records: 21,613. Period: May 2014 to May 2015.*
