import pandas as pd
import mlflow
import mlflow.pyfunc
import psycopg2
import os
import numpy as np
from datetime import datetime


# -------------------------------
# MLflow setup (connect to server inside Docker)
# -------------------------------
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Airbnb Price Prediction")


# -------------------------------
# 1. Load latest model from MLflow
# -------------------------------
def load_model():
    return mlflow.pyfunc.load_model("models:/AirbnbPriceModel/latest")


# -------------------------------
# 2. Season mapping (for analytics only)
# -------------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# =========================================================
# 🧠 MODEL MONITORING (RMSE)
# =========================================================
def get_model_batch():

    # Load datasets
    listings = pd.read_csv("data/listings.csv")
    calendar = pd.read_csv("data/calendar.csv")
    reviews = pd.read_csv("data/reviews.csv")

    # Ensure consistent column name
    if "id" in listings.columns:
        listings.rename(columns={"id": "listing_id"}, inplace=True)

    # -----------------------------
    # Feature engineering (same as training)
    # -----------------------------

    # Clean price column
    listings["price"] = listings["price"].replace(r"[\$,]", "", regex=True).astype(float)

    # Convert availability to numeric
    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})

    # Compute availability rate per listing
    availability = (
        calendar.groupby("listing_id")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "availability_rate"})
    )

    # Count reviews per listing
    review_counts = (
        reviews.groupby("listing_id")
        .size()
        .reset_index(name="review_count")
    )

    # Merge features into one dataframe
    df = listings.merge(availability, on="listing_id", how="left")
    df = df.merge(review_counts, on="listing_id", how="left")

    # Fill missing values
    df["availability_rate"] = df["availability_rate"].fillna(0)
    df["review_count"] = df["review_count"].fillna(0)

    # Select model features
    features = ["bedrooms", "bathrooms", "availability_rate", "review_count"]

    df = df.dropna(subset=features + ["price"])

    X = df[features].copy()
    y = df["price"].copy()

    # Sample batch
    batch = X.sample(n=150, random_state=42)
    actual = y.loc[batch.index]

    return batch, actual


# =========================================================
# 📊 DATA MONITORING (SEASONAL)
# =========================================================
def get_seasonal_metrics():

    calendar = pd.read_csv("data/calendar.csv")

    # Convert availability
    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})

    # Extract time features
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["month"] = calendar["date"].dt.month

    # Map to season
    calendar["season"] = calendar["month"].apply(get_season)

    # Compute average availability per season
    seasonal = (
        calendar.groupby("season")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "avg_availability"})
    )

    # Ensure all seasons exist
    all_seasons = ["Winter", "Spring", "Summer", "Autumn"]

    seasonal = (
        pd.DataFrame({"season": all_seasons})
        .merge(seasonal, on="season", how="left")
    )

    seasonal["avg_availability"] = seasonal["avg_availability"].fillna(0)

    return seasonal


# =========================================================
# 🏠 NEW: LISTING-LEVEL MONITORING
# bedrooms + availability_rate (NO DUPLICATES)
# =========================================================
def get_listing_metrics():

    listings = pd.read_csv("data/listings.csv")
    calendar = pd.read_csv("data/calendar.csv")

    # Standardize column name
    if "id" in listings.columns:
        listings.rename(columns={"id": "listing_id"}, inplace=True)

    # Convert availability
    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})

    # Compute availability rate per listing
    availability = (
        calendar.groupby("listing_id")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "availability_rate"})
    )

    # Select ONLY needed columns from listings
    listing_info = listings[["listing_id", "bedrooms"]].copy()

    # Merge bedrooms + availability_rate
    df = listing_info.merge(availability, on="listing_id", how="inner")

    # Remove duplicates (important requirement)
    df = df.drop_duplicates(subset=["listing_id"])

    return df


# -------------------------------
# RMSE calculation
# -------------------------------
def compute_rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())


# =========================================================
# 💾 STORE EVERYTHING IN POSTGRES
# =========================================================
def store_in_postgres(rmse, seasonal_df, listing_df):

    conn = psycopg2.connect(
        host="postgres",
        database="airbnb",
        user="admin",
        password="admin"
    )

    cur = conn.cursor()

    # ---------------------------
    # 1. Model monitoring table
    # ---------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_monitoring (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            rmse FLOAT
        )
    """)

    # ---------------------------
    # 2. Seasonal monitoring table
    # ---------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS seasonal_availability (
            id SERIAL PRIMARY KEY,
            season TEXT,
            avg_availability FLOAT,
            timestamp TIMESTAMP
        )
    """)

    # ---------------------------
    # 3. Listing-level monitoring table
    # ---------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS listing_metrics (
            listing_id BIGINT PRIMARY KEY,
            bedrooms FLOAT,
            availability_rate FLOAT
        )
    """)

    # ---------------------------
    # Insert RMSE
    # ---------------------------
    cur.execute("""
        INSERT INTO model_monitoring (timestamp, rmse)
        VALUES (%s, %s)
    """, (datetime.now(), float(rmse)))

    # ---------------------------
    # Insert seasonal metrics
    # ---------------------------
    for _, row in seasonal_df.iterrows():
        cur.execute("""
            INSERT INTO seasonal_availability (season, avg_availability, timestamp)
            VALUES (%s, %s, %s)
        """, (
            row["season"],
            float(row["avg_availability"]),
            datetime.now()
        ))

    # ---------------------------
    # Insert listing metrics (NO DUPLICATES)
    # Uses UPSERT to avoid conflicts
    # ---------------------------
    for _, row in listing_df.iterrows():
        cur.execute("""
            INSERT INTO listing_metrics (listing_id, bedrooms, availability_rate)
            VALUES (%s, %s, %s)
            ON CONFLICT (listing_id) DO UPDATE
            SET bedrooms = EXCLUDED.bedrooms,
                availability_rate = EXCLUDED.availability_rate
        """, (
            int(row["listing_id"]),
            float(row["bedrooms"]),
            float(row["availability_rate"])
        ))

    conn.commit()
    cur.close()
    conn.close()


# =========================================================
# 🚀 MAIN PIPELINE
# =========================================================
def run_monitoring():

    print("Loading model...")
    model = load_model()

    # MODEL MONITORING
    print("Getting batch...")
    X_batch, y_actual = get_model_batch()

    print("Predicting...")
    preds = model.predict(X_batch)

    print("Computing RMSE...")
    rmse = compute_rmse(y_actual, preds)
    print("RMSE:", rmse)

    # DATA MONITORING
    print("Computing seasonal metrics...")
    seasonal_df = get_seasonal_metrics()

    print("Computing listing-level metrics...")
    listing_df = get_listing_metrics()

    # STORE EVERYTHING
    print("Saving to DB...")
    store_in_postgres(rmse, seasonal_df, listing_df)

    print("Monitoring complete.")


# -------------------------------
if __name__ == "__main__":
    run_monitoring()