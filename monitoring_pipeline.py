import pandas as pd
import mlflow
import mlflow.pyfunc
import psycopg2
import os
import numpy as np
from datetime import datetime


# -------------------------------
# MLflow setup
# -------------------------------
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Airbnb Price Prediction")


# -------------------------------
# 1. Load Model
# -------------------------------
def load_model():
    return mlflow.pyfunc.load_model("models:/AirbnbPriceModel/latest")


# -------------------------------
# 2. Season logic (ONLY for analytics)
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
# 🧠 MODEL MONITORING PIPELINE (RMSE ONLY - CLEAN)
# =========================================================
def get_model_batch():

    listings = pd.read_csv("data/listings.csv")
    calendar = pd.read_csv("data/calendar.csv")
    reviews = pd.read_csv("data/reviews.csv")

    if "id" in listings.columns:
        listings.rename(columns={"id": "listing_id"}, inplace=True)

    # -----------------------------
    # BUILD TRAINING FEATURES EXACTLY
    # -----------------------------

    listings["price"] = listings["price"].replace(r"[\$,]", "", regex=True).astype(float)

    # availability_rate (IMPORTANT FEATURE)
    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})

    availability = (
        calendar.groupby("listing_id")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "availability_rate"})
    )

    # review_count (IMPORTANT FEATURE)
    review_counts = (
        reviews.groupby("listing_id")
        .size()
        .reset_index(name="review_count")
    )

    # merge features
    df = listings.merge(availability, on="listing_id", how="left")
    df = df.merge(review_counts, on="listing_id", how="left")

    df["availability_rate"] = df["availability_rate"].fillna(0)
    df["review_count"] = df["review_count"].fillna(0)

    # -----------------------------
    # MODEL FEATURES (MUST MATCH TRAINING)
    # -----------------------------
    features = ["bedrooms", "bathrooms", "availability_rate", "review_count"]

    df = df.dropna(subset=features + ["price"])

    X = df[features].copy()
    y = df["price"].copy()

    batch = X.sample(n=50, random_state=42).copy()
    actual = y.loc[batch.index]

    return batch, actual


# =========================================================
# 📊 DATA MONITORING PIPELINE (SEPARATE - SAFE)
# =========================================================
def get_seasonal_metrics():

    calendar = pd.read_csv("data/calendar.csv")

    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["month"] = calendar["date"].dt.month
    calendar["season"] = calendar["month"].apply(get_season)

    seasonal = (
        calendar
        .groupby("season")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "avg_availability"})
    )

    # ensure all seasons exist
    all_seasons = ["Winter", "Spring", "Summer", "Autumn"]

    seasonal = (
        pd.DataFrame({"season": all_seasons})
        .merge(seasonal, on="season", how="left")
    )

    seasonal["avg_availability"] = seasonal["avg_availability"].fillna(0)

    return seasonal


# -------------------------------
# 3. RMSE calculation
# -------------------------------
def compute_rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())


# -------------------------------
# 4. Store in Postgres
# -------------------------------
def store_in_postgres(rmse, seasonal_df):

    conn = psycopg2.connect(
        host="postgres",
        database="airbnb",
        user="admin",
        password="admin"
    )

    cur = conn.cursor()

    # RMSE table (model health)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_monitoring (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            rmse FLOAT
        )
    """)

    # Seasonal table (data health)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS seasonal_availability (
            id SERIAL PRIMARY KEY,
            season TEXT,
            avg_availability FLOAT,
            timestamp TIMESTAMP
        )
    """)

    # insert RMSE
    cur.execute("""
        INSERT INTO model_monitoring (timestamp, rmse)
        VALUES (%s, %s)
    """, (datetime.now(), float(rmse)))

    # insert seasonal stats
    for _, row in seasonal_df.iterrows():
        cur.execute("""
            INSERT INTO seasonal_availability (season, avg_availability, timestamp)
            VALUES (%s, %s, %s)
        """, (
            row["season"],
            float(row["avg_availability"]),
            datetime.now()
        ))

    conn.commit()
    cur.close()
    conn.close()


# -------------------------------
# 5. Main pipeline
# -------------------------------
def run_monitoring():

    print("Loading model...")
    model = load_model()

    # -----------------------
    # MODEL MONITORING ONLY
    # -----------------------
    print("Getting model batch...")
    X_batch, y_actual = get_model_batch()

    print("Predicting...")
    preds = model.predict(X_batch)

    print("Computing RMSE...")
    rmse = compute_rmse(y_actual, preds)
    print("RMSE:", rmse)

    # -----------------------
    # DATA MONITORING ONLY
    # -----------------------
    print("Computing seasonal metrics...")
    seasonal_df = get_seasonal_metrics()

    print(seasonal_df)

    # -----------------------
    # STORE BOTH SEPARATELY
    # -----------------------
    print("Saving to DB...")
    store_in_postgres(rmse, seasonal_df)

    print("Monitoring complete.")


# -------------------------------
if __name__ == "__main__":
    run_monitoring()