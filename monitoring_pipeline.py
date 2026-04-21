import pandas as pd
import mlflow
import mlflow.pyfunc
import psycopg2
import os
import numpy as np
from datetime import datetime


# =========================================================
# 🔌 MLflow setup (connect to MLflow container)
# =========================================================
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Airbnb Price Prediction")


# =========================================================
# 📦 1. Load latest model from MLflow registry
# =========================================================
def load_model():
    return mlflow.pyfunc.load_model("models:/AirbnbPriceModel/latest")


# =========================================================
# 🌦️ 2. Season mapping helper
# =========================================================
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
# 📊 3. GENERATE BATCH (THIS IS THE CORE CHANGE)
# Everything below will use THIS batch only
# =========================================================
def get_batch():

    # Load datasets
    listings = pd.read_csv("data/listings.csv")
    calendar = pd.read_csv("data/calendar.csv")
    reviews = pd.read_csv("data/reviews.csv")

    # Standardize ID column
    if "id" in listings.columns:
        listings.rename(columns={"id": "listing_id"}, inplace=True)

    # -------------------------
    # CLEAN + FEATURE ENGINEERING
    # -------------------------

    # Convert price from string → float
    listings["price"] = listings["price"].replace(r"[\$,]", "", regex=True).astype(float)

    # Convert availability to numeric
    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})

    # Availability rate per listing
    availability = (
        calendar.groupby("listing_id")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "availability_rate"})
    )

    # Review count per listing
    review_counts = (
        reviews.groupby("listing_id")
        .size()
        .reset_index(name="review_count")
    )

    # Merge everything
    df = listings.merge(availability, on="listing_id", how="left")
    df = df.merge(review_counts, on="listing_id", how="left")

    # Fill missing values
    df["availability_rate"] = df["availability_rate"].fillna(0)
    df["review_count"] = df["review_count"].fillna(0)

    # Add time-based features (for seasonal analysis)
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["month"] = calendar["date"].dt.month
    calendar["season"] = calendar["month"].apply(get_season)


    # Drop missing essential values
    df = df.dropna(subset=["bedrooms", "bathrooms", "price"])

    # -------------------------
    # SAMPLE BATCH (SIMULATES NEW DATA)
    # -------------------------
    batch = df.sample(n=100, random_state=None).copy()

    return batch


# =========================================================
# 🧠 4. MODEL MONITORING (RMSE)
# =========================================================
def compute_rmse(batch, model):

    # Features used in training
    features = ["bedrooms", "bathrooms", "availability_rate", "review_count"]

    X = batch[features]
    y = batch["price"]

    preds = model.predict(X)

    rmse = np.sqrt(((y - preds) ** 2).mean())

    return rmse


# =========================================================
# 🌦️ 5. SEASONAL MONITORING (FROM BATCH)
# =========================================================
def compute_seasonal_metrics(batch):

    # Reload calendar (full time-series data)
    calendar = pd.read_csv("data/calendar.csv")

    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["month"] = calendar["date"].dt.month
    calendar["season"] = calendar["month"].apply(get_season)

    # 🔥 KEY FIX: filter calendar to ONLY batch listings
    calendar = calendar[calendar["listing_id"].isin(batch["listing_id"])]

    # Compute seasonal availability properly
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
# 🏠 6. LISTING-LEVEL METRICS (FROM BATCH)
# =========================================================
def compute_listing_metrics(batch):

    # Only keep relevant columns
    df = batch[["listing_id", "bedrooms", "availability_rate"]].copy()

    # Remove duplicates (important)
    df = df.drop_duplicates(subset=["listing_id"])

    return df


# =========================================================
# 💾 7. STORE EVERYTHING IN POSTGRES
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
    # TABLE 1: RMSE tracking
    # ---------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_monitoring (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            rmse FLOAT
        )
    """)

    # ---------------------------
    # TABLE 2: Seasonal trends
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
    # TABLE 3: Listing-level data
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
    # Insert listing metrics (UPSERT)
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

    print("Generating batch...")
    batch = get_batch()

    print("Computing RMSE...")
    rmse = compute_rmse(batch, model)
    print("RMSE:", rmse)

    print("Computing seasonal metrics...")
    seasonal_df = compute_seasonal_metrics(batch)

    print("Computing listing metrics...")
    listing_df = compute_listing_metrics(batch)

    print("Saving to Postgres...")
    store_in_postgres(rmse, seasonal_df, listing_df)

    print("Monitoring complete.")


# =========================================================
if __name__ == "__main__":
    run_monitoring()