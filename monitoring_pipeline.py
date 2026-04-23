import pandas as pd
import mlflow
import mlflow.pyfunc
import psycopg2
import os
import numpy as np
from datetime import datetime


# =========================================================
# 🔌 MLflow setup
# Connect to the MLflow tracking server. The URI is read from
# the environment variable MLFLOW_TRACKING_URI so it works in
# both local and Docker environments without changing the code.
# =========================================================
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Airbnb Price Prediction")


# =========================================================
# 📦 1. LOAD MODEL
# Pulls the latest registered version of AirbnbPriceModel
# from the MLflow Model Registry. Using mlflow.pyfunc means
# we call .predict() the same way regardless of the underlying
# framework (XGBoost, sklearn, etc.).
# =========================================================
def load_model():
    return mlflow.pyfunc.load_model("models:/AirbnbPriceModel/latest")


# =========================================================
# 🌦️ 2. SEASON MAPPING HELPER
# Maps a month number (1–12) to one of four season strings.
# Used when adding a "season" column to the calendar DataFrame.
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
# 🗓️ 3. LOAD AND CLEAN CALENDAR  ← NEW HELPER
# Extracted into its own function so we load and clean the
# calendar exactly once and reuse the result everywhere.
# Previously the same read + clean block was duplicated inside
# get_batch() and compute_seasonal_metrics(), wasting I/O and
# risking inconsistency if one copy was edited without the other.
# =========================================================
def load_calendar():
    calendar = pd.read_csv("data/calendar.csv")

    # Convert availability flag from string to integer (t→1, f→0)
    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})

    # Parse dates and derive month + season columns
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["month"] = calendar["date"].dt.month
    calendar["season"] = calendar["month"].apply(get_season)

    return calendar


# =========================================================
# 📊 4. GENERATE BATCH
# Loads the three source files, engineers features, and returns
# a random 100-row sample that represents "new incoming data".
#
# Changes from original:
#   - Calendar is accepted as a parameter (loaded once outside)
#     instead of being re-read from disk here.
#   - The dead seasonal computation that was done here but never
#     used has been removed.
# =========================================================
def get_batch(calendar):

    # --- Load source files ---
    listings = pd.read_csv("data/listings.csv")
    reviews  = pd.read_csv("data/reviews.csv")
    # calendar is passed in — already cleaned by load_calendar()

    # Standardise the listing ID column name
    if "id" in listings.columns:
        listings.rename(columns={"id": "listing_id"}, inplace=True)

    # --- Clean listings ---
    # Price arrives as a string like "$1,200.00"; strip symbols and cast to float
    listings["price"] = (
        listings["price"]
        .replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

    # --- Availability rate per listing ---
    # Average of the 0/1 available flag over the full calendar period
    availability = (
        calendar
        .groupby("listing_id")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "availability_rate"})
    )

    # --- Review count per listing ---
    review_counts = (
        reviews
        .groupby("listing_id")
        .size()
        .reset_index(name="review_count")
    )

    # --- Merge all features onto listings ---
    df = listings.merge(availability,   on="listing_id", how="left")
    df = df.merge(review_counts,        on="listing_id", how="left")

    # Fill missing aggregates with 0 (listings with no calendar/review data)
    df["availability_rate"] = df["availability_rate"].fillna(0)
    df["review_count"]      = df["review_count"].fillna(0)

    # Drop rows where model features or target are missing
    df = df.dropna(subset=["bedrooms", "bathrooms", "price"])

    # --- Sample 100 rows to simulate a batch of new predictions ---
    # NOTE: random_state=None means each run draws a different sample,
    # which is intentional for monitoring variety. If you want fully
    # reproducible runs (e.g. for debugging), set random_state=42.
    batch = df.sample(n=100, random_state=None).copy()

    return batch


# =========================================================
# 🧠 5. COMPUTE RMSE
# Runs inference on the batch and computes Root Mean Squared
# Error against the actual listing prices.
# A rising RMSE over time signals model degradation.
# =========================================================
def compute_rmse(batch, model):

    # These must match the features used during training exactly.
    # If the model is retrained with different features, update this list.
    features = ["bedrooms", "bathrooms", "availability_rate", "review_count"]

    X     = batch[features]
    y     = batch["price"]
    preds = model.predict(X)

    rmse = np.sqrt(((y - preds) ** 2).mean())

    return rmse


# =========================================================
# 🌦️ 6. SEASONAL MONITORING
# Computes average availability broken down by season,
# restricted to the 100 listings in the current batch.
# This tells us whether the listings we just predicted on
# are more available in certain seasons — useful context
# for interpreting price prediction accuracy.
#
# Changes from original:
#   - Calendar is passed in instead of being re-read from disk.
#   - No more duplicate load/clean block.
# =========================================================
def compute_seasonal_metrics(batch, calendar):

    # Filter calendar to only the listings present in this batch.
    # We deliberately scope to the batch so seasonal metrics are
    # consistent with the RMSE (same 100 listings, same run).
    batch_calendar = calendar[calendar["listing_id"].isin(batch["listing_id"])]

    # Average availability per season across batch listings
    seasonal = (
        batch_calendar
        .groupby("season")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "avg_availability"})
    )

    # Guarantee all four seasons appear in the output, even if
    # none of the batch listings have calendar data for a season
    all_seasons = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal = (
        pd.DataFrame({"season": all_seasons})
        .merge(seasonal, on="season", how="left")
    )

    # Fill seasons with no data as 0 availability
    seasonal["avg_availability"] = seasonal["avg_availability"].fillna(0)

    return seasonal


# =========================================================
# 🏠 7. LISTING-LEVEL METRICS
# Extracts per-listing features from the batch for storage
# in Postgres, where they can be queried in Grafana.
# Deduplication guards against the unlikely case of duplicate
# listing_ids in the sample.
# =========================================================
def compute_listing_metrics(batch):

    df = batch[["listing_id", "bedrooms", "availability_rate"]].copy()
    df = df.drop_duplicates(subset=["listing_id"])

    return df


# =========================================================
# 💾 8. STORE IN POSTGRES
# Persists all three metric types in a single transaction.
# A single run_time timestamp is captured once and used for
# both the RMSE row and all seasonal rows, so every record
# from the same run shares an identical timestamp and can be
# grouped reliably in Grafana queries.
#
# Changes from original:
#   - run_time captured once at the top (was called multiple
#     times, producing slightly different timestamps per run).
#   - Connection wrapped in try/finally so it always closes
#     cleanly even if an insert raises an exception mid-way.
# =========================================================
def store_in_postgres(rmse, seasonal_df, listing_df):

    # Capture one timestamp for the entire run so all records
    # from this monitoring cycle share the same timestamp
    run_time = datetime.now()

    conn = psycopg2.connect(
        host="postgres",
        database="airbnb",
        user="admin",
        password="admin"
    )

    try:
        cur = conn.cursor()

        # --------------------------------------------------
        # TABLE 1: model_monitoring
        # One row per pipeline run — tracks RMSE over time.
        # Grafana plots this as a time-series to detect drift.
        # --------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_monitoring (
                id        SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                rmse      FLOAT
            )
        """)

        # --------------------------------------------------
        # TABLE 2: seasonal_availability
        # Four rows per run (one per season).
        # Tracks whether availability patterns shift over time.
        # --------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS seasonal_availability (
                id               SERIAL PRIMARY KEY,
                season           TEXT,
                avg_availability FLOAT,
                timestamp        TIMESTAMP
            )
        """)

        # --------------------------------------------------
        # TABLE 3: listing_metrics
        # One row per unique listing — upserted so re-running
        # the pipeline updates existing listings rather than
        # creating duplicates.
        # --------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS listing_metrics (
                listing_id        BIGINT PRIMARY KEY,
                bedrooms          FLOAT,
                availability_rate FLOAT
            )
        """)

        # --------------------------------------------------
        # Insert RMSE for this run
        # --------------------------------------------------
        cur.execute("""
            INSERT INTO model_monitoring (timestamp, rmse)
            VALUES (%s, %s)
        """, (run_time, float(rmse)))

        # --------------------------------------------------
        # Insert seasonal availability (4 rows, same timestamp)
        # --------------------------------------------------
        for _, row in seasonal_df.iterrows():
            cur.execute("""
                INSERT INTO seasonal_availability (season, avg_availability, timestamp)
                VALUES (%s, %s, %s)
            """, (
                row["season"],
                float(row["avg_availability"]),
                run_time   # same timestamp as RMSE row
            ))

        # --------------------------------------------------
        # Upsert listing metrics
        # ON CONFLICT updates existing rows so we always have
        # the latest bedrooms/availability_rate per listing.
        # --------------------------------------------------
        for _, row in listing_df.iterrows():
            cur.execute("""
                INSERT INTO listing_metrics (listing_id, bedrooms, availability_rate)
                VALUES (%s, %s, %s)
                ON CONFLICT (listing_id) DO UPDATE
                    SET bedrooms          = EXCLUDED.bedrooms,
                        availability_rate = EXCLUDED.availability_rate
            """, (
                int(row["listing_id"]),
                float(row["bedrooms"]),
                float(row["availability_rate"])
            ))

        # Commit all inserts as a single transaction
        conn.commit()

    finally:
        # Always close the connection, even if an error occurred above
        cur.close()
        conn.close()


# =========================================================
# 🚀 MAIN PIPELINE
# Orchestrates each step in order, passing outputs from one
# step as inputs to the next. Calendar is loaded once here
# and shared across get_batch() and compute_seasonal_metrics()
# to avoid redundant disk reads.
# =========================================================
def run_monitoring():

    print("Loading model...")
    model = load_model()

    # Load and clean calendar once — reused by get_batch() and
    # compute_seasonal_metrics() so the CSV is only read once
    print("Loading calendar...")
    calendar = load_calendar()

    print("Generating batch...")
    batch = get_batch(calendar)

    print("Computing RMSE...")
    rmse = compute_rmse(batch, model)
    print(f"  RMSE: {rmse:.4f}")

    print("Computing seasonal metrics...")
    seasonal_df = compute_seasonal_metrics(batch, calendar)

    print("Computing listing metrics...")
    listing_df = compute_listing_metrics(batch)

    print("Saving to Postgres...")
    store_in_postgres(rmse, seasonal_df, listing_df)

    print("✅ Monitoring complete.")


# =========================================================
if __name__ == "__main__":
    run_monitoring()