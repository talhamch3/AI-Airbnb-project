import pandas as pd

# For Testing, We are going to check if all 3 files are correct and not empty.
def test_data_loading():
    print("\n🚀 Starting Data Loading Test...")
    
    listings = pd.read_csv("data/listings.csv")
    calendar = pd.read_csv("data/calendar.csv")
    reviews = pd.read_csv("data/reviews.csv")

    print(f"📈 Loaded listings.csv successfully. Shape: {listings.shape}")
    print(f"📅 Loaded calendar.csv successfully. Shape: {calendar.shape}")
    print(f"💬 Loaded reviews.csv successfully. Shape: {reviews.shape}")

    assert not listings.empty, "Listings dataframe is empty!"
    assert not calendar.empty, "Calendar dataframe is empty!"
    assert not reviews.empty, "Reviews dataframe is empty!"
    
    print("✅ Data Loading Test passed successfully!")


# Check if RMSE Calculation is working on point and the required stuff doesn't break (Needs the model to load)
def test_rmse_computation():
    print("\n🚀 Starting RMSE Computation Test...")
    from monitoring_pipeline import compute_rmse, load_model, load_calendar, get_batch

    print("🤖 Loading machine learning model...")
    model = load_model()
    
    print("📅 Loading calendar data & creating current batch...")
    calendar = load_calendar()
    batch = get_batch(calendar)

    print("📊 Computing RMSE tracking metrics...")
    rmse = compute_rmse(batch, model)
    print(f"🎯 Calculated RMSE output: {rmse}")

    assert rmse >= 0, f"RMSE cannot be negative! Got: {rmse}"
    print("✅ RMSE Computation Test passed successfully!")


# Check if Seasonal Metrics Structure is on Point
def test_seasonal_metrics():
    print("\n🚀 Starting Seasonal Metrics Test...")
    from monitoring_pipeline import compute_seasonal_metrics, load_calendar, get_batch

    calendar = load_calendar()
    batch = get_batch(calendar)

    print("🍁 Extracting seasonal performance breakdowns...")
    seasonal = compute_seasonal_metrics(batch, calendar)
    
    print(f"📊 Found seasons in data: {list(seasonal['season'].unique())}")
    print(f"📏 Total seasonal groups found: {len(seasonal)}")

    assert len(seasonal) == 4, f"Expected 4 seasons, but got {len(seasonal)}"
    assert set(seasonal["season"]) == {"Winter", "Spring", "Summer", "Autumn"}, "Missing or incorrect season names!"
    print("✅ Seasonal Metrics Test passed successfully!")


# Check if listing ID is unique
def test_listing_metrics():
    print("\n🚀 Starting Listing Metrics Test...")
    from monitoring_pipeline import compute_listing_metrics, load_calendar, get_batch

    calendar = load_calendar()
    batch = get_batch(calendar)

    print("🔍 Computing individual listing metrics...")
    listings = compute_listing_metrics(batch)
    
    total_rows = len(listings)
    unique_ids = listings["listing_id"].nunique()
    print(f"🆔 Total rows: {total_rows} | Unique Listing IDs: {unique_ids}")

    assert listings["listing_id"].is_unique, f"Duplicate listing IDs found! ({total_rows - unique_ids} duplicates)"
    print("✅ Listing Metrics Test passed successfully!")