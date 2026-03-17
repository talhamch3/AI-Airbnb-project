import pandas as pd

def load_and_preprocess_data():

    # 1️⃣ Load datasets
    listings = pd.read_csv("/Users/talhamurtaza/Downloads/AI/listings.csv")
    calendar = pd.read_csv("/Users/talhamurtaza/Downloads/AI/calendar.csv")
    reviews = pd.read_csv("/Users/talhamurtaza/Downloads/AI/reviews.csv")

    # 2️⃣ Ensure listing ID column is consistent
    if "id" in listings.columns:
        listings.rename(columns={"id": "listing_id"}, inplace=True)

    # 3️⃣ Clean price column (remove $ and ,)
    listings["price"] = listings["price"].replace(r"[\$,]", "", regex=True).astype(float)

    # 4️⃣ Convert availability (t/f) → (1/0)
    calendar["available"] = calendar["available"].map({"t": 1, "f": 0})

    # 5️⃣ Calculate average availability per listing
    availability = (
        calendar.groupby("listing_id")["available"]
        .mean()
        .reset_index()
        .rename(columns={"available": "availability_rate"})
    )

    # 6️⃣ Count number of reviews per listing
    review_counts = (
        reviews.groupby("listing_id")
        .size()
        .reset_index(name="review_count")
    )

    # 7️⃣ Merge datasets
    df = listings.merge(availability, on="listing_id", how="left")
    df = df.merge(review_counts, on="listing_id", how="left")

    # 8️⃣ Fill missing values
    df["availability_rate"].fillna(0, inplace=True)
    df["review_count"].fillna(0, inplace=True)

    # 9️⃣ Select model features
    features = ["bedrooms", "bathrooms", "availability_rate", "review_count"]

    # Remove rows with missing values
    df = df.dropna(subset=features + ["price"])

    # 🔟 Split features and target
    X = df[features]
    y = df["price"]

    return X, y