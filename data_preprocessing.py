import pandas as pd

def load_and_preprocess_data():

    listings = pd.read_csv("/Users/talhamurtaza/Downloads/AI/listings.csv")
    calendar = pd.read_csv("/Users/talhamurtaza/Downloads/AI/calendar.csv")
    reviews = pd.read_csv("/Users/talhamurtaza/Downloads/AI/reviews.csv")

    if "id" in listings.columns:
        listings = listings.rename(columns={"id": "listing_id"})

    listings["price"] = listings["price"].replace('[\$,]', '', regex=True).astype(float)

    calendar["available"] = calendar["available"].map({'t': 1, 'f': 0})

    calendar_agg = calendar.groupby("listing_id")["available"].mean().reset_index()
    calendar_agg.rename(columns={"available": "availability_rate"}, inplace=True)

    reviews_agg = reviews.groupby("listing_id").size().reset_index(name="review_count")

    df = listings.merge(calendar_agg, on="listing_id", how="left")
    df = df.merge(reviews_agg, on="listing_id", how="left")

    df["availability_rate"].fillna(0, inplace=True)
    df["review_count"].fillna(0, inplace=True)

    features = ["bedrooms", "bathrooms", "availability_rate", "review_count"]

    df = df.dropna(subset=features + ["price"])

    X = df[features]
    y = df["price"]

    return X, y