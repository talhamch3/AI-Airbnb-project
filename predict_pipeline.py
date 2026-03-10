import mlflow.pyfunc
import pandas as pd

# Load the latest version of your registered model
MODEL_NAME = "AirbnbPriceModel"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")


def predict_price(bedrooms, bathrooms, availability_rate, review_count):
    """
    Predict Airbnb price for a single listing.

    Parameters:
        bedrooms (float)
        bathrooms (float)
        availability_rate (float, 0-1)
        review_count (int)

    Returns:
        float: predicted price
    """

    # Prepare input as a DataFrame
    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "availability_rate": availability_rate,
        "review_count": review_count
    }])

    # Predict
    prediction = model.predict(input_df)

    return prediction[0]


# Example usage
if __name__ == "__main__":
    example_price = predict_price(
        bedrooms=1,
        bathrooms=1,
        availability_rate=0.5,
        review_count=30
    )
    print("Predicted Airbnb Price:", example_price)