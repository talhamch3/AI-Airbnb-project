import pandas as pd
import mlflow.pyfunc

# 1️⃣ Model configuration
MODEL_NAME = "AirbnbPriceModel"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2️⃣ Load latest registered model from MLflow
model = mlflow.pyfunc.load_model("models:/AirbnbPriceModel@best")
# model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")


def predict_price(bedrooms, bathrooms, availability_rate, review_count):
    """
    Predict Airbnb listing price.

    Parameters
    ----------
    bedrooms : float
    bathrooms : float
    availability_rate : float
    review_count : int

    Returns
    -------
    float : predicted Airbnb price
    """

    # 3️⃣ Prepare input data
    input_data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "availability_rate": availability_rate,
        "review_count": review_count
    }])

    # 4️⃣ Generate prediction
    prediction = model.predict(input_data)

    return prediction[0]


# 5️⃣ Example test prediction
if __name__ == "__main__":

    predicted_price = predict_price(
        bedrooms=1,
        bathrooms=1,
        availability_rate=0.5,
        review_count=30
    )

    print("Predicted Airbnb Price:", predicted_price)