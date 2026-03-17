import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from data_preprocessing import load_and_preprocess_data


# 1️⃣ Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Airbnb Price Prediction")


def train():

    # 2️⃣ Load and preprocess data
    X, y = load_and_preprocess_data()

    # 3️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4️⃣ Define models to compare (Experiment Tracking)
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    # 5️⃣ Train and log experiments
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name) as run:

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Evaluate model
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            # Log parameters and metrics
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("rmse", rmse)

            # Log model artifact
            mlflow.sklearn.log_model(model, "model")

            print(f"{model_name} RMSE: {rmse}")

            # 6️⃣ Register model in MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/model"

            mlflow.register_model(
                model_uri=model_uri,
                name="AirbnbPriceModel"
            )


if __name__ == "__main__":
    train()