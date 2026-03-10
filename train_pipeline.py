import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from data_preprocessing import load_and_preprocess_data


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Airbnb Price Prediction")


def train():

    X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "GradientBoosting": GradientBoostingRegressor()
    }

    for name, model in models.items():

        with mlflow.start_run(run_name=name) as run:

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_param("model_name", name)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(model, "model")

            print(name, "RMSE:", rmse)

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"

            mlflow.register_model(
                model_uri=model_uri,
                name="AirbnbPriceModel"
            )


if __name__ == "__main__":
    train()