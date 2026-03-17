# Dockerfile
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy the FastAPI application file into the container
COPY fastapi_app.py .

# Install required Python packages
RUN pip install --no-cache-dir mlflow joblib numpy scikit-learn fastapi uvicorn

# Optional: Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001

# Expose port 8000 inside the container
EXPOSE 8000

# Correct ENTRYPOINT: load the app from fastapi_app.py
ENTRYPOINT ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]