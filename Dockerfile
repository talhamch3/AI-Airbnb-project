# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy your FastAPI app
COPY fastapi_app.py .

# Install required Python packages
RUN pip install --no-cache-dir mlflow joblib numpy scikit-learn fastapi uvicorn requests

# Set environment variables for MLflow
# Replace 10.128.0.5 with your VM internal IP
ENV MLFLOW_TRACKING_URI=http://10.128.0.2:5001
ENV MLFLOW_REGISTRY_URI=http://10.128.0.2:5001

# Expose port 8000 inside container
EXPOSE 8000

# Start FastAPI with uvicorn
ENTRYPOINT ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]