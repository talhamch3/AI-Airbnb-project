# Dockerfile
FROM python:3.9-slim    
WORKDIR /app
COPY fastapi_app.py .
# COPY scaler.joblib .   # remove if not used
RUN pip install mlflow joblib numpy scikit-learn fastapi uvicorn
# export mlflow tracking uri and registry uri as environment variables
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001
EXPOSE 8000
ENTRYPOINT ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]