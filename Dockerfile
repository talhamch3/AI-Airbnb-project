# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY  . .

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn joblib numpy pandas

# Expose port
EXPOSE 8000

# Run FastAPI
ENTRYPOINT ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]