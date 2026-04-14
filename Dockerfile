FROM python:3.9-slim

WORKDIR /app

COPY . .

# Install ALL dependencies
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn mlflow fastapi uvicorn joblib psycopg2-binary

# Expose API port
EXPOSE 8000

# Default command = run FastAPI
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]