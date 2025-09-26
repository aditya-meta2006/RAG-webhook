# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (faster builds with caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Cloud Run will use
EXPOSE 8080

# Run the app with Gunicorn + Uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
