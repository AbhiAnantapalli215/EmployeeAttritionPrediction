# Use official lightweight Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run the app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

RUN python model_builder.py
