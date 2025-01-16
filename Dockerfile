# Use the official Python image
FROM python:3.9-slim

# Copy all folders
COPY app/ /app/
COPY data/ /data/
COPY models/ /models/
COPY setup.py setup.py
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Expose a port (for web applications or APIs)
EXPOSE 8089

# Define environment variable
ENV FLASK_APP=inference.py

# Set the command to run your script
CMD ["python", "app/inference.py"]
