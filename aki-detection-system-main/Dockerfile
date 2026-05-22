# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to ensure output is sent straight to terminal
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main execution script
COPY main.py .

# Copy the source code directory (data.py, inference.py, network.py)
# and the trained model artifacts (.pkl files)
COPY src/ ./src/

# Create the data directory where the assessment system will mount history.csv
RUN mkdir -p /data

# The command to run your application
# The assessment system will provide MLLP_ADDRESS and PAGER_ADDRESS [cite: 53]
CMD ["python", "main.py"]