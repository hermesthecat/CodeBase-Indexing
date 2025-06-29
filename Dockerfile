# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
# --no-cache-dir: Disables the cache, which is useful in image builds to keep the layer size down.
# -r requirements.txt: Specifies the file from which to install packages.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Make port 8000 available to the world outside this container
# This is the port the API server will run on, as defined in the .env file
EXPOSE 8000

# Define the command to run the application
# This will be `python qwen3-api.py`. The host and port are configured
# inside the script to be read from environment variables.
CMD ["python", "qwen3-api.py"] 