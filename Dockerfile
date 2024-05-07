# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /app
COPY . /requirements.txt /code/

COPY /app/photosynthesis_model_config.yaml /code/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY . /code/
# Expose port 80 to the outside world
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run FastAPI when the container launches
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
