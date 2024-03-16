# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the image
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Force reinstall PyMuPDF to ensure it's properly installed
RUN pip install --no-cache-dir --force-reinstall PyMuPDF

# Create the static directory if it doesn't exist
RUN mkdir -p /app/static

# Copy the rest of your application
COPY . /app/

# Create a non-root user and change ownership
RUN useradd -m myuser && chown -R myuser /app
USER myuser

# Expose the port Streamlit will run on
EXPOSE 8503

# Run your Streamlit application
CMD ["streamlit", "run", "main.py", "--server.port=8503", "--server.address=0.0.0.0"]
