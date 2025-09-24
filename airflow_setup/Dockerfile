FROM apache/airflow:2.7.3-python3.10

# Switch to root to install minimal system dependencies
USER root

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install Python dependencies in smaller chunks
RUN pip install --no-cache-dir apache-airflow-providers-postgres==5.7.1 psycopg2-binary
RUN pip install --no-cache-dir pandas numpy pytz
RUN pip install --no-cache-dir opencv-python-headless pillow requests beautifulsoup4
RUN pip install --no-cache-dir ultralytics streamlit

# Create necessary directories
RUN mkdir -p /opt/airflow/data/snapshots /opt/airflow/logs