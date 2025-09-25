FROM apache/airflow:2.7.3-python3.10

# Switch to root to install minimal system dependencies  
USER root

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install Python dependencies (lightweight versions)
RUN pip install --no-cache-dir \
    apache-airflow-providers-postgres==5.7.1 \
    psycopg2-binary \
    pandas \
    numpy \
    pytz \
    opencv-python-headless \
    pillow \
    requests \
    beautifulsoup4 \
    ultralytics \
    streamlit

# Create necessary directories and set permissions
USER root
RUN mkdir -p /opt/airflow/data/snapshots /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins && \
    chown -R airflow:0 /opt/airflow

USER airflow