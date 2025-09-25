#!/bin/bash

# Simple startup script for Traffic Monitoring Airflow
echo "🚦 Starting Traffic Monitoring System..."

# Set environment variables
export AIRFLOW_HOME=/opt/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# Initialize database if needed
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "📊 Initializing Airflow database..."
    airflow db init
    
    echo "👤 Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
fi

# Start services
echo "🚀 Starting Airflow services..."
airflow webserver --daemon &
airflow scheduler &

# Keep container running
wait
