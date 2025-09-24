from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import subprocess
import os

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_python_libs():
    """ทดสอบว่า libraries จำเป็นติดตั้งครบหรือไม่"""
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
        
        import numpy as np
        print(f"Numpy version: {np.__version__}")
        
        print("All required libraries are available")
        return True
    except ImportError as e:
        print(f"Missing library: {e}")
        return False

def check_traffic_files():
    """ตรวจสอบไฟล์ traffic monitoring"""
    dashboard_path = "/opt/airflow/dashboard"
    
    if os.path.exists(dashboard_path):
        print(f"Dashboard directory exists: {dashboard_path}")
        
        # ตรวจสอบไฟล์สำคัญ
        stream_file = f"{dashboard_path}/bmatraffic_yolo_pipeline/src/stream_to_counts.py"
        config_file = f"{dashboard_path}/bmatraffic_yolo_pipeline/config/cameras.json"
        
        print(f"Stream file exists: {os.path.exists(stream_file)}")
        print(f"Config file exists: {os.path.exists(config_file)}")
        
        if os.path.exists(stream_file) and os.path.exists(config_file):
            return True
    else:
        print(f"Dashboard directory not found: {dashboard_path}")
    
    return False

def start_traffic_monitoring():
    """เริ่มต้น traffic monitoring (simplified version)"""
    try:
        # สร้างโฟลเดอร์ output
        output_dir = "/opt/airflow/data"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/snapshots", exist_ok=True)
        
        print(f"Created output directories: {output_dir}")
        
        # ในขั้นต้น เพียงแค่สร้างไฟล์ทดสอบ
        test_file = f"{output_dir}/airflow_test.txt"
        with open(test_file, 'w') as f:
            f.write(f"Airflow test at {datetime.now()}\n")
        
        print(f"Created test file: {test_file}")
        return True
        
    except Exception as e:
        print(f"Error in start_traffic_monitoring: {e}")
        return False

# สร้าง DAG
dag = DAG(
    'traffic_monitoring_test',
    default_args=default_args,
    description='Test Traffic Monitoring Setup',
    schedule_interval=timedelta(minutes=30),  # รันทุก 30 นาที
    catchup=False,
    max_active_runs=1,
)

# Task 1: ทดสอบ Python libraries
test_libs_task = PythonOperator(
    task_id='test_python_libraries',
    python_callable=test_python_libs,
    dag=dag,
)

# Task 2: ตรวจสอบไฟล์
check_files_task = PythonOperator(
    task_id='check_traffic_files',
    python_callable=check_traffic_files,
    dag=dag,
)

# Task 3: ทดสอบการสร้างไฟล์
test_output_task = PythonOperator(
    task_id='test_output_creation',
    python_callable=start_traffic_monitoring,
    dag=dag,
)

# กำหนดลำดับการทำงาน
test_libs_task >> check_files_task >> test_output_task