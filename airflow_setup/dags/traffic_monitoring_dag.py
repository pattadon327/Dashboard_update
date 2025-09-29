from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
import os
import subprocess
import time

default_args = {
    'owner': 'traffic_monitoring',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_traffic_process():
    """ตรวจสอบว่า traffic monitoring process ยังทำงานอยู่หรือไม่"""
    try:
        # ตรวจสอบ process ที่รัน bma_lastest.py
        result = subprocess.run(['pgrep', '-f', 'bma_lastest.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Traffic monitoring process is running: PID {result.stdout.strip()}")
            return True
        else:
            print("No traffic monitoring process found")
            return False
    except Exception as e:
        print(f"Error checking process: {e}")
        return False

def start_traffic_monitoring():
    """เริ่มต้น traffic monitoring process (lightweight version)"""
    try:
        dashboard_path = "/opt/airflow/dashboard/bmatraffic_yolo_pipeline/src"
        cameras_config = "/opt/airflow/dashboard/bmatraffic_yolo_pipeline/config/cameras.json"
        output_dir = "/opt/airflow/data"
        
        # สร้างโฟลเดอร์ output ถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/snapshots", exist_ok=True)
        
        print("🚦 Starting traffic monitoring with bma_lastest.py ...")
        
        # ใช้ bma_lastest.py แทน stream_to_counts.py และ argument ตามที่ user ใช้จริง
        cmd = [
            'python',
            f'{dashboard_path}/bma_lastest.py',
            '--cameras', cameras_config,
            '--bin_minutes', '5',
            '--frame_step_sec', '2',
            '--display'  # เพิ่ม --display ตามตัวอย่าง user
        ]
        print(f"Starting traffic monitoring with command: {' '.join(cmd)}")
        subprocess.Popen(cmd, cwd=dashboard_path,
                        stdout=open(f'{output_dir}/traffic_monitor.log', 'w'),
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid)
        time.sleep(10)
        if check_traffic_process():
            print("Traffic monitoring started successfully")
        else:
            raise Exception("Failed to start traffic monitoring process")
            
    except Exception as e:
        print(f"Error starting traffic monitoring: {e}")
        raise

def stop_traffic_monitoring():
    """หยุด traffic monitoring process"""
    try:
        result = subprocess.run(['pkill', '-f', 'bma_lastest.py'], 
                              capture_output=True, text=True)
        print(f"Stopped traffic monitoring processes")
        time.sleep(5)  # รอให้ process หยุดอย่างสมบูรณ์
    except Exception as e:
        print(f"Error stopping traffic monitoring: {e}")

def check_data_health():
    """ตรวจสอบสุขภาพของข้อมูลที่สร้างขึ้น"""
    try:
        data_dir = "/opt/airflow/data"
        snapshots_dir = f"{data_dir}/snapshots"
        
        # ตรวจสอบว่ามีไฟล์ CSV หรือไม่
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        # ตรวจสอบว่ามีรูปภาพ snapshot หรือไม่
        if os.path.exists(snapshots_dir):
            image_files = [f for f in os.listdir(snapshots_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        else:
            image_files = []
        
        print(f"Found {len(csv_files)} CSV files and {len(image_files)} image files")
        
        if len(csv_files) == 0:
            raise Exception("No CSV data files found")
        
        # ตรวจสอบไฟล์ล่าสุด
        latest_csv = max([f"{data_dir}/{f}" for f in csv_files], key=os.path.getmtime)
        last_modified = os.path.getmtime(latest_csv)
        current_time = time.time()
        
        # ถ้าไฟล์ล่าสุดเก่ากว่า 10 นาที แสดงว่าอาจมีปัญหา
        if current_time - last_modified > 600:  # 10 minutes
            print(f"WARNING: Latest CSV file is {(current_time - last_modified)/60:.1f} minutes old")
        else:
            print(f"Data health check passed - latest file updated {(current_time - last_modified)/60:.1f} minutes ago")
            
    except Exception as e:
        print(f"Error in data health check: {e}")
        raise

# สร้าง DAG
dag = DAG(
    'traffic_monitoring_pipeline',
    default_args=default_args,
    description='Traffic Monitoring with YOLO Detection Pipeline',
    schedule_interval=timedelta(hours=2),  # รันทุก 2 ชั่วโมงเพื่อประหยัดทรัพยากร
    catchup=False,
    max_active_runs=1,
)

# Task 1: ตรวจสอบว่า process ยังทำงานอยู่หรือไม่
check_process_task = PythonOperator(
    task_id='check_traffic_process',
    python_callable=check_traffic_process,
    dag=dag,
)

# Task 2: เริ่มต้น traffic monitoring ถ้าจำเป็น
start_monitoring_task = PythonOperator(
    task_id='start_traffic_monitoring',
    python_callable=start_traffic_monitoring,
    dag=dag,
)

# Task 3: ตรวจสอบสุขภาพข้อมูล
health_check_task = PythonOperator(
    task_id='data_health_check',
    python_callable=check_data_health,
    dag=dag,
)

# Task 4: รีสตาร์ท process ถ้าจำเป็น (ถ้า health check ล้มเหลว)
restart_monitoring_task = PythonOperator(
    task_id='restart_monitoring_if_needed',
    python_callable=lambda: [stop_traffic_monitoring(), start_traffic_monitoring()],
    dag=dag,
    trigger_rule='one_failed',  # รันเฉพาะเมื่อ health check ล้มเหลว
)

# กำหนดลำดับการทำงาน
check_process_task >> start_monitoring_task >> health_check_task
health_check_task >> restart_monitoring_task