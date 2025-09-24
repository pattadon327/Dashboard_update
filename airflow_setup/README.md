# Traffic Monitoring Airflow Setup

ระบบ Airflow สำหรับจัดการ Traffic Monitoring Pipeline

## โครงสร้างไฟล์:

```
Dashboard_update/airflow_setup/
├── docker-compose.yml          # Airflow services configuration  
├── Dockerfile                  # Custom Airflow image (optional)
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
├── dags/                      # Airflow DAGs directory
│   ├── traffic_monitoring_dag.py      # หลัก DAG สำหรับควบคุม traffic monitoring
│   └── traffic_monitoring_test.py     # ทดสอบ DAG สำหรับตรวจสอบระบบ
├── logs/                      # Airflow logs
└── plugins/                   # Airflow plugins (ถ้าจำเป็น)
```

## การเริ่มต้นใช้งาน:

### 1. เข้าไปยังโฟลเดอร์ Airflow
```powershell
cd "D:\eco_project\Dashboard_update\airflow_setup"
```

### 2. เริ่มต้น Airflow
```powershell
docker-compose up -d
```

### 3. เข้าใช้งาน Airflow Web UI
- URL: http://localhost:8080
- Username: admin
- Password: admin

### 4. การทำงานของ DAGs

#### traffic_monitoring_dag (หลัก):
- รันทุกชั่วโมงเพื่อตรวจสอบระบบ
- ตรวจสอบว่า traffic monitoring process ยังทำงานอยู่
- เริ่มต้น process หากหยุดทำงาน
- ตรวจสอบสุขภาพข้อมูล CSV และรูปภาพ
- รีสตาร์ทถ้าจำเป็น

#### traffic_monitoring_test (ทดสอบ):
- รันทุก 30 นาทีเพื่อทดสอบระบบ
- ตรวจสอบ Python libraries
- ตรวจสอบไฟล์ traffic monitoring
- สร้างไฟล์ทดสอบ

### 5. การจัดการข้อมูล
- CSV files จะถูกสร้างใน: `/opt/airflow/data/`
- Images จะถูกสร้างใน: `/opt/airflow/data/snapshots/`
- Logs ใน: `./logs/`

### 6. Path Mapping (Docker Volumes):
- `./dags` → `/opt/airflow/dags`
- `./logs` → `/opt/airflow/logs`
- `./plugins` → `/opt/airflow/plugins`
- `../ai-detect-traffic` → `/opt/airflow/dashboard`
- `./data` → `/opt/airflow/data`

## คำสั่งการจัดการ:

### เริ่มระบบ:
```powershell
docker-compose up -d
```

### หยุดระบบ:
```powershell
docker-compose down
```

### ดู logs:
```powershell
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-webserver
```

### รีสตาร์ท:
```powershell
docker-compose restart
```

## หมายเหตุ:
- ระบบใช้ PostgreSQL เป็นฐานข้อมูล
- ติดตั้ง dependencies อัตโนมัติผ่าน _PIP_ADDITIONAL_REQUIREMENTS
- ไฟล์ของ traffic monitoring จะถูก mount เข้าไปใน container