#!/bin/bash

# 🔍 Traffic Monitoring System - Health Check Script

echo "🚦 ตรวจสอบระบบ Traffic Monitoring..."
echo "=================================================="

# ตรวจสอบโครงสร้างไฟล์
echo "📁 ตรวจสอบโครงสร้างไฟล์:"

if [ -f "/opt/airflow/dashboard/bmatraffic_yolo_pipeline/src/stream_to_counts.py" ]; then
    echo "✅ AI Detection script: พร้อมใช้งาน"
else
    echo "❌ AI Detection script: ไม่พบไฟล์!"
fi

if [ -f "/opt/airflow/dashboard/bmatraffic_yolo_pipeline/config/cameras.json" ]; then
    echo "✅ Camera config: พร้อมใช้งาน"
else
    echo "❌ Camera config: ไม่พบไฟล์!"
fi

if [ -f "/opt/airflow/dashboard/bmatraffic_yolo_pipeline/src/yolov8n.pt" ]; then
    echo "✅ YOLO Model: พร้อมใช้งาน"
    ls -lh /opt/airflow/dashboard/bmatraffic_yolo_pipeline/src/yolov8n.pt
else
    echo "❌ YOLO Model: ไม่พบไฟล์!"
fi

echo ""
echo "📊 ตรวจสอบโฟลเดอร์ข้อมูล:"
mkdir -p /opt/airflow/data/snapshots
ls -la /opt/airflow/data/

echo ""
echo "🔧 ตรวจสอบ Python packages:"
python -c "import cv2, ultralytics, pandas; print('✅ AI packages: พร้อมใช้งาน')" 2>/dev/null || echo "❌ AI packages: ขาดหายไป!"

echo ""
echo "🌐 ทดสอบการเชื่อมต่อ BMA Traffic:"
curl -s --connect-timeout 5 "http://www.bmatraffic.com" > /dev/null && echo "✅ BMA Website: เชื่อมต่อได้" || echo "❌ BMA Website: เชื่อมต่อไม่ได้"

echo ""
echo "📋 สรุปสถานะระบบ:"
echo "- Docker: $(docker --version 2>/dev/null | cut -d' ' -f3 || echo 'ไม่พบ')"
echo "- Airflow: $(airflow version 2>/dev/null || echo 'ไม่พบ')"
echo "- Python: $(python --version 2>/dev/null || echo 'ไม่พบ')"

echo ""
echo "🎯 การใช้งาน:"
echo "- Airflow UI: http://localhost:8080"  
echo "- Username: admin"
echo "- Password: admin"
echo "=================================================="
