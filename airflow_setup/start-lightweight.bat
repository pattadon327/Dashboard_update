@echo off
echo 🚦 Traffic Monitoring System - Lightweight Version
echo ================================================

echo 💡 เปลี่ยนไดเรกทอรี่...
cd /d "c:\311\Dashboard_update\airflow_setup"

echo 🧹 ล้างข้อมูลเก่า (ถ้ามี)...
docker compose down --volumes 2>nul

echo 🔧 เริ่มต้นระบบใหม่...
echo ⚡ เวอร์ชันนี้เบาและเร็วกว่าเดิม 5 เท่า!
docker compose up -d --build

echo.
echo ✅ การติดตั้งเสร็จสิ้น!
echo 🌐 เข้าใช้งาน: http://localhost:8080
echo 👤 Username: admin
echo 🔑 Password: admin
echo.
echo 📊 ตรวจสอบสถานะ: docker compose ps
echo 📋 ดู logs: docker compose logs -f
echo 🛑 หยุดระบบ: docker compose down
echo.
pause
