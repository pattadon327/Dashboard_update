# Multi-Interval Traffic Prediction Dashboard

## 📋 Overview

Dashboard สำหรับการทำนายจราจรแบบหลายช่วงเวลา (5 นาที, 15 นาที, 30 นาที, และ 1 ชั่วโมง) โดยใช้ Streamlit และ XGBoost Models

## 🚀 การติดตั้งและเรียกใช้งาน

### ติดตั้ง Dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn xgboost pillow
```

### เรียกใช้งาน Dashboard

```bash
streamlit run dash_main.py
```

## 📁 ไฟล์ที่จำเป็น

### 1. โมเดล Machine Learning (จำเป็น)

ไฟล์โมเดลที่ต้องมีในโฟลเดอร์ `D:/eco_project/Dashboard_update/`:

- `traffic_model_5m.pkl` - โมเดลทำนาย 5 นาที
- `traffic_model_15m.pkl` - โมเดลทำนาย 15 นาที
- `traffic_model_30m.pkl` - โมเดลทำนาย 30 นาที
- `traffic_model_1h.pkl` - โมเดลทำนาย 1 ชั่วโมง

### 2. ข้อมูล Training Dataset (จำเป็น)

- `D:/eco_project/Dashboard_update/traffic_dataset_with_targets.csv`
  - ข้อมูลสำหรับการวิเคราะห์และการตรวจสอบ
  - ต้องมีคอลัมน์: `timestamp`, `vehicle_count`, `lag5m`, `lag10m`, `lag15m`, `day_of_week`, `hour`, `target_next_5m`, `target_next_15m`, `target_next_30m`, `target_next_1h`

### 3. ข้อมูล Real-time (เสริม)

- `D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/sathon12_footbridge_id996_5min.csv`
  - ข้อมูลจราจรปัจจุบันจาก CCTV
  - ต้องมีคอลัมน์: `timestamp`, `vehicle_count`, `lag_1`, `lag_2`, `lag_3`, `day_of_week`, `hour`

### 4. รูปภาพ CCTV (เสริม)

- `D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/snapshots/sathon12_footbridge_id996.jpg` - รูป YOLO Detected
- `D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/snapshots/sathon12_footbridge_id996.jpg.raw.jpg` - รูป Raw CCTV

## 🔧 การตั้งค่า Path

หากต้องการเปลี่ยน path ของไฟล์ ให้แก้ไขในไฟล์ `dash_add.py`:

### โมเดล

```python
model_paths = {
    '5m': 'D:/eco_project/Dashboard_update/traffic_model_5m.pkl',
    '15m': 'D:/eco_project/Dashboard_update/traffic_model_15m.pkl',
    '30m': 'D:/eco_project/Dashboard_update/traffic_model_30m.pkl',
    '1h': 'D:/eco_project/Dashboard_update/traffic_model_1h.pkl'
}
```

### ข้อมูล Training

```python
df = pd.read_csv('D:/eco_project/Dashboard_update/traffic_dataset_with_targets.csv')
```

### ข้อมูล Real-time

```python
csv_path = 'D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/sathon12_footbridge_id996_5min.csv'
```

## 🎯 ฟีเจอร์หลัก

### 1. หน้า Prediction

- **Live CCTV Images**: แสดงรูปภาพจาก CCTV แบบ Real-time
- **Current Data Prediction**: ทำนายจากข้อมูลปัจจุบัน
- **Manual Input**: กรอกข้อมูลเองเพื่อทดสอบ
- **Multi-Interval Results**: ผลทำนาย 4 ช่วงเวลาพร้อมกัน

### 2. หน้า Data Analysis

- **Dataset Overview**: สถิติพื้นฐานของข้อมูล
- **Traffic Patterns**: แนวโน้มจราจรตามเวลา
- **Multi-Interval Analysis**: วิเคราะห์การทำนายหลายช่วงเวลา
- **Feature Correlations**: ความสัมพันธ์ระหว่างตัวแปร

### 3. หน้า Model Info

- **Model Details**: รายละเอียดของโมเดล
- **Feature Statistics**: สถิติของ Features
- **Sample Data**: ตัวอย่างข้อมูล

## 📊 รูปแบบข้อมูล

### Input Features (6 ตัวแปร)

1. `vehicle_count`: จำนวนรถปัจจุบัน
2. `lag5m` (lag_1): จำนวนรถ 5 นาทีที่แล้ว
3. `lag10m` (lag_2): จำนวนรถ 10 นาทีที่แล้ว
4. `lag15m` (lag_3): จำนวนรถ 15 นาทีที่แล้ว
5. `hour`: ชั่วโมง (0-23)
6. `day_of_week_encoded`: วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)

### Target Variables

- `target_next_5m`: จำนวนรถในอนาคต 5 นาที
- `target_next_15m`: จำนวนรถในอนาคต 15 นาที
- `target_next_30m`: จำนวนรถในอนาคต 30 นาที
- `target_next_1h`: จำนวนรถในอนาคต 1 ชั่วโมง

## 🔍 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

1. **โมเดลโหลดไม่ได้**

   - ตรวจสอบว่าไฟล์ .pkl อยู่ใน path ที่ถูกต้อง
   - ตรวจสอบสิทธิ์การอ่านไฟล์
2. **ข้อมูลโหลดไม่ได้**

   - ตรวจสอบรูปแบบวันที่ในไฟล์ CSV
   - ตรวจสอบชื่อคอลัมน์ให้ตรงกับที่กำหนด
3. **รูปภาพไม่แสดง**

   - ตรวจสอบ path ของรูปภาพ
   - ตรวจสอบว่าไฟล์รูปภาพมีอยู่จริง

## 📈 Performance

- Dashboard รองรับการ Auto-refresh ทุก 30 วินาที
- แสดงผลได้ทั้งข้อมูล Training และ Real-time
- รองรับการทำนายหลายช่วงเวลาพร้อมกัน

## 🛠️ เทคโนโลยีที่ใช้

- **Frontend**: Streamlit
- **Machine Learning**: XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Image Processing**: PIL

## 📝 หมายเหตุ

- ระบบจะทำงานได้แม้ไม่มีข้อมูล Real-time (จะใช้ข้อมูล Training แทน)
- การ Auto-refresh จะทำงานเฉพาะเมื่อมีข้อมูล Real-time
- โมเดลทั้ง 4 ตัวต้องมีการ Train ด้วย Features เดียวกัน

---

**Created by**: Traffic Prediction Team
**Last Updated**: September 2025
