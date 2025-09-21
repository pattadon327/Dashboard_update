import time, math, json, os, re, pytz, pandas as pd
from datetime import datetime, timedelta

# กำหนดประเภทรถที่ต้องการนับด้วย YOLO
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}

def now_local(tz_name="Asia/Bangkok"):
    tz = pytz.timezone(tz_name)
    return datetime.now(tz)

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-]+", "-", name.lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s or "camera"

def align_to_window(ts: datetime, minutes: int, tz_name="Asia/Bangkok"):
    """Return (window_start, window_end) that contains ts, aligned to N‑minute boundaries."""
    tz = pytz.timezone(tz_name)
    ts = ts.astimezone(tz)
    minute = (ts.minute // minutes) * minutes
    win_start = ts.replace(minute=minute, second=0, microsecond=0)
    win_end = win_start + timedelta(minutes=minutes)
    return win_start, win_end

def class_filter_map(model):
    """
    สร้าง mapping ระหว่าง class id ของ YOLO ไปยังชื่อประเภทรถ
    เฉพาะประเภทที่อยู่ใน VEHICLE_CLASS_NAMES เท่านั้น
    
    YOLOv8 classes ที่เกี่ยวข้อง:
    - 2: car
    - 3: motorcycle
    - 5: bus
    - 7: truck
    - 1: bicycle
    """
    # Build mapping from model.names id -> label string
    id_to_name = {int(i): n for i, n in model.names.items()} if hasattr(model, "names") else {}
    return {i: n for i, n in id_to_name.items() if n in VEHICLE_CLASS_NAMES}


def get_lag_values(csv_path, current_time, bin_minutes=5):
    """
    อ่านค่า lag จากไฟล์ CSV ที่มีอยู่ เพื่อใช้เป็น feature สำหรับการทำนาย
    
    Parameters:
    - csv_path: ที่อยู่ของไฟล์ CSV
    - current_time: เวลาปัจจุบัน
    - bin_minutes: ช่วงเวลาในการเก็บข้อมูล (นาที)
    
    Returns:
    - lag_1, lag_2, lag_3: ค่า vehicle_count ของ 1, 2, 3 คาบเวลาก่อนหน้า
    """
    # ค่า default ถ้าไม่มีข้อมูลก่อนหน้า
    lag_1, lag_2, lag_3 = 0, 0, 0
    
    # ถ้าไฟล์ไม่มีอยู่ ให้ return ค่า default
    if not os.path.exists(csv_path):
        return lag_1, lag_2, lag_3
    
    try:
        # อ่านไฟล์ CSV ที่มีอยู่
        df = pd.read_csv(csv_path)
        
        # ถ้าไฟล์ว่าง หรือมีเพียงแถวเดียว
        if len(df) < 1:
            return lag_1, lag_2, lag_3
            
        # แปลงคอลัมน์ timestamp เป็น datetime object
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'object':  # เป็น string (dd/mm/yyyy HH:MM)
                df['_dt'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
            elif '_unix_ts' in df.columns:  # เป็น Unix timestamp
                df['_dt'] = pd.to_datetime(df['_unix_ts'], unit='s')
            else:
                return lag_1, lag_2, lag_3
                
            # เรียงข้อมูลตามเวลา
            df = df.sort_values('_dt')
            
            # หาข้อมูลล่าสุด 3 แถว
            last_rows = df.tail(3)
            
            # ดึงค่า lag จากข้อมูลที่มี
            if len(last_rows) >= 1:
                lag_1 = last_rows.iloc[-1]['vehicle_count']
            if len(last_rows) >= 2:
                lag_2 = last_rows.iloc[-2]['vehicle_count']
            if len(last_rows) >= 3:
                lag_3 = last_rows.iloc[-3]['vehicle_count']
                
    except Exception as e:
        print(f"[warning] Error reading lag values: {e}")
    
    return int(lag_1), int(lag_2), int(lag_3)


def update_target_values(csv_path, current_time, current_value):
    """
    อัพเดตค่า target_next_1h ในไฟล์ CSV สำหรับการ forecast
    
    Parameters:
    - csv_path: ที่อยู่ของไฟล์ CSV
    - current_time: เวลาปัจจุบัน
    - current_value: ค่า vehicle_count ปัจจุบัน
    """
    # ถ้าไฟล์ไม่มีอยู่ ไม่ต้องทำอะไร
    if not os.path.exists(csv_path):
        return
    
    try:
        # อ่านไฟล์ CSV
        df = pd.read_csv(csv_path)
        
        # ถ้าไฟล์ว่าง หรือมีแถวน้อยกว่า 2 แถว
        if len(df) < 2:
            return
            
        # แปลงคอลัมน์ timestamp เป็น datetime object
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'object':  # เป็น string (dd/mm/yyyy HH:MM)
                df['_dt'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
            elif '_unix_ts' in df.columns:  # เป็น Unix timestamp
                df['_dt'] = pd.to_datetime(df['_unix_ts'], unit='s')
            else:
                return
                
            # หาข้อมูลล่าสุด 2 แถว
            last_rows = df.tail(2)
            current_time_str = current_time.strftime('%Y-%m-%d %H:%M')
            
            # ตรวจสอบว่าข้อมูลล่าสุด 2 แถวห่างกัน 1 ชั่วโมงหรือไม่
            # ถ้าใช่ ให้อัพเดตค่า target_next_1h ของแถวก่อนหน้า
            last_time = last_rows.iloc[-2]['_dt']
            current_time_dt = pd.to_datetime(current_time_str)
            
            # ถ้าเวลาห่างกัน 55-65 นาที (ประมาณ 1 ชั่วโมง)
            time_diff = (current_time_dt - last_time).total_seconds() / 60
            if 55 <= time_diff <= 65:
                # อัพเดตค่า target_next_1h ในแถวก่อนหน้า
                index_to_update = df.index[-2]  # แถวก่อนสุดท้าย
                df.at[index_to_update, 'target_next_1h'] = int(current_value)
                
                # เขียนกลับไปที่ไฟล์ CSV
                df.to_csv(csv_path, index=False)
                print(f"[update] Updated target_next_1h for row at {last_time.strftime('%H:%M')} to {current_value}")
            
    except Exception as e:
        print(f"[warning] Error updating target values: {e}")
