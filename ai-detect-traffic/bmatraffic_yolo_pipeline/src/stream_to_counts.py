import argparse, os, json, time, traceback, cv2, pandas as pd, numpy as np, re, requests
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
from typing import Optional
from ultralytics import YOLO
from utils import align_to_window, class_filter_map, now_local, slugify, get_lag_values
from urllib.parse import urljoin, urlparse, parse_qs

def enhance_image_for_detection(image):
    """
    ปรับปรุงภาพให้เหมาะสำหรับการตรวจจับรถ โดยเฉพาะในเวลากลางคืน
    ใช้ adaptive histogram equalization และปรับ contrast/brightness อัตโนมัติ
    """
    if image is None or image.size == 0:
        return image
    
    # สำเนาภาพต้นฉบับ
    enhanced = image.copy()
    
    # ตรวจสอบความสว่างโดยรวมของภาพ
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # ถ้าภาพมืดมาก (กลางคืน) ให้ปรับปรุงแสง
    if mean_brightness < 80:  # ค่าความสว่างต่ำ = กลางคืน
        print(f"[enhance] Dark image detected (brightness: {mean_brightness:.1f}), applying night enhancement")
        
        # 1. ใช้ CLAHE (Contrast Limited Adaptive Histogram Equalization) เพื่อเพิ่ม contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        # แยกช่องสี BGR
        b, g, r = cv2.split(enhanced)
        
        # ใช้ CLAHE กับแต่ละช่องสี
        b_enhanced = clahe.apply(b)
        g_enhanced = clahe.apply(g)
        r_enhanced = clahe.apply(r)
        
        # รวมช่องสีกลับเข้าด้วยกัน
        enhanced = cv2.merge([b_enhanced, g_enhanced, r_enhanced])
        
        # 2. ปรับ gamma เพื่อเพิ่มความสว่างในพื้นที่มืด
        gamma = 1.2  # ค่า gamma > 1 จะทำให้ภาพสว่างขึ้น
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        # 3. ปรับ brightness และ contrast เล็กน้อย
        alpha = 1.15  # contrast multiplier (> 1 เพิ่ม contrast)
        beta = 15     # brightness addition
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
    elif mean_brightness > 200:  # ภาพสว่างมาก (อาจเป็นภาพขาวหรือแสงจ้า)
        print(f"[enhance] Very bright image detected (brightness: {mean_brightness:.1f}), applying bright enhancement")
        
        # ลด brightness และเพิ่ม contrast เล็กน้อย
        alpha = 1.1   # เพิ่ม contrast เล็กน้อย
        beta = -10    # ลด brightness
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
    elif mean_brightness < 120:  # ภาพค่อนข้างมืด (เช่น เย็น/เช้า)
        print(f"[enhance] Medium dark image detected (brightness: {mean_brightness:.1f}), applying moderate enhancement")
        
        # ใช้ CLAHE แบบอ่อนๆ
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # ปรับ brightness เล็กน้อย
        alpha = 1.05
        beta = 10
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    else:
        # ภาพมีความสว่างปกติ (กลางวัน) ไม่ต้องปรับ
        print(f"[enhance] Normal brightness image (brightness: {mean_brightness:.1f}), no enhancement needed")
    
    return enhanced

def detect_lighting_condition(image):
    """
    ตรวจสอบสภาพแสงของภาพ
    return: 'night', 'dawn_dusk', 'day', 'bright'
    """
    if image is None or image.size == 0:
        return 'unknown'
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # วิเคราะห์สภาพแสง
    if mean_brightness < 60:
        return 'night'
    elif mean_brightness < 120:
        return 'dawn_dusk'
    elif mean_brightness > 200:
        return 'bright'
    else:
        return 'day'

def resolve_stream_url(url: str, headers=None):
    """
    If the URL is a direct media stream (.m3u8, .mp4), return as is.
    If it's an HTML page (e.g., BMATraffic PlayVideo.aspx?ID=...), fetch and parse for an HLS/MP4 source.
    """
    headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; TrafficCounter/1.0)"}
    lower = url.lower()
    if lower.endswith(".mp4") or ".m3u8" in lower:
        return url

    try:
        from bs4 import BeautifulSoup

        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try common tags on the page
        for tag in soup.find_all(["source", "video"]):
            src_attr = tag.get("src") or tag.get("data-src")
            if not src_attr:
                continue
            full = urljoin(url, src_attr)
            if ".m3u8" in full or full.lower().endswith(".mp4"):
                return full

        # Try finding m3u8 in scripts/text
        m = re.search(r'(https?://[^"\']+\.m3u8[^"\']*)', r.text, flags=re.IGNORECASE)
        if m:
            return m.group(1)

        # Try to find ShowImage/ShowImageMark URLs inside inline scripts/HTML
        snap_inline = find_snapshot_in_text(r.text, url)
        if snap_inline:
            return snap_inline

        # Try <img> tags that often point to snapshot endpoints
        for img in soup.find_all("img"):
            src_attr = img.get("src") or img.get("data-src")
            if not src_attr:
                continue
            full = urljoin(url, src_attr)
            if is_image_like_url(full):
                return full

        # Follow iframes and repeat search inside
        for ifr in soup.find_all("iframe"):
            src = ifr.get("src")
            if not src:
                continue
            iframe_url = urljoin(url, src)
            try:
                r2 = requests.get(iframe_url, headers=headers, timeout=15)
                r2.raise_for_status()
                s2 = BeautifulSoup(r2.text, "html.parser")
                # search inside iframe for sources
                for tag in s2.find_all(["source", "video"]):
                    src2 = tag.get("src") or tag.get("data-src")
                    if not src2:
                        continue
                    full2 = urljoin(iframe_url, src2)
                    if ".m3u8" in full2 or full2.lower().endswith(".mp4"):
                        return full2
                m2 = re.search(r'(https?://[^"\']+\.m3u8[^"\']*)', r2.text, flags=re.IGNORECASE)
                if m2:
                    return m2.group(1)
                # Also check <img> inside iframe
                for img in s2.find_all("img"):
                    src2 = img.get("src") or img.get("data-src")
                    if not src2:
                        continue
                    full2 = urljoin(iframe_url, src2)
                    if is_image_like_url(full2):
                        return full2
                # And check inline scripts inside iframe for snapshot endpoints
                snap2 = find_snapshot_in_text(r2.text, iframe_url)
                if snap2:
                    return snap2
            except Exception:
                continue

        # Heuristic fallback: PlayVideo.aspx?ID=xxxx -> try known snapshot endpoints
        if "playvideo.aspx" in lower:
            guess = guess_snapshot_from_id(url, headers)
            if guess:
                return guess

    except Exception:
        pass

    # Fallback: return original (OpenCV may fail if not a direct stream)
    return url


# กำหนดประเภทรถที่ต้องการนับ
SUPPORTED = {"car", "motorcycle", "bus", "truck", "bicycle", "person"}

def open_stream(url: str):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {url}")
    return cap

def is_image_like_url(url: str) -> bool:
    lower = url.lower()
    if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
        return True
    # Many BMATraffic endpoints like show.aspx?image=996 or ShowImageMark.aspx
    if "show.aspx" in lower and "image=" in lower:
        return True
    if "showimagemark.aspx" in lower:
        return True
    return False

def derive_bma_referer(u: str) -> str:
    """
    If URL is a snapshot like show.aspx?image=996, derive the PlayVideo.aspx?ID=996
    to use as Referer. Otherwise, return the input URL.
    """
    try:
        lower = u.lower()
        if "show.aspx" in lower and ("image=" in lower or "id=" in lower):
            q = parse_qs(urlparse(u).query)
            cam_id = None
            for key in ("image", "id", "camera", "cameraid", "cam", "camid"):
                v = q.get(key)
                if v:
                    cam_id = v[0]
                    break
            if cam_id:
                return f"http://www.bmatraffic.com/PlayVideo.aspx?ID={cam_id}"
    except Exception:
        pass
    return u

def find_snapshot_in_text(html_text: str, base_url: str) -> Optional[str]:
    # Look for common BMATraffic snapshot endpoints in inline code
    patterns = [
        r'(["\'])(?P<u>ShowImageMark\.aspx[^"\']*)\1',
        r'(["\'])(?P<u>ShowImage\.aspx[^"\']*)\1',
        r'(["\'])(?P<u>show\.aspx\?image=[^"\']+)\1',
    ]
    for p in patterns:
        m = re.search(p, html_text, flags=re.IGNORECASE)
        if m:
            rel = m.group('u')
            return urljoin(base_url, rel)
    return None

def guess_snapshot_from_id(page_url: str, headers=None) -> Optional[str]:
    try:
        q = parse_qs(urlparse(page_url).query)
        cam_id = None
        for k, v in q.items():
            if k.lower() in ("id", "cam", "camid", "camera", "cameraid") and v:
                cam_id = v[0]
                break
        if not cam_id:
            return None
        candidates = [
            f"http://www.bmatraffic.com/ShowImageMark.aspx?ID={cam_id}",
            f"http://www.bmatraffic.com/ShowImageMark.aspx?id={cam_id}",
            f"http://www.bmatraffic.com/ShowImage.aspx?image={cam_id}",
            f"http://www.bmatraffic.com/show.aspx?image={cam_id}",
        ]
        headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; TrafficCounter/1.0)", "Referer": page_url}
        for u in candidates:
            try:
                r = requests.get(u, headers=headers, timeout=10)
                if r.status_code == 200:
                    ct = r.headers.get("Content-Type", "").lower()
                    if "image" in ct or is_image_like_url(u):
                        return u
            except Exception:
                continue
    except Exception:
        return None
    return None

def fetch_snapshot_frame(url: str, headers=None, session=None):
    """ดึงภาพจาก endpoint ที่เป็น snapshot (show.aspx, ShowImageMark.aspx, ไฟล์ภาพโดยตรง)"""
    # ต้องส่ง header พิเศษสำหรับ BMATraffic ที่ต้องการ Referer และ User-Agent เหมือนเบราว์เซอร์จริง
    full_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "th,en;q=0.9,en-US;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
    }
    
    # รวม headers จากพารามิเตอร์
    if headers:
        full_headers.update(headers)
        
    # แสดง headers ที่กำลังใช้ (สำหรับ debug)
    print(f"[debug] Fetching {url} with headers: {full_headers}")
    
    # ถ้า url เป็น PlayVideo.aspx, แปลงเป็น show.aspx?image=ID
    if "playvideo.aspx" in url.lower() and "id=" in url.lower():
        try:
            q = parse_qs(urlparse(url).query)
            cam_id = q.get("id", [None])[0] or q.get("ID", [None])[0]
            if cam_id:
                url = f"http://www.bmatraffic.com/show.aspx?image={cam_id}"
        except Exception:
            pass
            
    # เติมพารามิเตอร์ time=<epoch_ms> เพื่อกันแคช ตามรูปแบบของ BMA
    sep = "&" if "?" in url else "?"
    epoch_ms = int(time.time() * 1000)
    fetch_url = f"{url}{sep}time={epoch_ms}"
    # กันกรณีมี '&&' ซ้อน
    fetch_url = fetch_url.replace("&&", "&")
    
    # ถ้าเป็น ShowImageMark หรือ show.aspx, เตรียม referer ให้ถูกต้อง
    if "show" in fetch_url.lower() and "image=" in fetch_url.lower():
        try:
            q = parse_qs(urlparse(fetch_url).query)
            cam_id = q.get("image", [None])[0]
            if cam_id:
                full_headers["Referer"] = f"http://www.bmatraffic.com/PlayVideo.aspx?ID={cam_id}"
        except Exception:
            pass
    
    # ถ้าเป็นไฟล์ภาพโดยตรงจาก BMATraffic ให้เตรียม referer จาก URL
    if "bmatraffic.com" in fetch_url.lower() and ("/images/" in fetch_url.lower()):
        try:
            # ดึง camera ID จาก URL ของรูปภาพ (เช่น /images/traffic/996.jpg)
            parts = fetch_url.split("/")
            filename = parts[-1]  # เช่น 996.jpg
            cam_id = filename.split(".")[0]  # เช่น 996
            if cam_id.isdigit():
                full_headers["Referer"] = f"http://www.bmatraffic.com/PlayVideo.aspx?ID={cam_id}"
        except Exception:
            pass
    
    # ใช้ session ถ้ามี มิฉะนั้นใช้ requests ธรรมดา
    req = session.get if session else requests.get
    
    # พิเศษสำหรับ BMA Traffic ให้เพิ่ม Cache Buster ทุกครั้ง
    if "bmatraffic.com" in url:
        sep = "&" if "?" in fetch_url else "?"
        timestamp = int(time.time() * 1000)
        fetch_url = f"{fetch_url.split('?')[0]}?image={url.split('=')[-1]}&dummy={timestamp}"
    
    print(f"[debug] Final fetch URL: {fetch_url}")
    print(f"[debug] Headers: {full_headers}")
    
    # สร้างรายการ URLs ที่จะลอง (เพิ่มความหลากหลาย)
    urls_to_try = [fetch_url]
    
    # ถ้าเป็น BMATraffic เตรียม URLs สำรองเพิ่มเติม
    if "bmatraffic.com" in url:
        try:
            cam_id = None
            # ลองดึง camera ID จาก URL
            if "show.aspx" in url.lower() and "image=" in url.lower():
                q = parse_qs(urlparse(url).query)
                cam_id = q.get("image", [None])[0]
            elif "playvideo.aspx" in url.lower() and "id=" in url.lower():
                q = parse_qs(urlparse(url).query)
                cam_id = q.get("id", [None])[0] or q.get("ID", [None])[0]
                
            if cam_id:
                timestamp = int(time.time() * 1000)
                # เพิ่ม URLs ที่แตกต่างกัน
                urls_to_try.extend([
                    f"http://www.bmatraffic.com/show.aspx?image={cam_id}&dummy={timestamp}",
                    f"http://www.bmatraffic.com/ShowImageMark.aspx?ID={cam_id}&dummy={timestamp}",
                    f"http://www.bmatraffic.com/ShowImage.aspx?image={cam_id}&dummy={timestamp}",
                ])
        except Exception:
            pass
    
    # ทำการเรียก URL - ลองทุก URL ในรายการจนกว่าจะสำเร็จ
    response = None
    last_error = None
    
    for try_url in urls_to_try:
        try:
            print(f"[fetch] Trying URL: {try_url}")
            r = req(try_url, headers=full_headers, timeout=10)
            r.raise_for_status()
            
            # ถ้าได้รับ response ที่สมบูรณ์ ให้ใช้ response นี้
            if r.content and len(r.content) > 1000 and "image" in r.headers.get("Content-Type", "").lower():
                response = r
                print(f"[fetch] Success with URL: {try_url}")
                break
                
            # ถ้าไม่ใช่รูปภาพหรือรูปภาพมีขนาดเล็กเกินไป
            print(f"[fetch] URL returned non-image or small content: {try_url}, size={len(r.content)}, type={r.headers.get('Content-Type')}")
            last_error = f"Content is not an image or too small: {r.headers.get('Content-Type')}"
            
        except Exception as e:
            print(f"[fetch] Error with URL {try_url}: {e}")
            last_error = str(e)
    
    # ถ้าไม่มี URL ใดสำเร็จ
    if not response:
        raise RuntimeError(f"All URLs failed: {last_error}")
        
    # ใช้ response ที่สำเร็จ
    r = response
    
    # ตรวจสอบ Content-Type ว่าเป็นรูปภาพหรือไม่
    ct = r.headers.get("Content-Type", "")
    print(f"[debug] Response content type: {ct}")
    
    # ถ้าเป็น text/html แต่ขนาดเล็ก อาจจะเป็นหน้า error
    if "image" not in ct.lower():
        print(f"[warning] Content is not an image: {ct}")
        print(f"[debug] Response size: {len(r.content)} bytes")
        print(f"[debug] First 120 bytes: {r.content[:120]!r}")
        
        # ถ้าเป็น BMATraffic, ลอง URL อื่นๆ
        if "bmatraffic.com" in url:
            try:
                cam_id = None
                # ดึง camera ID จาก URL
                if "show.aspx" in url.lower() and "image=" in url.lower():
                    q = parse_qs(urlparse(url).query)
                    cam_id = q.get("image", [None])[0]
                elif "playvideo.aspx" in url.lower() and "id=" in url.lower():
                    q = parse_qs(urlparse(url).query)
                    cam_id = q.get("id", [None])[0] or q.get("ID", [None])[0]
                
                if cam_id:
                    # ลอง URL แบบใหม่
                    # แสดงให้เห็นถึงการปรับ URL และ headers
                    print(f"[retry] Original URL failed, trying with new approach for camera ID {cam_id}")
                    alt_url = f"http://www.bmatraffic.com/show.aspx?image={cam_id}&dummy={timestamp}"
                    print(f"[retry] Trying alternative URL: {alt_url}")
                    
                    # เก็บ cookies จากการเรียกก่อนหน้า
                    if "Set-Cookie" in r.headers:
                        cookies = r.headers.get("Set-Cookie")
                        full_headers["Cookie"] = cookies
                        print(f"[retry] Using cookies from previous response: {cookies}")
                    
                    # เรียก URL ใหม่
                    r_new = req(alt_url, headers=full_headers, timeout=10)
                    if r_new.status_code == 200 and "image" in r_new.headers.get("Content-Type", "").lower():
                        r = r_new  # ใช้ response ใหม่แทน
                        print(f"[retry] Success with alternative URL: {alt_url}")
                        print(f"[retry] New content type: {r.headers.get('Content-Type')}")
                    else:
                        print(f"[retry] Alternative URL failed: status={r_new.status_code}, ct={r_new.headers.get('Content-Type', '')}")
            except Exception as e:
                print(f"[retry] Error trying alternative URL: {e}")
    
    # ลอง decode ด้วย OpenCV ก่อน
    data = np.frombuffer(r.content, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    # ถ้า OpenCV ไม่สามารถ decode ได้ ลอง Pillow (เช่น กรณี WebP)
    if frame is None:
        try:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
            
    if frame is None:
        raise RuntimeError("Failed to decode snapshot image")
    return frame


def aggregate_loop(camera, model, bin_minutes=5, frame_step_sec=2, out_dir="data", tz="Asia/Bangkok", display=False):
    """
    ฟังก์ชันหลักสำหรับการประมวลผลวิดีโอจากกล้อง
    
    การทำงาน:
    1. แบ่งเวลาเป็นหน้าต่างขนาด 5 นาที (หรือตามที่กำหนดใน bin_minutes)
    2. ในแต่ละหน้าต่างเวลา จะดึงภาพและนับรถเพียงครั้งเดียวเท่านั้น (ไม่สะสมจากหลายเฟรม)
    3. บันทึกภาพ snapshot ล่าสุดที่ใช้ในการนับรถ (1 รูปต่อหน้าต่างเวลา)
    4. บันทึกข้อมูลการนับลงในไฟล์ CSV ในรูปแบบที่เหมาะกับ Time Series Forecasting
    
    Parameters:
    - camera: ข้อมูลกล้อง (dict) จากไฟล์ cameras.json
    - model: โมเดล YOLO ที่โหลดแล้ว (YOLOv8)
    - bin_minutes: ช่วงเวลาในการรวบรวมข้อมูล (นาที) สำหรับการนับรถ
    - frame_step_sec: ความถี่ในการดึง frame (วินาที) เพื่อควบคุมการใช้ทรัพยากร
    - out_dir: โฟลเดอร์สำหรับบันทึกไฟล์ CSV และรูปภาพ
    - tz: timezone
    - display: แสดงภาพขณะประมวลผลหรือไม่ (สำหรับ debug)
    """
    os.makedirs(out_dir, exist_ok=True)
    snapshot_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    cam_name = camera.get("name") or camera.get("slug") or "camera"
    cam_slug = camera.get("slug") or slugify(cam_name)
    original_url = camera["url"]
    snapshot_url = camera.get("snapshot_url")
    headers = camera.get("headers", {})

    if snapshot_url:
        url = snapshot_url
        snapshot_mode = True
        print(f"[resolver] Using snapshot_url for {camera.get('name')} -> {url}")
    else:
        url = resolve_stream_url(original_url)
        snapshot_mode = is_image_like_url(url)
        if snapshot_mode:
            print(f"[resolver] Using snapshot mode for {camera.get('name')} -> {url}")
        else:
            print(f"[resolver] Using video mode for {camera.get('name')} -> {url}")

    # ใช้ PlayVideo.aspx?ID=... เป็น Referer
    referer_url = derive_bma_referer(original_url if snapshot_mode else url)
    if "Referer" not in headers and snapshot_mode:
        headers["Referer"] = referer_url
    
    if snapshot_mode:
        print(f"[resolver] Using Referer: {headers.get('Referer', referer_url)}")
        
    # ต้องสร้าง session ใหม่สำหรับทุก reconnect ด้วย
    print("[resolver] Will create a new session for each connection attempt")

    csv_path = os.path.join(out_dir, f"{cam_slug}_{bin_minutes}min.csv")
    snapshot_path = os.path.join(snapshot_dir, f"{cam_slug}.jpg")

    # counters within current 5-min window
    win_start, win_end = None, None
    counts = {k: 0 for k in SUPPORTED}  # จะถูกรีเซ็ตเมื่อเริ่มหน้าต่างเวลาใหม่
    frames_processed = 0
    last_write = time.time()
    last_snapshot_window_start = None
    vehicle_counted_for_window = False  # เพิ่มตัวแปรเพื่อติดตามว่าได้นับรถในหน้าต่างเวลานี้ไปแล้วหรือยัง

    id2name = class_filter_map(model)

    while True:
        try:
            cap = None
            session = None
            if not snapshot_mode:
                cap = open_stream(url)
            else:
                # Prepare a session to carry cookies from the original page (helps bypass anti-hotlink)
                session = requests.Session()
                try:
                    # สร้าง session และ เข้าหน้าหลัก + หน้า PlayVideo ก่อนเพื่อให้ได้ cookies ที่ถูกต้อง
                    main_page_headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "Accept-Language": "th,en;q=0.9,en-US;q=0.8",
                    }
                    # เข้าหน้าหลักก่อน
                    print("[session] Visiting BMATraffic homepage to establish session...")
                    session.get("http://www.bmatraffic.com/", headers=main_page_headers, timeout=10)
                    # เข้าหน้า PlayVideo
                    print(f"[session] Visiting {referer_url} to get cookies...")
                    session.get(referer_url, headers=main_page_headers, timeout=10)
                    
                    # แสดง cookies ที่ได้
                    cookies_str = '; '.join([f"{c.name}={c.value}" for c in session.cookies])
                    print(f"[session] Session established with cookies: {cookies_str}")
                except Exception as e:
                    print(f"[session] Error establishing session: {e}")
            while True:
                # Wait frame_step_sec between frames to control compute
                time.sleep(frame_step_sec)
                
                # ลดความถี่ในการดึงเฟรมใหม่สำหรับการนับรถ (ลดภาระ CPU)
                # ให้ดึงเฟรมเพียงบางเฟรมเท่านั้นในแต่ละหน้าต่างเวลา (5 นาที)
                # จะดึงเฟรมถี่ขึ้นเมื่อใกล้จะสิ้นสุดหน้าต่างเวลา หรือเมื่อเริ่มหน้าต่างใหม่
                
                time_left_in_window = (win_end - ts).total_seconds() if win_end else 0
                
                # เพิ่มความถี่ในการดึงเฟรมเมื่อใกล้จะจบหน้าต่างเวลาหรือเพิ่งเริ่มหน้าต่างใหม่
                if time_left_in_window < 60 or frames_processed < 5:  # 1 นาทีสุดท้ายหรือ 5 เฟรมแรก
                    # ดึงเฟรมตามปกติ
                    pass
                else:
                    # ลดความถี่ในการดึงเฟรมลงในช่วงกลางของหน้าต่างเวลา
                    extra_sleep = min(frame_step_sec * 4, 10)  # รอเพิ่มไม่เกิน 10 วินาที
                    time.sleep(extra_sleep)
                    print(f"[optimize] Reduced frame rate in mid-window (time left: {time_left_in_window:.1f}s)")
                
                if snapshot_mode:
                    # Include Referer header pointing to the original page to bypass hotlink protections
                    fetch_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
                    # Add Referer and any custom headers from camera config
                    fetch_headers.update(headers)
                    if "Referer" not in fetch_headers:
                        fetch_headers["Referer"] = referer_url
                    # ถ้ามี cookies จาก session นำมาใส่ใน headers
                    if session and session.cookies:
                        cookies_str = '; '.join([f"{c.name}={c.value}" for c in session.cookies])
                        fetch_headers["Cookie"] = cookies_str
                    frame = fetch_snapshot_frame(url, headers=fetch_headers, session=session)
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("Stream read failed (frame is None). Reconnecting...")

                ts = now_local(tz)

                # Initialize window if needed
                if win_start is None or ts >= win_end:
                    # Flush previous window to CSV
                    if win_start is not None:
                        # เขียนข้อมูลลง CSV เมื่อสิ้นสุดหน้าต่างเวลา (5 นาที)
                        write_window(csv_path, win_start, win_end, counts, frames_processed, tz)
                        print(f"[window] End of 5-minute window. Total vehicles: {sum(counts.values())} | Details: {counts}")
                        
                    # Reset for new window
                    win_start, win_end = align_to_window(ts, bin_minutes, tz)
                    print(f"[window] Starting new 5-minute window: {win_start.strftime('%H:%M')} - {win_end.strftime('%H:%M')}")
                    counts = {k: 0 for k in SUPPORTED}  # รีเซ็ตการนับสำหรับหน้าต่างใหม่
                    frames_processed = 0
                    vehicle_counted_for_window = False  # รีเซ็ตตัวแปรบอกว่ายังไม่ได้นับรถในหน้าต่างเวลาใหม่นี้
                    
                    # จะทำการแคปรูปและนับเมื่อได้รับเฟรมที่เหมาะสม
                    print(f"[capture] Will capture and count vehicles once in this 5-minute window")
                    
                # ตรวจสอบว่าเราได้นับรถในหน้าต่างเวลานี้ไปแล้วหรือยัง
                # เราจะนับรถเพียงครั้งเดียวในแต่ละหน้าต่างเวลา 5 นาที
                if vehicle_counted_for_window:
                    # ข้ามการประมวลผล YOLO เนื่องจากได้นับรถไปแล้วในหน้าต่างเวลานี้
                    print(f"[skip] Already counted vehicles in this window ({win_start.strftime('%H:%M')} - {win_end.strftime('%H:%M')}), skipping YOLO processing")
                    continue
                else:
                    # แสดงข้อความว่าเรากำลังจะนับรถในหน้าต่างเวลานี้
                    print(f"[count] Counting vehicles for window: {win_start.strftime('%H:%M')} - {win_end.strftime('%H:%M')}")

                # Check if frame is valid for YOLO
                if frame is None or frame.size == 0:
                    print("[warning] Skipping invalid frame for YOLO inference")
                    continue
                
                # ตรวจสอบสภาพแสงและปรับปรุงภาพก่อนส่งเข้า YOLO
                lighting_condition = detect_lighting_condition(frame)
                print(f"[lighting] Detected lighting condition: {lighting_condition}")
                
                # สร้างภาพที่ปรับปรุงแล้วสำหรับ YOLO (แต่ยังคงบันทึกภาพต้นฉบับ)
                enhanced_frame = enhance_image_for_detection(frame)
                
                # ตรวจสอบภาพและบันทึก snapshot ดิบสำหรับหน้าต่างเวลานี้ 
                # (snapshot ที่มีการวาดเพิ่มเติมจะถูกบันทึกในภายหลัง)
                if last_snapshot_window_start != win_start:
                    try:
                        # บันทึกภาพต้นฉบับ (ไม่ใช่ภาพที่ปรับปรุงแล้ว) และตรวจสอบว่าเป็นภาพขาวหรือไม่
                        is_valid_image = save_snapshot(snapshot_path + ".raw.jpg", frame)
                        
                        if is_valid_image:
                            # ถ้าภาพสมบูรณ์ ให้บันทึกเวลาหน้าต่างและทำการนับต่อไป
                            last_snapshot_window_start = win_start
                            print(f"Valid raw snapshot saved: {os.path.basename(snapshot_path)}.raw.jpg @ {win_start.strftime('%Y-%m-%d %H:%M')}")
                        else:
                            # ถ้าภาพเป็นภาพขาว ให้ลองสร้าง session ใหม่
                            print(f"[retry] Blank image detected, trying to reconnect with new session...")
                            
                            # สร้าง session ใหม่เพื่อลองเชื่อมต่อใหม่
                            if snapshot_mode and session:
                                try:
                                    # ปิด session เดิม
                                    session.close()
                                    # สร้าง session ใหม่
                                    session = requests.Session()
                                    # เข้าหน้าหลัก + หน้า PlayVideo อีกครั้ง
                                    main_page_headers = {
                                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                                        "Accept-Language": "th,en;q=0.9,en-US;q=0.8",
                                    }
                                    print("[session] Creating a new session due to blank image...")
                                    session.get("http://www.bmatraffic.com/", headers=main_page_headers, timeout=10)
                                    session.get(referer_url, headers=main_page_headers, timeout=10)
                                    # แสดง cookies ที่ได้
                                    cookies_str = '; '.join([f"{c.name}={c.value}" for c in session.cookies])
                                    print(f"[session] New session established with cookies: {cookies_str}")
                                    
                                    # ไม่ให้นับในหน้าต่างนี้ เพื่อลองใหม่ในเฟรมถัดไป
                                    vehicle_counted_for_window = False
                                    print(f"[retry] Will try to count again in next frame with new session")
                                    
                                    # ข้ามการประมวลผล YOLO ในรอบนี้
                                    continue
                                except Exception as e:
                                    print(f"[session] Error refreshing session: {e}")
                            
                    except Exception as se:
                        print(f"Snapshot save failed: {se}")
                
                # Draw rectangle for debug if display is enabled
                if display:
                    debug_frame = frame.copy()  # ใช้ภาพต้นฉบับสำหรับ display
                    h, w = debug_frame.shape[:2]
                    cv2.rectangle(debug_frame, (0, 0), (w-1, h-1), (0, 255, 0), 2)
                    # เพิ่มข้อความแสดงว่านี่คือเฟรมที่ใช้ในการนับ
                    cv2.putText(debug_frame, "COUNTING FRAME", (10, h-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # แสดงสภาพแสง
                    cv2.putText(debug_frame, f"Light: {lighting_condition}", (10, h-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    debug_frame = frame
                    
                # YOLO inference - ใช้ภาพที่ปรับปรุงแล้ว
                try:
                    print(f"[yolo] Running inference on enhanced frame: {enhanced_frame.shape} (lighting: {lighting_condition})")
                    res = model.predict(source=enhanced_frame, verbose=False, device=0 if os.environ.get("YOLO_GPU","").strip() else None)
                    if not res:
                        print("[yolo] No results from YOLO model")
                        frames_processed += 1
                        continue
                        
                    r0 = res[0]
                    if r0.boxes is None or len(r0.boxes) == 0:  # no detections
                        print("[yolo] No objects detected")
                        frames_processed += 1
                        continue

                    # Get valid detections
                    cls_ids = r0.boxes.cls.tolist()
                    confidence_scores = r0.boxes.conf.tolist()
                    boxes = r0.boxes.xyxy.cpu().numpy()
                    
                    # Log confidence scores
                    print(f"[yolo] Detected {len(cls_ids)} objects with confidence: {confidence_scores}")
                    
                    # Apply confidence threshold แบบปรับตามสภาพแสง
                    # กลางคืนใช้ threshold ที่ต่ำกว่าเพื่อจับรถได้มากขึ้น
                    if lighting_condition == 'night':
                        confidence_threshold = 0.10  # ลดลงสำหรับกลางคืน
                    elif lighting_condition == 'dawn_dusk':
                        confidence_threshold = 0.12  # ค่ากลางสำหรับเวลาเช้า/เย็น
                    elif lighting_condition == 'bright':
                        confidence_threshold = 0.20  # เพิ่มขึ้นสำหรับแสงจ้า (ลด false positive)
                    else:  # day
                        confidence_threshold = 0.15  # ค่าปกติสำหรับกลางวัน
                    
                    print(f"[yolo] Using confidence threshold: {confidence_threshold:.2f} for {lighting_condition} conditions")
                    
                    confident_detections = [(int(cid), score, box) 
                                          for cid, score, box in zip(cls_ids, confidence_scores, boxes) 
                                          if score > confidence_threshold]
                    print(f"[yolo] After threshold ({confidence_threshold}): {len(confident_detections)} valid detections")
                    
                    # นับรถทุกคันที่ตรวจพบในเฟรมนี้ (แค่ครั้งเดียวต่อหน้าต่างเวลา) โดยแยกตามประเภท
                    vehicles_in_frame = {k: 0 for k in SUPPORTED}
                    
                    # Count vehicles by class
                    all_detections = []  # เก็บข้อมูลทุกการตรวจจับไว้แสดงผล
                    for cid, score, box in confident_detections:
                        label = id2name.get(cid)
                        # เก็บข้อมูลการตรวจจับทั้งหมด
                        all_detections.append((cid, label or f"class_{cid}", score, box))
                        
                        if label in SUPPORTED:
                            # เพิ่มค่าในประเภทของรถที่ตรวจพบในเฟรมนี้
                            vehicles_in_frame[label] += 1
                            # บันทึกลงในการนับรวมของหน้าต่างเวลานี้ (นับแค่ครั้งเดียว)
                            counts[label] += 1
                    
                    # แสดงการตรวจจับทั้งหมดที่พบ (ทั้งที่นับและไม่นับ)
                    print(f"[yolo] All detections: {[(d[1], d[2]) for d in all_detections]}")
                    
                    # เปลี่ยนสถานะเป็นได้นับรถในหน้าต่างเวลานี้แล้ว (จะไม่นับซ้ำอีก)
                    vehicle_counted_for_window = True
                    
                    # เขียนข้อมูลลง CSV ทันทีหลังการนับรถ (ไม่ต้องรอจนจบหน้าต่างเวลา)
                    write_window(csv_path, win_start, win_end, counts, frames_processed, tz, notes="immediate_write")
                    print(f"[csv] Immediately wrote counts to CSV: {sum(counts.values())} vehicles | Details: {counts}")
                    
                    # วาด boxes บน debug frame ทั้งในกรณี display และไม่ display (เพื่อให้บันทึกลงไฟล์ได้)
                    x1, y1, x2, y2 = box
                    # ใช้สีแดงสำหรับกรอบรถ
                    cv2.rectangle(debug_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 0, 255), 2)
                    # แสดงป้ายกำกับและความเชื่อมั่น
                    cv2.putText(debug_frame, f"{label}: {score:.2f}", 
                                (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # แสดงจำนวนรถที่พบในเฟรมนี้
                    total_in_frame = sum(vehicles_in_frame.values())
                    print(f"[yolo] Vehicles in current frame: {total_in_frame} {vehicles_in_frame}")
                    
                except Exception as e:
                    print(f"[yolo] Error in YOLO inference: {e}")
                    continue

                frames_processed += 1

                # Optional display for debugging
                # เตรียม debug_frame ใหม่เสมอเพื่อให้แน่ใจว่าเริ่มจากภาพสะอาด
                debug_frame = frame.copy()
                
                # วาดเส้นกรอบสีเขียวรอบภาพ
                h, w = debug_frame.shape[:2]
                cv2.rectangle(debug_frame, (0, 0), (w-1, h-1), (0, 255, 0), 2)
                
                # แสดงข้อความ "COUNTING FRAME" สีแดง
                cv2.putText(debug_frame, "COUNTING FRAME", (10, h-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # แสดงจำนวนรถรวมที่จับได้
                total_vehicles = sum(counts.values())
                cv2.putText(debug_frame, f"Vehicles: {total_vehicles}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
                # แสดงค่า threshold และสภาพแสง
                cv2.putText(debug_frame, f"Thresh: {confidence_threshold:.2f} ({lighting_condition})", (10, h-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # แสดงการปรับปรุงภาพ (ถ้ามี)
                if lighting_condition in ['night', 'dawn_dusk']:
                    cv2.putText(debug_frame, "Enhanced for low light", (10, h-80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # แสดงจำนวนรถแยกตามประเภท
                y_pos = 60
                for k, v in counts.items():
                    if v > 0:
                        cv2.putText(debug_frame, f"{k}: {v}", (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_pos += 25
                
                # บันทึกภาพที่มีการวาดเพิ่มเติมแล้ว (แทนที่จะบันทึกเฉพาะ frame ต้นฉบับ)
                # บันทึกเฉพาะเมื่อได้นับรถในหน้าต่างเวลานี้ไปแล้วเท่านั้น
                if vehicle_counted_for_window and last_snapshot_window_start == win_start:
                    try:
                        print("[snapshot] Saving enhanced debug frame with detection results...")
                        
                        # วาดการตรวจจับทั้งหมด (รวมถึงวัตถุที่ไม่ใช่ประเภทที่เรานับ)
                        for obj_class_id, obj_label, obj_score, obj_box in all_detections:
                            x1, y1, x2, y2 = obj_box
                            
                            # กำหนดสีต่างกันระหว่างวัตถุที่นับ (สีแดง) และไม่นับ (สีน้ำเงิน)
                            box_color = (0, 0, 255) if obj_label in SUPPORTED else (255, 0, 0)
                            
                            # วาดกรอบ
                            cv2.rectangle(debug_frame, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        box_color, 2)
                            
                            # ใส่ป้ายกำกับ
                            cv2.putText(debug_frame, f"{obj_label}: {obj_score:.2f}", 
                                        (int(x1), int(y1)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # เพิ่ม timestamp ที่มุมขวาบน
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(debug_frame, timestamp, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # บันทึกภาพที่มีข้อมูลเพิ่มเติมแล้ว
                        ok = cv2.imwrite(snapshot_path, debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if ok:
                            print(f"[snapshot] Successfully saved enhanced debug frame to {snapshot_path}")
                        else:
                            print(f"[snapshot] Failed to save enhanced debug frame")
                    except Exception as e:
                        print(f"[snapshot] Error saving enhanced debug frame: {e}")
                
                if display:
                    try:
                        # แสดงผลบนหน้าจอ
                        cv2.imshow(cam_slug, debug_frame)
                        # Press 'q' to exit, reduce CPU usage with longer wait
                        if cv2.waitKey(100) & 0xFF == ord('q'):
                            raise KeyboardInterrupt()
                    except Exception as e:
                        print(f"[display] Error showing frame: {e}")

        except KeyboardInterrupt:
            # Flush and exit
            if win_start is not None:
                write_window(csv_path, win_start, win_end, counts, frames_processed, tz)
            print("Stopped by user.")
            break
        except Exception as e:
            # On any error, flush partial and retry after short delay
            if win_start is not None:
                write_window(csv_path, win_start, win_end, counts, frames_processed, tz, notes=f"reconnect: {e}")
            else:
                print(f"[reconnect] {e}")
            time.sleep(5)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            if display:
                try:
                    cv2.destroyWindow(cam_slug)
                except Exception:
                    pass

def write_window(csv_path, win_start, win_end, counts, frames_processed, tz, notes=""):
    """บันทึกข้อมูลการนับรถลงในไฟล์ CSV ในรูปแบบที่เหมาะสำหรับโมเดล ML"""
    from utils import get_lag_values  # Import here to avoid circular imports
    
    # ใช้ end time เป็นเวลาอ้างอิง
    dt = win_end
    
    # ข้อมูลหลัก - ใช้รูปแบบ timestamp ตามตัวอย่าง DD/MM/YYYY HH:MM
    formatted_timestamp = dt.strftime('%d/%m/%Y %H:%M')
    
    # นับรวมรถทุกประเภทเป็น vehicle_count เพียงค่าเดียว พร้อมตรวจสอบค่าที่เกินจริง
    vehicle_count = sum(counts.values())
    
    # ตรวจสอบและแก้ไขค่าที่ผิดปกติ - ถ้าจำนวนรถสูงเกินไปสำหรับกล้องหนึ่งตัว (เช่น > 50 คัน)
    # อาจเป็นเพราะการตรวจจับผิดพลาดหรือนับซ้ำ - จำกัดค่าไม่ให้เกิน 50
    MAX_REASONABLE_COUNT = 50
    if vehicle_count > MAX_REASONABLE_COUNT:
        print(f"[warning] Unusually high vehicle count: {vehicle_count}, capping at {MAX_REASONABLE_COUNT}")
        print(f"[warning] Original counts by type: {counts}")
        vehicle_count = MAX_REASONABLE_COUNT
    
    # สร้าง lag values (ค่าย้อนหลัง 1, 2, 3 คาบเวลา)
    lag_1, lag_2, lag_3 = get_lag_values(csv_path, dt)
    
    # ข้อมูลเกี่ยวกับเวลา
    day_of_week = dt.strftime('%A')  # ชื่อวัน (Monday, Tuesday, ...)
    hour = dt.hour  # ชั่วโมง (0-23)
    
    # สร้าง row สำหรับบันทึก CSV - เฉพาะ columns ที่จำเป็นสำหรับโมเดล
    row = {
        "timestamp": formatted_timestamp,
        "vehicle_count": int(vehicle_count),
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "day_of_week": day_of_week,
        "hour": hour
    }
    
    # บันทึกลง CSV
    df = pd.DataFrame([row])
    hdr = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=hdr, index=False)
    
    print(f"[{dt.strftime('%H:%M')}] {os.path.basename(csv_path)} +1 row (vehicle_count={vehicle_count}, single count for 5-min window)")

def save_snapshot(snapshot_path: str, frame):
    """
    Save one latest snapshot image. If an older image exists, delete it first
    and then write the new one. The frame is the exact image used for YOLO detection.
    """
    try:
        if os.path.exists(snapshot_path):
            try:
                os.remove(snapshot_path)
            except Exception:
                # If deletion fails (e.g., locked), continue to overwrite
                pass
    except Exception as e:
        print(f"[snapshot] Error removing old snapshot: {e}")
                
    # Check if frame is valid
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame - frame is None or empty")
    
    try:
        # ตรวจสอบว่าภาพเป็นภาพขาวหรือไม่
        is_blank = False
        # คำนวณพิกเซลที่ไม่ใช่สีขาว (นับเฉพาะพิกเซลที่มีค่ามากกว่า 240 ในทุกช่องสี)
        non_white_pixels = np.sum(np.any(frame < 240, axis=2))
        total_pixels = frame.shape[0] * frame.shape[1]
        white_percentage = 100 - (non_white_pixels * 100 / total_pixels)
        
        # ถ้าภาพมีพื้นที่สีขาวมากกว่า 95% ถือว่าเป็นภาพขาว
        if white_percentage > 95:
            is_blank = True
            print(f"[warning] Image appears to be blank (white): {white_percentage:.1f}% white pixels")
            # บันทึกข้อความแจ้งเตือนลงในภาพ
            cv2.putText(frame, "WARNING: BLANK IMAGE", (10, frame.shape[0] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{white_percentage:.1f}% white pixels", (10, (frame.shape[0] // 2) + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            
        # Debug image information
        h, w = frame.shape[:2]
        print(f"[snapshot] Saving image: {w}x{h}, dtype={frame.dtype}, shape={frame.shape}")
            
        # Make a copy of the frame to avoid any reference issues
        frame_copy = frame.copy()
        
        # Draw timestamp on the image
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_copy, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write JPEG image with high quality (95%)
        ok = cv2.imwrite(snapshot_path, frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")
            
        # Verify file was actually written
        if not os.path.exists(snapshot_path) or os.path.getsize(snapshot_path) < 1000:  # < 1KB is suspicious
            raise RuntimeError(f"Snapshot file too small: {os.path.getsize(snapshot_path) if os.path.exists(snapshot_path) else 0} bytes")
        
        status = "WARNING: BLANK IMAGE" if is_blank else "OK"
        print(f"[snapshot] Successfully saved {os.path.getsize(snapshot_path)} bytes to {snapshot_path} (Status: {status})")
        
        return not is_blank  # คืนค่า True ถ้าภาพไม่ใช่ภาพขาว
    except Exception as e:
        print(f"[snapshot] Error saving snapshot: {e}")
        raise

def main():
    ap = argparse.ArgumentParser(description="Traffic Monitoring with YOLOv8 and Time-Series CSV Output")
    ap.add_argument("--cameras", default="config/cameras.json", help="Path to cameras.json configuration file")
    ap.add_argument("--bin_minutes", type=int, default=5, help="Time window in minutes for data aggregation")
    ap.add_argument("--frame_step_sec", type=float, default=2.0, help="Seconds between processed frames (controls CPU usage)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path (default: yolov8n.pt for best performance)")
    ap.add_argument("--out_dir", default="data", help="Output directory for CSV files and snapshots")
    ap.add_argument("--display", action="store_true", help="Show frames for debugging; press q to quit")
    args = ap.parse_args()

    with open(args.cameras, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tz = cfg.get("timezone", "Asia/Bangkok")
    cameras = [c for c in cfg.get("cameras", []) if c.get("enabled", True)]

    if not cameras:
        raise SystemExit("No cameras enabled. Edit config/cameras.json or run scrape_bmatraffic.py.")

    model = YOLO(args.model)

    print(f"[setup] Using YOLOv8 model: {args.model}")
    print(f"[setup] Data will be saved to: {args.out_dir}")
    print(f"[setup] Time window: {args.bin_minutes} minutes")
    print(f"[setup] Frame sampling: every {args.frame_step_sec} seconds")
    print(f"[setup] Display mode: {'enabled' if args.display else 'disabled'}")
    
    # Simple sequential run (1 camera at a time). For multiple cams, run one process per camera.
    for cam in cameras:
        try:
            print(f"Starting camera: {cam.get('name') or cam.get('slug')}")
            aggregate_loop(cam, model, args.bin_minutes, args.frame_step_sec, args.out_dir, tz, display=args.display)
        except KeyboardInterrupt:
            print("\n[stop] Program stopped by user (Ctrl+C)")
            break
        except Exception as e:
            print(f"[error] Camera failed: {e}")
            time.sleep(3)

if __name__ == "__main__":
    main()
