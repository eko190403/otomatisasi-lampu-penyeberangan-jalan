from ultralytics import YOLO
import cv2, serial, time, csv, os, sys
from datetime import datetime
import numpy as np

# ==========================
# KONFIGURASI
# ==========================
MODEL_PATH = "yolov8n.pt"

CAMERA_INDEX = 0
COM_PORT = "COM3"
BAUDRATE = 9600

CONF_THRESH = 0.35
DETECT_MIN = 5
GREEN_DURATION = 15
YELLOW_DURATION = 3

LOG_FILE = "event_log.csv"
FRAME_LOG_FILE = "frame_log.csv"
LATENCY_LOG_FILE = "latency_event.csv"
EVENT_LOG_FILE = "event_log_final.csv"
SCREENSHOT_DIR = "screenshots_final"

CONDITION = "Cerah"
JARAK = 4.0

# ==========================
# INIT MODEL
# ==========================
model = YOLO(MODEL_PATH)

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# ==========================
# INIT CSV
# ==========================
def init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["Timestamp", "Jumlah_Orang", "Status_Lampu", "Jarak", "Condition"]
            )

    if not os.path.exists(FRAME_LOG_FILE):
        with open(FRAME_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["FrameID", "Timestamp", "DetectionTime_ms", "Jumlah_Orang",
                 "Status_Lampu", "Screenshot_Path", "Confidence_Avg"]
            )

    if not os.path.exists(LATENCY_LOG_FILE):
        with open(LATENCY_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["EventID", "EventStart", "EventEnd", "Duration_s",
                 "Detection_Latency_s", "Response_Latency_s", "Total_Latency_s",
                 "Jumlah_Orang_Max", "Screenshot_Count"]
            )

    if not os.path.exists(EVENT_LOG_FILE):
        with open(EVENT_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["EventID", "EventStart", "EventEnd", "Duration_s",
                 "Jumlah_Orang_Max", "Status_Awal", "Status_Akhir", "Screenshot_Count"]
            )

init_csv()

# ==========================
# ARDUINO CONNECT
# ==========================
def connect_arduino():
    try:
        s = serial.Serial(COM_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
        print("[INFO] Arduino connected")
        return s
    except:
        print("[WARN] Arduino tidak terhubung")
        return None

def send_serial(cmd):
    if arduino:
        try:
            arduino.write((cmd + "\n").encode())
            arduino.flush()
        except:
            print("[WARN] Gagal kirim serial")

def check_serial():
    btn_pressed = False
    if arduino and arduino.in_waiting > 0:
        try:
            while arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()
                if line:
                    print(f"[ARDUINO] {line}")
                    if line == "BUTTON_PRESSED":
                        btn_pressed = True
        except:
            pass
    return btn_pressed

# ==========================
# LOGGING
# ==========================
def log_event_simple(jumlah, status):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, jumlah, status, JARAK, CONDITION])

def save_screenshot(frame, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{ts}.jpg"
    path = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return path

def log_frame(frame_id, det_time, jumlah, status, path, conf):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(FRAME_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([frame_id, ts, det_time, jumlah, status, path, conf])

def log_event_final(*row):
    with open(EVENT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def log_latency(*row):
    with open(LATENCY_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[FATAL] Kamera gagal dibuka")
    sys.exit(1)

arduino = connect_arduino()

# ==========================
# STATE
# ==========================
ROI = None
ROI_READY = False

current_lamp = "MERAH"
detected_start = None
green_start = None

event_active = False
event_id = 0
event_start_time = None
event_first_detection = None
event_lamp_change = None
event_jumlah_max = 0
event_screenshot_count = 0
event_status_awal = None

frame_counter = 0
jeda_aktif = False
jeda_start = None
manual_trigger = False

# Untuk FPS
fps = 0.0
last_fps_time = time.time()
frame_fps_count = 0

# ==========================
# MAIN LOOP
# ==========================
while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame_counter += 1

    # Cek serial dari Arduino (misal tombol ditekan)
    if check_serial():
        manual_trigger = True

    # ROI
    if not ROI_READY:
        h, w = frame.shape[:2]
        rw = int(w * 0.8)
        rh = int(h * 0.8)
        rx = (w - rw) // 2
        ry = (h - rh) // 2
        ROI = (rx, ry, rw, rh)
        ROI_READY = True

    rx, ry, rw, rh = ROI
    frame_roi = frame[ry:ry+rh, rx:rx+rw].copy()

    # YOLO
    t0 = time.time()
    results = model(frame_roi, imgsz=640, verbose=False)
    detection_time_ms = (time.time() - t0) * 1000

    jumlah_orang = 0
    confidences = []

    detections = []
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        for box, c, cl in zip(xyxy, conf, cls):
            if c >= CONF_THRESH:
                x1,y1,x2,y2 = map(int, box)
                label = class_names[cl] if cl < len(class_names) else str(cl)
                detections.append((x1+rx,y1+ry,x2+rx,y2+ry,c,label))
                confidences.append(c)
                if cl == 0:
                    jumlah_orang += 1

    conf_avg = np.mean(confidences) if confidences else 0.0

    now_time = time.time()

    # ==========================
    # EVENT START
    # ==========================
    # Event aktif jika ada orang atau ada trigger manual
    if (jumlah_orang > 0 or manual_trigger) and not event_active:
        event_active = True
        event_id += 1
        event_start_time = datetime.now()
        event_first_detection = now_time
        event_jumlah_max = jumlah_orang
        event_screenshot_count = 1
        event_status_awal = current_lamp
        save_screenshot(frame, f"event{event_id}_start")

    if event_active and jumlah_orang > event_jumlah_max:
        event_jumlah_max = jumlah_orang

    # ==========================
    # LAMP LOGIC
    # ==========================
    if current_lamp == "MERAH":

        if jumlah_orang > 0 or manual_trigger:
            if detected_start is None and not manual_trigger:
                detected_start = now_time
            elif manual_trigger or (now_time - detected_start >= DETECT_MIN):
                send_serial(f"HIJAU:{GREEN_DURATION}")
                current_lamp = "HIJAU"
                green_start = now_time
                event_lamp_change = now_time
                log_event_simple(jumlah_orang, "HIJAU")
                detected_start = None
                manual_trigger = False
        else:
            detected_start = None

    elif current_lamp == "HIJAU":
        if green_start and now_time - green_start >= GREEN_DURATION:
            # Masuk ke mode KUNING setelah HIJAU
            current_lamp = "KUNING"
            jeda_aktif = True
            jeda_start = now_time
            send_serial(f"KUNING:{YELLOW_DURATION}")
            green_start = None
            log_event_simple(0, "KUNING")

    elif current_lamp == "KUNING":
        if jeda_aktif and jeda_start is not None:
            sisa_jeda = max(0, int(YELLOW_DURATION - (now_time - jeda_start)))
            if now_time - jeda_start >= YELLOW_DURATION:
                send_serial("MERAH")
                current_lamp = "MERAH"
                jeda_aktif = False
                jeda_start = None
                log_event_simple(0, "MERAH")

                if event_active:
                    event_end_time = datetime.now()
                    duration = (event_end_time - event_start_time).total_seconds()

                    detection_latency = (
                        event_lamp_change - event_first_detection
                    ) if event_lamp_change else 0

                    response_latency = 0
                    total_latency = detection_latency

                    event_screenshot_count += 1
                    save_screenshot(frame, f"event{event_id}_end")

                    log_event_final(
                        event_id,
                        event_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        event_end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        round(duration,2),
                        event_jumlah_max,
                        event_status_awal,
                        "MERAH",
                        event_screenshot_count
                    )

                    log_latency(
                        event_id,
                        event_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        event_end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        round(duration,2),
                        round(detection_latency,2),
                        round(response_latency,2),
                        round(total_latency,2),
                        event_jumlah_max,
                        event_screenshot_count
                    )

                    event_active = False
                    event_start_time = None
                    event_first_detection = None
                    event_lamp_change = None
                    event_jumlah_max = 0
                    event_screenshot_count = 0

    # ==========================
    # VISUAL
    # ==========================
    overlay = frame.copy()

    # Hitung FPS
    frame_fps_count += 1
    now_fps_time = time.time()
    if now_fps_time - last_fps_time >= 1.0:
        fps = frame_fps_count / (now_fps_time - last_fps_time)
        last_fps_time = now_fps_time
        frame_fps_count = 0

    cv2.rectangle(overlay,(rx,ry),(rx+rw,ry+rh),(255,255,0),2)

    for x1,y1,x2,y2,c,label in detections:
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,255),2)
        text = f"{label} {c:.2f}"
        cv2.putText(overlay, text, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)


    # Warna lampu
    if current_lamp == "HIJAU":
        lamp_color = (0,255,0)
    elif current_lamp == "MERAH":
        lamp_color = (0,0,255)
    else:
        lamp_color = (0,255,255)  # KUNING

    cv2.putText(overlay,f"Lampu: {current_lamp}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,lamp_color,3)

    if current_lamp=="HIJAU" and green_start:
        sisa = max(0,int(GREEN_DURATION-(now_time-green_start)))
        cv2.putText(overlay,f"Sisa HIJAU: {sisa}s",(20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    if current_lamp=="KUNING" and jeda_aktif and jeda_start is not None:
        sisa_jeda = max(0, int(YELLOW_DURATION - (now_time - jeda_start)))
        cv2.putText(overlay, f"Kuning: {sisa_jeda}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    cv2.putText(overlay,f"Orang: {jumlah_orang}",(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    # Tampilkan FPS
    cv2.putText(overlay, f"FPS: {fps:.2f}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Tampilkan waktu real time
    waktu_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(overlay, f"Waktu: {waktu_str}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Deteksi Penyebrangan", overlay)

    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()