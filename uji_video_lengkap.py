

import cv2
import time
import csv
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import pandas as pd

# ================================
# KONFIGURASI UMUM
# ================================
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.35
ROI_PERCENT = 0.8

# EVENT PARAM
MIN_EVENT_DURATION = 0.5
MIN_GAP_BETWEEN_EVENT = 1.0

# LAMP PARAM
LAMP_HOLD_TIME = 2.0

# Jarak penyeberangan (meter)
JARAK = 4.0

# ================================
# FUNGSI PENGUJIAN VIDEO
# ================================
def uji_video(video_path, condition, output_prefix):
    """
    Menguji satu video dan menyimpan hasil ke file CSV
    
    Args:
        video_path: Path ke file video
        condition: "malam" atau "siang"
        output_prefix: Prefix untuk nama file output
    
    Returns:
        dict dengan hasil pengujian
    """
    print(f"\n{'='*70}")
    print(f" PENGUJIAN VIDEO: {video_path}")
    print(f" Kondisi: {condition.upper()}")
    print(f"{'='*70}")
    
    # Output files
    event_log = f"{output_prefix}_event_log.csv"
    frame_log = f"{output_prefix}_frame_log.csv"
    latency_log = f"{output_prefix}_latency_log.csv"
    screenshot_dir = f"{output_prefix}_screenshots"
    
    # Init model
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Video tidak bisa dibuka: {video_path}")
        return None
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"\n📹 Video Info:")
    print(f"   Total Frames: {total_frames}")
    print(f"   FPS: {video_fps:.2f}")
    print(f"   Duration: {video_duration:.2f}s")
    
    # ROI setup
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame pertama gagal")
        return None
    
    h, w = frame.shape[:2]
    roi_w = int(w * ROI_PERCENT)
    roi_h = int(h * ROI_PERCENT)
    roi_x = (w - roi_w) // 2
    roi_y = (h - roi_h) // 2
    ROI = (roi_x, roi_y, roi_w, roi_h)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create screenshot dir
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    
    # Init CSV files
    with open(event_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "EventID", "EventStart", "EventEnd", "Duration_s",
            "Jumlah_Orang_Max", "Status_Awal", "Status_Akhir", 
            "Screenshot_Count", "Jarak", "Condition"
        ])
    
    with open(frame_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "FrameID", "Timestamp", "DetectionTime_ms", "Jumlah_Orang",
            "Status_Lampu", "Screenshot_Path", "Confidence_Avg", 
            "Jarak", "Condition"
        ])
    
    with open(latency_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "EventID", "EventStart", "EventEnd", "Duration_s",
            "Detection_Latency_s", "Response_Latency_s", "Total_Latency_s",
            "Jumlah_Orang_Max", "Screenshot_Count", "Jarak", "Condition"
        ])
    
    # State variables
    event_active = False
    event_start_time = None
    event_first_detection = None
    last_event_end_time = 0.0
    
    lamp_status = "MERAH"
    lamp_last_change = time.time()
    
    event_id = 0
    event_jumlah_max = 0
    event_screenshot_count = 0
    event_status_awal = None
    frame_counter = 0
    
    # Metrics collection
    all_detection_times = []
    all_confidences = []
    all_people_counts = []
    
    print(f"\n⏳ Processing...")
    start_time = time.time()
    
    # ================================
    # MAIN PROCESSING LOOP
    # ================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rx, ry, rw, rh = ROI
        frame_roi = frame[ry:ry+rh, rx:rx+rw]
        frame_counter += 1
        
        # Progress indicator
        if frame_counter % 100 == 0:
            progress = (frame_counter / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_counter}/{total_frames})")
        
        # Inference
        t_inf_start = time.time()
        results = model(frame_roi, verbose=False)
        t_inf_end = time.time()
        inference_time = t_inf_end - t_inf_start
        detection_time_ms = inference_time * 1000
        all_detection_times.append(detection_time_ms)
        
        # Count persons & confidence
        jumlah_orang = 0
        confidences = []
        det = results[0]
        
        if det.boxes is not None:
            for box in det.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= CONF_THRESH:
                    jumlah_orang += 1
                    confidences.append(float(box.conf[0]))
        
        conf_avg = np.mean(confidences) if confidences else 0.0
        if confidences:
            all_confidences.extend(confidences)
        
        now = time.time()
        screenshot_path = ""
        
        # Event Logic
        if jumlah_orang > 0:
            all_people_counts.append(jumlah_orang)
            
            if not event_active:
                if now - last_event_end_time >= MIN_GAP_BETWEEN_EVENT:
                    event_active = True
                    event_start_time = now
                    event_first_detection = now
                    event_id += 1
                    event_jumlah_max = jumlah_orang
                    event_screenshot_count = 0
                    event_status_awal = lamp_status
            else:
                if jumlah_orang > event_jumlah_max:
                    event_jumlah_max = jumlah_orang
            
            # Save screenshot
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"event{event_id}_frame{frame_counter}_{ts}.jpg"
            screenshot_path = os.path.join(screenshot_dir, filename)
            cv2.imwrite(screenshot_path, frame)
            event_screenshot_count += 1
            
        else:
            if event_active:
                duration = now - event_start_time
                
                if duration >= MIN_EVENT_DURATION:
                    start_ts = datetime.fromtimestamp(event_start_time)
                    end_ts = datetime.fromtimestamp(now)
                    
                    will_change_lamp = (lamp_status == "MERAH")
                    detection_latency = MIN_EVENT_DURATION
                    
                    if will_change_lamp:
                        response_latency = now - event_first_detection
                        lamp_status = "HIJAU"
                        lamp_last_change = now
                        new_status = "HIJAU"
                    else:
                        response_latency = 0
                        new_status = lamp_status
                    
                    total_latency = detection_latency + response_latency
                    
                    # Log event
                    with open(event_log, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            event_id,
                            start_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            end_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            f"{duration:.2f}",
                            event_jumlah_max,
                            event_status_awal,
                            new_status,
                            event_screenshot_count,
                            JARAK,
                            condition
                        ])
                    
                    # Log latency
                    with open(latency_log, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            event_id,
                            start_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            end_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            f"{duration:.2f}",
                            f"{detection_latency:.2f}",
                            f"{response_latency:.2f}",
                            f"{total_latency:.2f}",
                            event_jumlah_max,
                            event_screenshot_count,
                            JARAK,
                            condition
                        ])
                
                event_active = False
                event_start_time = None
                event_first_detection = None
                last_event_end_time = now
        
        # Lamp release
        if lamp_status == "HIJAU":
            if now - lamp_last_change >= LAMP_HOLD_TIME:
                lamp_status = "MERAH"
                lamp_last_change = now
        
        # Frame logging (only frames with detection)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        if jumlah_orang > 0:
            with open(frame_log, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    frame_counter,
                    timestamp,
                    f"{detection_time_ms:.2f}",
                    jumlah_orang,
                    lamp_status,
                    screenshot_path,
                    f"{conf_avg:.3f}",
                    JARAK,
                    condition
                ])
    
    cap.release()
    processing_time = time.time() - start_time
    
    # ================================
    # SUMMARY RESULTS
    # ================================
    try:
        event_data = pd.read_csv(event_log)
        frame_data = pd.read_csv(frame_log)
        latency_data = pd.read_csv(latency_log)
    except:
        print("[WARNING] Tidak ada data terdeteksi")
        return None
    
    results_summary = {
        "condition": condition,
        "video_path": video_path,
        "total_frames": total_frames,
        "video_fps": video_fps,
        "video_duration": video_duration,
        "processing_time": processing_time,
        "total_events": len(event_data),
        "total_frames_detected": len(frame_data),
        "total_screenshots": event_data['Screenshot_Count'].sum() if len(event_data) > 0 else 0,
        "avg_detection_time_ms": np.mean(all_detection_times) if all_detection_times else 0,
        "min_detection_time_ms": np.min(all_detection_times) if all_detection_times else 0,
        "max_detection_time_ms": np.max(all_detection_times) if all_detection_times else 0,
        "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
        "min_confidence": np.min(all_confidences) if all_confidences else 0,
        "max_confidence": np.max(all_confidences) if all_confidences else 0,
        "avg_people_per_frame": np.mean(all_people_counts) if all_people_counts else 0,
        "max_people_detected": max(all_people_counts) if all_people_counts else 0,
        "event_log": event_log,
        "frame_log": frame_log,
        "latency_log": latency_log,
        "screenshot_dir": screenshot_dir
    }
    
    if len(event_data) > 0:
        results_summary.update({
            "avg_event_duration": event_data['Duration_s'].mean(),
            "min_event_duration": event_data['Duration_s'].min(),
            "max_event_duration": event_data['Duration_s'].max(),
            "avg_orang_per_event": event_data['Jumlah_Orang_Max'].mean(),
        })
    
    if len(latency_data) > 0:
        results_summary.update({
            "avg_detection_latency": latency_data['Detection_Latency_s'].mean(),
            "avg_response_latency": latency_data['Response_Latency_s'].mean(),
            "avg_total_latency": latency_data['Total_Latency_s'].mean(),
        })
    
    # Print summary
    print(f"\n{'='*70}")
    print(f" HASIL PENGUJIAN: {condition.upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 EVENT SUMMARY")
    print(f"   Total Events Detected  : {results_summary['total_events']}")
    print(f"   Total Frames Processed : {total_frames}")
    print(f"   Frames with Detection  : {results_summary['total_frames_detected']}")
    print(f"   Total Screenshots      : {results_summary['total_screenshots']}")
    print(f"   Processing Time        : {processing_time:.2f}s")
    
    print(f"\n⏱️  PERFORMANCE METRICS")
    print(f"   Avg Detection Time     : {results_summary['avg_detection_time_ms']:.2f} ms")
    print(f"   Min Detection Time     : {results_summary['min_detection_time_ms']:.2f} ms")
    print(f"   Max Detection Time     : {results_summary['max_detection_time_ms']:.2f} ms")
    print(f"   Avg Confidence Score   : {results_summary['avg_confidence']:.3f}")
    
    if len(event_data) > 0:
        print(f"\n👥 DETECTION SUMMARY")
        print(f"   Max People Detected    : {results_summary['max_people_detected']}")
        print(f"   Avg People per Event   : {results_summary['avg_orang_per_event']:.1f}")
        print(f"   Avg Event Duration     : {results_summary['avg_event_duration']:.2f}s")
    
    if len(latency_data) > 0:
        print(f"\n⚡ LATENCY ANALYSIS")
        print(f"   Avg Detection Latency  : {results_summary['avg_detection_latency']:.2f}s")
        print(f"   Avg Response Latency   : {results_summary['avg_response_latency']:.2f}s")
        print(f"   Avg Total Latency      : {results_summary['avg_total_latency']:.2f}s")
    
    print(f"\n📁 OUTPUT FILES")
    print(f"   ✓ {event_log}")
    print(f"   ✓ {frame_log}")
    print(f"   ✓ {latency_log}")
    print(f"   ✓ {screenshot_dir}/")
    
    return results_summary


# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":
    print("="*70)
    print("   SISTEM DETEKSI PEJALAN KAKI - PENGUJIAN VIDEO LENGKAP")
    print("   Model: YOLOv8n | Confidence Threshold: 0.35")
    print("="*70)
    
    # Daftar video untuk pengujian
    videos = [
        ("Vidio jalan.MOV", "malam", "malam"),
        ("vidio jalan2.mp4", "siang", "siang"),
    ]
    
    all_results = []
    
    for video_path, condition, prefix in videos:
        if os.path.exists(video_path):
            result = uji_video(video_path, condition, prefix)
            if result:
                all_results.append(result)
        else:
            print(f"[WARNING] Video tidak ditemukan: {video_path}")
    
    # ================================
    # GABUNGAN HASIL
    # ================================
    if len(all_results) > 0:
        print("\n" + "="*70)
        print("   RINGKASAN GABUNGAN SEMUA VIDEO")
        print("="*70)
        
        # Save gabungan results
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv("hasil_pengujian_gabungan.csv", index=False)
        print("\n✓ Hasil gabungan disimpan: hasil_pengujian_gabungan.csv")
        
        # Print comparison table
        print("\n📊 PERBANDINGAN HASIL PENGUJIAN")
        print("-" * 70)
        print(f"{'Metrik':<30} {'Malam':>15} {'Siang':>15}")
        print("-" * 70)
        
        for i, r in enumerate(all_results):
            if i == 0:
                malam = r
            else:
                siang = r
        
        if len(all_results) >= 2:
            print(f"{'Total Events':<30} {malam['total_events']:>15} {siang['total_events']:>15}")
            print(f"{'Frames Detected':<30} {malam['total_frames_detected']:>15} {siang['total_frames_detected']:>15}")
            print(f"{'Avg Detection Time (ms)':<30} {malam['avg_detection_time_ms']:>15.2f} {siang['avg_detection_time_ms']:>15.2f}")
            print(f"{'Avg Confidence':<30} {malam['avg_confidence']:>15.3f} {siang['avg_confidence']:>15.3f}")
            print(f"{'Max People Detected':<30} {malam['max_people_detected']:>15} {siang['max_people_detected']:>15}")
            
            if 'avg_total_latency' in malam and 'avg_total_latency' in siang:
                print(f"{'Avg Total Latency (s)':<30} {malam['avg_total_latency']:>15.2f} {siang['avg_total_latency']:>15.2f}")
        
        print("-" * 70)
    
    print("\n" + "="*70)
    print("   PENGUJIAN SELESAI!")
    print("   Data siap untuk dianalisis dan dibuat BAB 4 Skripsi")
    print("="*70)
