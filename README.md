# Smart Pelican Crossing dengan YOLOv8

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![Arduino](https://img.shields.io/badge/Arduino-C%2B%2B-teal)

Sistem Deteksi Penyebrangan Jalan Otomatis (Smart Pelican Crossing) berbasis Computer Vision. Proyek ini menggunakan YOLOv8 untuk mendeteksi pejalan kaki secara real-time dari kamera dan secara otomatis mengontrol perangkat keras lampu lalu lintas melalui mikrokontroler Arduino.

Proyek ini dibangun untuk keperluan pengujian dan evaluasi tugas.

---

## Fitur Utama

- **Deteksi Pejalan Kaki (YOLOv8):** Mendeteksi pejalan kaki di area tunggu secara real-time. Jika orang terdeteksi selama lebih dari 5 detik, sistem otomatis mengaktifkan lampu penyeberangan (Hijau).
- **Protokol Serial Dinamis:** Durasi lampu hijau dan lampu transisi kuning dikontrol secara dinamis dari Python ke Arduino tanpa nilai hardcode.
- **Transisi Keselamatan (Lampu Kuning):** Dilengkapi dengan fase transisi lampu kuning selama 3 detik sebelum kembali merah untuk standar keselamatan lalu lintas.
- **Fitur Fail-Safe (Tombol Manual):** Menyediakan push button fisik bagi pejalan kaki. Jika kamera gagal/error, pejalan kaki tetap bisa menyeberang dengan menekan tombol.
- **Audio Feedback:** Terintegrasi dengan modul MP3 (DFPlayer) untuk memutar instruksi suara ketika lampu hijau menyala.
- **Automated Data Logging:** Sistem otomatis mencatat log waktu (event, frame, latency) ke format CSV untuk analisis performa pengujian.

---

## Persyaratan Perangkat Keras (Hardware)
- Papan Arduino (Uno/Nano/Mega)
- 1x Modul Seven Segment TM1637 (Pin: CLK=2, DIO=3)
- 3x Modul LED / Relay Lampu Lalu Lintas (Pin: Merah=4, Hijau=5, Kuning=7)
- 1x Modul Suara DFPlayer Mini + Speaker (Pin: RX=10, TX=11)
- 1x Tombol Push Button (Pin: 8 ke GND)
- Webcam / CCTV untuk input visual

---

## Persiapan Perangkat Lunak (Software)

1. Pastikan Anda telah menginstal Python 3.8 atau yang lebih baru.
2. Clone repositori ini dan masuk ke dalam folder proyek.
3. Install dependensi library yang dibutuhkan:
   ```bash
   pip install ultralytics opencv-python numpy pandas pyserial matplotlib seaborn
   ```
4. Pastikan file model pre-trained `yolov8n.pt` tersedia di dalam root directory proyek.

---

## Cara Menjalankan

### 1. Konfigurasi Hardware (Arduino)
- Buka file `lampu.ino` menggunakan Arduino IDE.
- Sesuaikan konfigurasi pin jika terdapat perubahan wiring.
- Upload program ke board Arduino.

### 2. Menjalankan Deteksi Real-Time (Webcam)
Gunakan skrip utama untuk menjalankan deteksi secara live:
```bash
python yolo_traffic1.py
```
- Sistem akan otomatis mendeteksi port Arduino (secara default `COM3`, Anda dapat mengubahnya di dalam baris kode).
- Kamera akan aktif dan memproses frame secara real-time.
- Tekan **q** atau **ESC** pada jendela kamera untuk menghentikan program.

### 3. Pengujian Berdasarkan Video (Evaluasi Model)
Jika Anda ingin menguji akurasi model menggunakan rekaman video yang sudah ada (contoh: kondisi Siang vs Malam) tanpa perangkat keras tambahan:
```bash
python uji_video_lengkap.py
```
*Catatan: Pastikan file `Vidio jalan.MOV` dan `vidio jalan2.mp4` tersedia. Hasil analisis lengkap akan di-generate dalam bentuk file CSV dan grafik (PNG).*

---

## Struktur Folder Penting
- `lampu.ino`: Source code utama untuk mikrokontroler (Arduino).
- `yolo_traffic1.py`: Skrip inti sistem deteksi real-time menggunakan webcam.
- `uji_video_lengkap.py`: Skrip simulasi pengujian pada dataset video rekaman.
- `analisis_lengkap.py` & `analisis_hasil.py`: Skrip pengolah hasil data CSV untuk evaluasi matriks performa (Precision, Recall, F1-Score, dan Latency).
- `screenshots_final/`: Menyimpan tangkapan layar (screenshot) setiap event yang terdeteksi.

---

## Catatan Penting
- Jangan mengubah nama file log CSV (event_log, frame_log, dll) saat sistem sedang berjalan untuk mencegah bentrok penulisan (write error).
- Pastikan kabel tombol manual di-set menggunakan logika INPUT_PULLUP (dari Pin 8 langsung terhubung ke jalur Ground/GND).

---

## Kontak
Jika ada error terkait deteksi atau koneksi serial, periksa kembali device manager (COM Port) dan versi library ultralytics Anda.
