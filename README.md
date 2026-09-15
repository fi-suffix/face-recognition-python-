# Facial Recognition CCTV Service (Python)

Layanan backend **face recognition** berbasis **FastAPI + OpenCV DNN**. Bertugas:

- mengambil daftar kamera aktif & embeddings karyawan dari dashboard Laravel (`http://localhost:8000`);
- membaca stream RTSP (lewat MediaMTX), mendeteksi wajah (**YuNet**), mengenali wajah (**SFace**);
- menyimpan snapshot hasil deteksi ke folder `snapshots/`;
- mencatat log deteksi ke Laravel melalui API.

## Dependensi

### Sistem / tools

| Perangkat | Fungsi |
| --- | --- |
| Python | 3.13+ |
| ffmpeg | mempublish webcam ke RTSP (`start_rtsp.ps1`) |
| MediaMTX | RTSP server multi-client (`rtsp-server/`) |
| Dashboard Laravel | sumber data kamera/employee + tempat penyimpanan log |

### Library Python — `requirements.txt`

| Package | Versi | Keterangan |
| --- | --- | --- |
| fastapi | 0.111.0 | REST API framework |
| uvicorn | 0.30.1 | ASGI server |
| opencv-python | 5.0.0 | deteksi (YuNet) + recognisi (SFace) via OpenCV DNN |
| numpy | 1.26.4 | operasi vektor embedding |
| requests | 2.32.3 | HTTP call ke API Laravel |
| python-dotenv | 1.0.1 | membaca `.env` |
| python-multipart | 0.0.32 | upload foto pada `/extract-embedding` |

### Model ONNX

Model diunduh otomatis saat startup pertama ke folder `models/`:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

(Di-ignore git. Jika jaringan ke GitHub diblokir, unduh manual dan taruh di `models/`.)

## Setup

```powershell
cd D:\PKL-project\Facial-recognition-cctv

# 1. Virtual environment (opsional tapi disarankan)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependensi
pip install -r requirements.txt

# 3. Konfigurasi
Copy-Item .env.example .env  # jika ada; atau buat manual dari tabel di bawah
```

## Konfigurasi — `.env`

| Key | Nilai contoh | Fungsi |
| --- | --- | --- |
| `LARAVEL_API_URL` | `http://localhost:8000/api` | URL API Laravel |
| `LARAVEL_API_KEY` | `your-secret-api-key-here` | API key — **harus sama** dengan `FACE_RECOGNITION_API_KEY` di `.env` Laravel (dikirim sebagai `Authorization: Bearer <key>`) |
| `FACE_RECOGNITION_THRESHOLD` | `0.45` | ambang batas kesamaan wajah (rendah = lebih longgar) |
| `CAMERA_RECONNECT_INTERVAL` | `5` | jeda (detik) reconnect bila stream putus |
| `LOG_COOLDOWN` | `3` | jeda (detik) antar log untuk fingerprint yang sama |
| `RECOGNITION_EVERY_N_FRAMES` | `2` | proses NN setiap N frame |
| `RECOGNITION_INTERVAL` | `2.0` | jeda (detik) antar pass recognition per kamera (membuat video tetap halus) |
| `MAX_STREAM_WIDTH` | `960` | lebar maksimum video MJPG |
| `STREAM_FPS` | `20` | target FPS |

## Cara menjalankan

Urutan yang benar:

1. **Pastikan dashboard Laravel berjalan** di `http://localhost:8000`
   (lihat README `dashboard-face-recognition`). Service Python membaca kamera & embeddings dari API ini.

2. **Start RTSP server + publish webcam:**

   ```powershell
   powershell -ExecutionPolicy Bypass -File start_rtsp.ps1
   ```

   Script ini:
   - menjalankan `rtsp-server\mediamtx.exe` (RTSP server di `rtsp://localhost:8554`);
   - mempublish webcam ke `rtsp://localhost:8554/live` via ffmpeg (TCP transport).

   > Jika sudah ada proses yang listen di port 8554, script hanya akan publish webcam.

3. **Jalankan service:**

   ```powershell
   python main.py
   ```

   Uvicorn akan listen di `http://localhost:8001`, otomatis memuat kamera aktif dari Laravel
   dan mulai memproses setiap stream. Logger mengenali wajah menggunakan embeddings
   yang diambil dari Laravel (di-cache, otomatis reload saat employee/foto diubah).

### Sebagai background service (Windows)

```powershell
# sesuaikan jika ingin jalan di background tanpa menutup terminal
cmd /c "cd /d D:\PKL-project\Facial-recognition-cctv && python main.py >> service.log 2>> service.log.err"
```

Output dan error tertulis ke `service.log` & `service.log.err` (di-ignore git).

## Endpoint API

| Method | Path | Keterangan |
| --- | --- | --- |
| GET | `/health` | health check (`{"status":"ok",...}`) |
| GET | `/cameras/status` | status semua stream |
| POST | `/cameras/{id}/start` | mulai stream kamera |
| POST | `/cameras/{id}/stop` | hentikan stream kamera |
| GET | `/cameras/{id}/stream` | video MJPG (live view) |
| GET | `/cameras/{id}/snapshot` | satu frame JPG (`?bbox=1` untuk gambar kotak deteksi) |
| POST | `/recognize` | recognisi dari satu frame (base64) |
| POST | `/test-detect` | deteksi one-shot untuk uji |
| POST | `/extract-embedding` | upload foto → embedding SFace (multipart) |
| POST | `/reload-embeddings` | muat ulang embeddings dari Laravel |
| POST | `/recompute-embeddings` | hitung ulang embeddings (rekomendasi: pakai `models/*.onnx`) |

Integrasi dengan dashboard Laravel terjadi di jalur ini: kamera diaktifkan/dinonaktifkan
melalui CRUD kamera di dashboard, yang otomatis memanggil `/cameras/{id}/start|stop`.

## Alur data

```
[Webcam / CCTV] --RTSP--> [MediaMTX :8554] --ffmpeg publish--> rtsp://localhost:8554/live
        ^
        --OPENCV (TCP ?tcp)--> [main.py :8001]
                                   |  detection (YuNet) -> recognition (SFace)
                                   |  snapshot -> snapshots/{camera}/{date}/*.jpg
                                   |  POST detections -> Laravel /api/face-recognition/detection-logs
                                   v
                          [Laravel dashboard :8000]
```

## Struktur penting

```
main.py                 # seluruh service (thread streaming + recognition worker)
start_rtsp.ps1          # start MediaMTX + publish webcam
requirements.txt        # dependensi Python
.env                    # konfigurasi (di-ignore git)
rtsp-server/            # mediamtx.exe + mediamtx.yml (di-ignore jurusnya)
snapshots/              # hasil snapshot (di-ignore git)
models/                 # ONNX: YuNet + SFace (di-ignore git)
test_webcam_live.py     # test live streaming webcam
test_webcam_detect.py   # test deteksi webcam
```

<!-- nih su-->