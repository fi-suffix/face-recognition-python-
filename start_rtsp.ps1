# Start MediaMTX RTSP server, then publish webcam to rtsp://localhost:8554/live
$ffmpeg = "C:\Users\padil\Downloads\ffmpeg-2026-09-07-git-ecc7eb519e-full_build\bin\ffmpeg.exe"
if (-not (Test-Path $ffmpeg)) {
    $ffmpeg = "ffmpeg"
}

$rtspServerDir = "D:\PKL-project\Facial-recognition-cctv\rtsp-server"

# 1) Start MediaMTX (real multi-client RTSP server on :8554)
if (-not (Get-NetTCPConnection -State Listen -LocalPort 8554 -ErrorAction SilentlyContinue)) {
    Write-Host "Starting MediaMTX RTSP server on rtsp://localhost:8554 ..." -ForegroundColor Green
    Start-Process -FilePath "$rtspServerDir\mediamtx.exe" -WorkingDirectory $rtspServerDir -WindowStyle Minimized
    Start-Sleep -Seconds 3
} else {
    Write-Host "MediaMTX already listening on :8554" -ForegroundColor Yellow
}

# 2) Publish the webcam to MediaMTX (TCP transport)
Write-Host "Publishing webcam to rtsp://localhost:8554/live" -ForegroundColor Green
& $ffmpeg -f dshow -video_size 640x480 -framerate 20 -i video="USB2.0 HD UVC WebCam" -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live