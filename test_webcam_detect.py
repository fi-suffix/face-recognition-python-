import cv2
import requests

# Capture frame from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture frame from webcam")
    exit(1)

# Save test image
cv2.imwrite('webcam_test_frame.jpg', frame)

# Send to test-detect endpoint
with open('webcam_test_frame.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/test-detect',
        files={'file': f}
    )

print("Status:", response.status_code)
print(response.json())