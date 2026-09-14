import cv2
import requests
import base64
import numpy as np

# Test webcam live detection
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit(1)

print("Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Send to recognition endpoint
    try:
        response = requests.post(
            'http://localhost:8001/recognize',
            json={'camera_id': 999, 'image_base64': img_base64},
            timeout=5
        )
        
        if response.ok:
            result = response.json()
            
            # Draw results on frame
            for det in result.get('detections', []):
                x, y, w, h = det['bbox']
                color = (0, 255, 0) if det['status'] == 'recognized' else (0, 0, 255)
                label = f"{det.get('employee_name', 'Unknown')} ({det['confidence']:.2f})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            print(f"Recognition error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Recognition request failed: {e}")
    
    cv2.imshow('Webcam Face Detection Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()