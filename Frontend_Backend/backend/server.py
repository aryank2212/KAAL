import cv2
import numpy as np
import time
from ultralytics import YOLO
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import threading
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the model
import os
model_path = os.path.join(os.path.dirname(__file__), "..", "..", "MODEL", "yolov8m.pt")
print(f"Loading YOLO model: {model_path}")
model = YOLO(model_path)

# --- DISTANCE ESTIMATION CONSTANTS ---
KNOWN_WIDTHS = {
    "person": 50,
    "cell phone": 7,
    "laptop": 35,
    "bottle": 8,
    "cup": 8,
    "chair": 50,
    "backpack": 30,
    "car": 180,
}
FOCAL_LENGTH = 600 

def estimate_distance(pixel_width, real_width):
    if pixel_width == 0: return 0
    return (real_width * FOCAL_LENGTH) / pixel_width

def analyze_frequency(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask_size = 50 
    high_freq_data = np.copy(magnitude_spectrum)
    high_freq_data[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
    noise_score = np.mean(high_freq_data)
    return magnitude_spectrum, noise_score

# Global state
latest_frame = None
latest_data = {
    "status": "INITIALIZING",
    "noise_score": 0,
    "detections": [],
    "systemMetrics": {
        "cpu": 45,
        "memory": 62,
        "gpu": 28,
        "bandwidth": 92,
        "latency": 18,
        "uptime": "00:00:00"
    },
    "data": []
}

def video_worker():
    global latest_frame, latest_data
    cap = cv2.VideoCapture(0)
    ATTACK_THRESHOLD = 110.0 
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame = cv2.resize(frame, (640, 480))
        
        # YOLO
        results = model.predict(source=frame, conf=0.4, verbose=False)
        out_frame = results[0].plot()
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pixel_width = x2 - x1
                real_width = KNOWN_WIDTHS.get(cls_name.lower(), 50)
                distance_m = estimate_distance(pixel_width, real_width) / 100.0
                detections.append({"label": cls_name, "distance": round(distance_m, 2)})
        
        # FFT
        _, noise_score = analyze_frequency(frame)
        is_attack = noise_score > ATTACK_THRESHOLD
        
        # Update UI text on frame
        color = (0, 0, 255) if is_attack else (0, 255, 0)
        status_text = "ADVERSARIAL ATTACK" if is_attack else "SYSTEM NOMINAL"
        cv2.putText(out_frame, f"NOISE: {noise_score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if is_attack:
            cv2.rectangle(out_frame, (0, 0), (640, 480), (0, 0, 255), 10)

        # Update latest_frame
        _, buffer = cv2.imencode('.jpg', out_frame)
        latest_frame = buffer.tobytes()
        
        # Update metrics
        uptime_sec = int(time.time() - start_time)
        uptime_str = time.strftime('%H:%M:%S', time.gmtime(uptime_sec))
        
        # Dummy graph data
        new_data_point = {
            "time": uptime_sec,
            "value": 10 + np.sin(uptime_sec * 0.5) * 15 + np.random.rand() * 5,
            "value2": 5 + np.cos(uptime_sec * 0.3) * 10 + np.random.rand() * 5,
            "value3": noise_score / 5.0
        }
        
        current_data = latest_data["data"]
        current_data.append(new_data_point)
        if len(current_data) > 20:
            current_data.pop(0)

        latest_data = {
            "status": "ENGAGING" if is_attack or detections else "NOMINAL",
            "noise_score": round(float(noise_score), 2),
            "detections": detections,
            "systemMetrics": {
                "cpu": 40 + np.random.randint(0, 10),
                "memory": 60 + np.random.randint(0, 5),
                "gpu": 30 + (40 if detections else 0),
                "bandwidth": 85 + np.random.randint(0, 10),
                "latency": 20 + np.random.randint(0, 5),
                "uptime": uptime_str
            },
            "data": current_data
        }

@app.get("/api/python-data")
async def get_python_data():
    return latest_data

@app.get("/api/video_feed")
async def video_feed():
    def gen():
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=video_worker, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=3000)
