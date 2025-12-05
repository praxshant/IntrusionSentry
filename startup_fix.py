import os
import urllib.request

if not os.path.exists('yolov8n.pt'):
    print("[INFO] Downloading YOLOv8 model...")
    try:
        urllib.request.urlretrieve(
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8n.pt'
        )
        print("[INFO] YOLOv8 model downloaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to download YOLOv8: {e}")

# Create screenshots directory
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')
    print("[INFO] Created screenshots directory")
