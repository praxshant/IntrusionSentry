<<<<<<< HEAD
# IntrusionSentry: Intrusion Detection System

IntrusionSentry is a Python-based Intrusion Detection System (IDS) prototype. It uses a USB camera, YOLOv8 for person detection, and a Flask web dashboard to monitor a user-defined zone for entries, exits, and unauthorized access. Events are logged to a SQLite database and screenshots are captured for each event.
=======
# IntrusionSentry SaaS - Hybrid Intrusion Detection System

## Overview
IntrusionSentry is a flexible intrusion detection system supporting BOTH deployment modes:
- **Server Mode**: Enterprise CCTV/USB camera monitoring with advanced, clothing-invariant Re-ID running on a server GPU.
- **Browser Mode**: Personal webcam monitoring with client-side detection using TensorFlow.js.
>>>>>>> 3b85ec0 ( v2.0 - Major Update: Exit Detection, Browser Mode, Action Buttons)

This hybrid architecture lets you choose the best trade-off between accuracy, cost, and privacy, or combine both with optional server-side Re-ID for browser clients.

<<<<<<< HEAD
- **Real-time Detection**: Detects persons in a live video feed using YOLOv8 and optionally person re-identification (Re-ID) with configurable models.
- **Live Video Stream**: MJPEG stream from the camera with zone overlay on the dashboard.
- **Zone Monitoring**: Monitors a polygonal zone defined in `config.json`.
- **Event Logging**: Logs 'entry', 'exit', and 'unauthorized' events to a SQLite database.
- **Screenshot Capture**: Captures screenshots for each event type.
- **Person Enrollment**: Enroll up to 5 persons via the dashboard and track their entry/exit individually.
- **Web Dashboard**: Responsive Flask dashboard with live feed, event table, and visual alerts.
- **GPU Accelerated**: Uses CUDA for faster processing if available.
- **Configurable Re-ID**: Supports model selection, threshold, and crop strategy in `config.json`.
=======
## Architecture

### Server Mode (Enterprise CCTV)
- YOLOv8 person detection on a server (GPU recommended)
- OSNet + `face_recognition` for clothing-invariant Re-ID
- RTSP/IP/USB camera support
- 24/7 cloud/server deployment with Flask dashboard and database

### Browser Mode (Personal Security)
- TensorFlow.js COCO-SSD detection directly in the browser
- Optional server-side Re-ID matching via API
- No installation required; works on any device with a webcam
- Privacy-first: frame processing stays on-device unless Re-ID match is requested

## Quick Start

### Option 1: Docker Deployment (Recommended)
Clone repository
```bash
git clone <repo-url>
cd IntrusionSentry
```

Copy environment template
```bash
cp .env.example .env
```

Edit `.env` to set `MODE=server` or `MODE=browser`

Build and run
```bash
docker-compose up -d
```

Access at http://localhost:5000

### Option 2: Manual Setup
Install dependencies
```bash
pip install -r requirements.txt
```

Configure mode in `config.json`
```json
"mode": "server"  // or "browser" or "hybrid"
```

Run application
```bash
python app.py
```

## Configuration

### `config.json` Structure
```json
{
  "mode": "server",
  "cameras": [
    { "id": 0, "type": "usb", "source": 0 }
  ],
  "zones": [],
  "reid_threshold": 0.60,
  "browser_mode": {
    "enable_reid_matching": true,
    "detection_fps": 5
  }
}
```

- `mode`: "server" or "browser" or "hybrid".
- `cameras`: Server mode only; RTSP/IP/USB camera sources.
- `zones`: Detection zones.
- `reid_threshold`: Similarity threshold for person matching.
- `browser_mode.enable_reid_matching`: If true, browser clients can send frames for server-side Re-ID.
- `browser_mode.detection_fps`: Client-side detection rate to balance CPU and accuracy.

## Features Comparison

| Feature | Server Mode | Browser Mode |
|---------|-------------|--------------|
| Person Detection | YOLOv8 (high accuracy) | TF.js COCO-SSD (good) |
| Re-ID Accuracy | OSNet + face (excellent) | Optional server match |
| Camera Type | CCTV, RTSP, USB | User webcam |
| Processing Location | Cloud/server | User browser |
| GPU Required | Yes (recommended) | No |
| Deployment Cost | $50-200/month | $0-10/month |

## API Endpoints

### Common (Both Modes)
- `GET /status` - System status
- `GET /events` - Event history
- `POST /enroll` - Enroll person
- `POST /report_event` - Log event (browser mode sends to server)

### Server Mode Only
- `GET /video_feed` - MJPEG stream

### Browser Mode Only
- `POST /match_frame` - Re-ID matching

## Deployment Guides

### AWS EC2 with GPU (Server Mode)
- Launch a GPU instance (e.g., g4dn.xlarge) with Ubuntu 22.04.
- Install NVIDIA drivers and Docker + NVIDIA Container Toolkit.
- Set environment in `.env` (e.g., `MODE=server`).
- Expose port 5000 (security group) and run `docker-compose up -d`.
- Point RTSP/USB sources in `config.json`.

### Render.com (Browser Mode)
- Deploy the Flask app as a web service; set `MODE=browser` in environment.
- No GPU required; inexpensive or free tiers are sufficient.
- Browser runs detection locally; server handles status/events and optional Re-ID.

### DigitalOcean Droplet (Server Mode)
- Create a Droplet; install Docker and optionally NVIDIA drivers (for GPU Droplets).
- Set `MODE=server` and configure RTSP/IP cameras in `config.json`.
- Open port 5000 and run with `docker-compose up -d`.
>>>>>>> 3b85ec0 ( v2.0 - Major Update: Exit Detection, Browser Mode, Action Buttons)

## Project Structure

```
IntrusionSentry/
├── app.py                  # Main Flask app with detection and alert logic
├── models.py               # SQLAlchemy models for database
├── config.json             # Camera, zone, and Re-ID configuration
├── static/
│   └── style.css           # Dashboard styling
├── templates/
│   └── index.html          # Web dashboard
├── screenshots/            # Captured screenshots
├── database.db             # SQLite database
├── ultralytics_patch.py    # YOLOv8 patching utilities
├── yolov8n.pt              # YOLOv8 model weights
```

## Prerequisites

- **Python**: 3.10
- **Recommended Environment**: Conda (`torch_gpu`)
- **Dependencies**:
  - torch==2.1.0
  - numpy==1.26.4
  - flask==2.3.3
  - ultralytics==8.0.196
  - opencv-python==4.8.1
  - sqlalchemy==2.0.23

## Setup and Installation

1. Clone/download the repository.
2. Activate your Conda environment:
   ```bash
   conda activate torch_gpu
   ```
3. Navigate to the project directory:
   ```bash
   cd C:\Users\ACER\OneDrive\Desktop\IntrusionSentry
   ```
4. Initialize the database (optional, auto-created by app.py):
   ```bash
   python models.py
   ```

## Configuration

Edit `config.json` to set camera, zone, and Re-ID options:

```json
{
  "cameras": [
    { "id": 0, "type": "usb", "source": 0 }
  ],
  "zones": [
    { "name": "Zone1", "points": [[0,0],[640,0],[640,480],[0,480]] }
  ],
  "reid_threshold": 0.75,
  "reid_model": "osnet_x1_0",
  "reid_crop_strategy": "full_body",
  "entry_debounce_frames": 8,
  "exit_debounce_frames": 12,
  "zone_margin_px": 20
}
```

- **cameras**: List of camera sources.
- **zones**: Polygonal zone coordinates.
- **reid_threshold**: Similarity threshold for person matching.
- **reid_model**: Model name for Re-ID (e.g., osnet_x1_0).
- **reid_crop_strategy**: Crop method for Re-ID.
- **entry/exit_debounce_frames**: Frames required to confirm entry/exit.
- **zone_margin_px**: Margin for zone detection.

## How to Run

1. Connect your USB camera.
2. Run the main application:
   ```bash
   python app.py
   ```
3. Open your browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
4. View the dashboard, live feed, and event table.

## Enrollment & Multi-person Tracking

1. Open the dashboard and go to "Enrollment & Settings".
2. Select number of persons (1–5) and click Apply.
3. For each slot:
   - Enter a label (optional).
   - Have the person stand in the zone.
   - Click "Capture" to enroll.
4. Entry/exit of enrolled persons is detected and logged. Screenshots are saved for each event.

**Notes:**
- Enrollment is in-memory per run (cleared on restart).
- Matching uses HSV color histogram and optionally Re-ID model.
- Lighting and clothing contrast affect accuracy.

## Troubleshooting

<<<<<<< HEAD
- Camera not detected: Check connection and ensure no other app is using it.
- Screenshots not saving: Ensure `screenshots/` exists and is writable.
- GPU acceleration: Requires compatible CUDA device.
=======
### Server Mode
- Camera not detected: Verify USB connection or RTSP URL (test with VLC/ffmpeg). Ensure no other process is using the device.
- Low FPS: Confirm CUDA is active; use GPU builds; lower input resolution or increase detection interval.
- High latency: Deploy near cameras; prefer MJPEG or tuned RTSP; disable heavy debug overlays.

### Browser Mode
- Camera permission denied: Grant site camera access; use HTTPS on the web; localhost is allowed.
- TensorFlow.js slow: Use Chrome/Edge; enable hardware acceleration; reduce `browser_mode.detection_fps`.
- Re-ID match failing: Ensure backend is reachable and `POST /match_frame` enabled; verify CORS and payload size limits.
>>>>>>> 3b85ec0 ( v2.0 - Major Update: Exit Detection, Browser Mode, Action Buttons)

## How It Works

- **app.py**: Main logic, detection loop, Flask server, event logging.
- **models.py**: SQLAlchemy models and DB initialization.
- **index.html**: Dashboard UI, AJAX event updates, enrollment controls.
- **ultralytics_patch.py**: YOLOv8 patching utilities.

## License

<<<<<<< HEAD
- `GET /status` — Returns active slots and enrollment status.
- `POST /set_slots` — Set number of active slots. `{ "count": 1..5 }`
- `POST /enroll` — Enroll a person. `{ "slot": 1..5, "label": "optional" }`
- `GET /events` — Fetch all logged events.
- `POST /reset` — Reset all enrollments and events.
- `GET /video_feed` — MJPEG video stream for dashboard.

## License

This project is for educational and prototyping purposes.
=======
Educational and commercial use permitted.
>>>>>>> 3b85ec0 ( v2.0 - Major Update: Exit Detection, Browser Mode, Action Buttons)
