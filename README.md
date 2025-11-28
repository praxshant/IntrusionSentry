# IntrusionSentry: Intrusion Detection System

IntrusionSentry is a Python-based Intrusion Detection System (IDS) prototype. It uses a USB camera, YOLOv8 for person detection, and a Flask web dashboard to monitor a user-defined zone for entries, exits, and unauthorized access. Events are logged to a SQLite database and screenshots are captured for each event.

## Features

- **Real-time Detection**: Detects persons in a live video feed using YOLOv8 and optionally person re-identification (Re-ID) with configurable models.
- **Live Video Stream**: MJPEG stream from the camera with zone overlay on the dashboard.
- **Zone Monitoring**: Monitors a polygonal zone defined in `config.json`.
- **Event Logging**: Logs 'entry', 'exit', and 'unauthorized' events to a SQLite database.
- **Screenshot Capture**: Captures screenshots for each event type.
- **Person Enrollment**: Enroll up to 5 persons via the dashboard and track their entry/exit individually.
- **Web Dashboard**: Responsive Flask dashboard with live feed, event table, and visual alerts.
- **GPU Accelerated**: Uses CUDA for faster processing if available.
- **Configurable Re-ID**: Supports model selection, threshold, and crop strategy in `config.json`.

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

- Camera not detected: Check connection and ensure no other app is using it.
- Screenshots not saving: Ensure `screenshots/` exists and is writable.
- GPU acceleration: Requires compatible CUDA device.

## How It Works

- **app.py**: Main logic, detection loop, Flask server, event logging.
- **models.py**: SQLAlchemy models and DB initialization.
- **index.html**: Dashboard UI, AJAX event updates, enrollment controls.
- **ultralytics_patch.py**: YOLOv8 patching utilities.

## API Endpoints

- `GET /status` — Returns active slots and enrollment status.
- `POST /set_slots` — Set number of active slots. `{ "count": 1..5 }`
- `POST /enroll` — Enroll a person. `{ "slot": 1..5, "label": "optional" }`
- `GET /events` — Fetch all logged events.
- `POST /reset` — Reset all enrollments and events.
- `GET /video_feed` — MJPEG video stream for dashboard.

## License

This project is for educational and prototyping purposes.
