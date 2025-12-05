<<<<<<< HEAD
import ultralytics_patch
import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'false'

import sys
CLOUD_DEPLOYMENT = not os.path.exists('/dev/video0')  

# Download YOLO model if missing
if not os.path.exists('yolov8n.pt'):
    print("[INFO] Downloading YOLOv8 model...")
    import urllib.request
    try:
        urllib.request.urlretrieve(
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8n.pt'
        )
        print("[INFO] ✅ YOLOv8 downloaded")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)
        
import cv2
import json
import threading
import time
from datetime import datetime
import numpy as np

try:
    import face_recognition
    USE_FACE_RECOGNITION = True
    print("[INFO] ✅ Face recognition enabled (clothing-invariant matching)")
except ImportError:
    USE_FACE_RECOGNITION = False
    print("[WARN] ⚠️ Face recognition not available")

from flask import Flask, jsonify, render_template, send_from_directory, Response, request
from ultralytics import YOLO
from models import Event, SessionLocal, init_db
import torch
import torchreid
from sklearn.metrics.pairwise import cosine_similarity

torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

app = Flask(__name__)
yolo_model = YOLO('yolov8n.pt')

with open('config.json', 'r') as f:
    config = json.load(f)

# Global locks and variables
frame_lock = threading.Lock()
enroll_lock = threading.Lock()
latest_frame = None
current_frame = None
cap = None
active_slots = 5
enrolled_slots = {i: {'label': None, 'embedding': None, 'photo_path': None} for i in range(1, 6)}

zone_config = config['zones'][0]
zone_poly = np.array(zone_config['points'], dtype=np.int32)

# Re-ID initialization
USE_REID = True
try:
    reid_model_name = config.get('reid_model', 'osnet_x1_0')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reid_extractor = torchreid.utils.FeatureExtractor(
        model_name=reid_model_name,
        model_path=None,
        device=device
    )
    if torch.__version__.startswith('2.') and device == 'cuda':
        print(f"[INFO] PyTorch {torch.__version__} detected")
    print(f"[INFO] ✅ Re-ID model loaded: {reid_model_name}, Device: {device}, Threshold: {config.get('reid_threshold', 0.70)}")
except Exception as e:
    print(f"[ERROR] Re-ID failed: {e}")
    USE_REID = False

# Debouncing
ENTRY_FRAMES = config.get('entry_debounce_frames', 8)
EXIT_FRAMES = config.get('exit_debounce_frames', 12)
ZONE_MARGIN = config.get('zone_margin_px', 20)
ABSENCE_GRACE_FRAMES = 5  # FIX 3: Allow 5 consecutive misses before counting toward exit
EXIT_ZONE_MARGIN = 50  # FIX 3: Larger margin for exit detection

slot_present_prev = {i: False for i in range(1, 6)}
present_counts = {i: 0 for i in range(1, 6)}
absent_counts = {i: 0 for i in range(1, 6)}
grace_counters = {i: 0 for i in range(1, 6)}  # FIX 3: Grace period counters
last_entry_time = {i: 0 for i in range(1, 6)}  # FIX 4: Track last event times
last_exit_time = {i: 0 for i in range(1, 6)}  # FIX 4: Track last event times

# Global counters for dashboard
total_entry_events = 0
total_exit_events = 0
total_unauthorized_events = 0

def is_in_zone(box, zone_polygon, margin_px=0):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    test_pt = np.array([[cx, cy]], dtype=np.int32)
    
    if margin_px > 0:
        expanded = zone_polygon.copy().astype(np.float32)
        center = expanded.mean(axis=0)
        expanded = center + (expanded - center) * (1 + margin_px / 100.0)
        expanded = expanded.astype(np.int32)
        return cv2.pointPolygonTest(expanded, (cx, cy), False) >= 0
    
    return cv2.pointPolygonTest(zone_polygon, (cx, cy), False) >= 0

def save_screenshot(frame, event_type, person_label='Unknown', person_slot=None, crop_box=None):
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{event_type}_{timestamp_str}.jpg"
    path = os.path.join('screenshots', filename)
    
    if crop_box is not None:
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(path, crop)
    else:
        cv2.imwrite(path, frame)
    
    return path

def compute_person_histogram(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def extract_person_embedding(frame, box):
    if not USE_REID:
        return None
    
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop_strategy = config.get('reid_crop_strategy', 'full_body')
    height = y2 - y1
    width = x2 - x1
    
    if crop_strategy == 'upper_body':
        mid_y = y1 + height // 2
        crop = frame[y1:mid_y, x1:x2]
    elif crop_strategy == 'upper_torso':
        crop_y = y1 + int(height * 0.6)
        crop = frame[y1:crop_y, x1:x2]
    else:
        crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    target_aspect = 0.5
    current_aspect = crop.shape[1] / crop.shape[0] if crop.shape[0] > 0 else 0
    
    if current_aspect < target_aspect * 0.8:
        target_width = int(crop.shape[0] * target_aspect)
        pad_width = max(0, (target_width - crop.shape[1]) // 2)
        crop = cv2.copyMakeBorder(crop, 0, 0, pad_width, pad_width, cv2.BORDER_REPLICATE)
    
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    body_emb = reid_extractor([crop_rgb])
    
    if hasattr(body_emb, 'detach'):
        body_emb = body_emb.detach().cpu().numpy()
    
    body_emb = np.asarray(body_emb)
    if body_emb.ndim == 1:
        body_emb = body_emb.reshape(1, -1)
    body_emb = body_emb / (np.linalg.norm(body_emb) + 1e-8)
    
    face_emb = None
    if USE_FACE_RECOGNITION:
        try:
            face_locations = face_recognition.face_locations(crop_rgb, model='hog')
            if face_locations:
                face_encodings = face_recognition.face_encodings(crop_rgb, face_locations)
                if face_encodings:
                    face_emb = np.asarray(face_encodings[0]).reshape(1, -1)
                    face_emb = face_emb / (np.linalg.norm(face_emb) + 1e-8)
        except Exception as e:
            print(f"[WARN] Face extraction failed: {e}")
    
    return {'body': body_emb, 'face': face_emb}

def match_slot_for_box(frame, box):
    with enroll_lock:
        candidates = {k: v for k, v in enrolled_slots.items() if 1 <= k <= active_slots and v['embedding'] is not None}
    
    if not candidates:
        return None, 0.0
    
    if USE_REID:
        embedding = extract_person_embedding(frame, box)
        if embedding is None:
            return None, 0.0
        
        best_slot, best_score = None, -1.0
        
        for slot, data in candidates.items():
            ref_emb = data['embedding']
            
            if isinstance(ref_emb, dict):
                ref_body = ref_emb.get('body')
                ref_face = ref_emb.get('face')
            else:
                ref_body = np.asarray(ref_emb)
                if ref_body.ndim == 1:
                    ref_body = ref_body.reshape(1, -1)
                ref_face = None
            
            body_score = cosine_similarity(embedding['body'], ref_body)[0][0]
            face_score = 0.0
            has_face_match = False
            
            if embedding.get('face') is not None and ref_face is not None:
                face_score = cosine_similarity(embedding['face'], ref_face)[0][0]
                has_face_match = True
            
            combined_score = 0.6 * body_score + 0.4 * face_score if has_face_match else body_score
            
            if has_face_match:
                print(f"[DEBUG] Slot {slot}: body={body_score:.3f}, face={face_score:.3f}, combined={combined_score:.3f}")
            else:
                print(f"[DEBUG] Slot {slot}: body_only={body_score:.3f}")
            
            if combined_score > best_score:
                best_slot, best_score = slot, combined_score
        
        reid_threshold = config.get('reid_threshold', 0.60)
        if best_score >= reid_threshold:
            print(f"[DEBUG] ✅ MATCHED Slot {best_slot} (score={best_score:.3f})")
            return (best_slot, float(best_score))
        else:
            print(f"[DEBUG] ❌ NO MATCH (best={best_score:.3f} < {reid_threshold})")
            return (None, float(best_score))
    else:
        hist = compute_person_histogram(frame, box)
        if hist is None:
            return None, 0.0
        
        best_slot, best_score = None, -1.0
        for slot, data in candidates.items():
            if data.get('hist') is None:
                continue
            score = cv2.compareHist(hist, data['hist'], cv2.HISTCMP_CORREL)
            if score > best_score:
                best_slot, best_score = slot, score
        
        return (best_slot, float(best_score)) if best_score >= 0.6 else (None, float(best_score))

def enhance_for_enrollment(frame, box):
    """
    Enhance image quality for enrollment in low-light/blur conditions.
    Returns enhanced crop or None.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    # Denoise
    crop_enhanced = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    crop_enhanced = cv2.filter2D(crop_enhanced, -1, kernel)
    
    # Contrast improve via CLAHE on L channel
    lab = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    crop_enhanced = cv2.merge([l, a, b])
    crop_enhanced = cv2.cvtColor(crop_enhanced, cv2.COLOR_LAB2BGR)
    
    return crop_enhanced

def check_enrollment_quality(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return False, "Invalid crop"
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    
    if brightness < 30:
        return False, "❌ Too dark"
    if brightness > 220:
        return False, "❌ Too bright"
    
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 15:  # Reduced from 30 to 15
        return False, "❌ Extremely blurry - try again"
    
    height = y2 - y1
    width = x2 - x1
    if height < 60 or width < 30:
        return False, "❌ Too small - move closer"
    
    # Face detection is OPTIONAL - does not block enrollment
    if USE_FACE_RECOGNITION:
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(crop_rgb)
            if not faces:
                print("[INFO] No face detected - will use body Re-ID only")
            else:
                face_height = faces[0][2] - faces[0][0]
                if face_height < height * 0.15:
                    print("[WARN] Small face detected but allowing enrollment")
        except Exception as e:
            print(f"[INFO] Face check skipped: {e}")
    
    contrast = float(np.std(gray))
    if contrast < 30:
        return False, "❌ Low contrast"
    
    return True, "✅ Quality OK"

def frame_grabber():
    global latest_frame, cap
    camera_config = config['cameras'][0]
    source = camera_config['source']
    
    print(f"[INFO] Opening camera source: {source}")
    
    # Cloud deployment check
    if CLOUD_DEPLOYMENT:
        print("[WARN] ⚠️ Cloud environment detected - using dummy frames")
        while True:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Cloud Deployment - No Camera", (100, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(dummy_frame, "Upload images or use local deployment", (80, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            with frame_lock:
                latest_frame = dummy_frame.copy()
            time.sleep(0.1)
        return
    
    # Local camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        return
    
    print(f"[INFO] ✅ Camera opened")
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.1)
            cap.release()
            cap = cv2.VideoCapture(source)
        time.sleep(0.01)

def detection_loop():
    global total_entry_events, total_exit_events, total_unauthorized_events
    
    db_session = SessionLocal()
    last_unauth_time = 0.0
    UNAUTH_COOLDOWN = 4.0
    frame_counter = 0
    
    while True:
        try:
            frame_counter += 1
            
            # FIX 3: Enhanced status logging
            if frame_counter % 90 == 0:
                with enroll_lock:
                    enrolled = {i: enrolled_slots[i]['label'] for i in range(1, 6) if enrolled_slots[i]['label']}
                print(f"[STATUS] Frame {frame_counter} | Enrolled: {enrolled}")
                print(f"[STATUS] Absence counts: {absent_counts}")
                print(f"[STATUS] Grace counters: {grace_counters}")
            
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            results = yolo_model(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
            
            slots_now = {i: False for i in range(1, 6)}
            unauthorized_boxes = []
            
            # FIX 3: Dual-zone detection (main zone + exit zone)
            for box in boxes:
                in_main_zone = is_in_zone(box, zone_poly, margin_px=ZONE_MARGIN)
                in_exit_zone = is_in_zone(box, zone_poly, margin_px=EXIT_ZONE_MARGIN)
                
                if not in_exit_zone:
                    continue  # Person completely outside monitoring area
                
                best_slot, score = match_slot_for_box(frame, box)
                
                if not in_main_zone:
                    # Person in exit zone but not main zone - mark as potentially exiting
                    if best_slot:
                        slots_now[best_slot] = False  # Allow exit counting to proceed
                    continue
                
                if best_slot is None:
                    unauthorized_boxes.append((box, float(score)))
                else:
                    slots_now[best_slot] = True
            
            with enroll_lock:
                # FIX 1: Only monitor ENROLLED slots, not all active slots
                monitored = [i for i in range(1, active_slots + 1) if enrolled_slots[i]['label'] is not None]
                labels = {i: enrolled_slots[i]['label'] for i in monitored}
            
            for slot in monitored:
                now = slots_now[slot]
                prev = slot_present_prev[slot]
                
                # FIX 3: Hysteresis logic with grace period
                if now:
                    present_counts[slot] = min(ENTRY_FRAMES, present_counts[slot] + 1)
                    absent_counts[slot] = 0
                    grace_counters[slot] = 0  # Reset grace period
                else:
                    grace_counters[slot] += 1
                    if grace_counters[slot] >= ABSENCE_GRACE_FRAMES:
                        absent_counts[slot] = min(EXIT_FRAMES, absent_counts[slot] + 1)
                        present_counts[slot] = 0
                    # Otherwise keep current counts (don't increment/decrement)
                
                event_type = None
                if not prev and present_counts[slot] >= ENTRY_FRAMES:
                    event_type = 'entry'
                    slot_present_prev[slot] = True
                    total_entry_events += 1
                elif prev and absent_counts[slot] >= EXIT_FRAMES:
                    event_type = 'exit'
                    slot_present_prev[slot] = False
                    total_exit_events += 1
                    # FIX 3: Exit debug logging
                    print(f"[EXIT-DEBUG] Slot {slot}: absent_counts={absent_counts[slot]}, EXIT_FRAMES={EXIT_FRAMES}")
                
                if event_type:
                    label = labels.get(slot) or f"Person{slot}"
                    screenshot = save_screenshot(frame, event_type, person_label=label, person_slot=slot)
                    
                    new_event = Event(
                        event_type=event_type,
                        zone_name=zone_config['name'],
                        screenshot_path=screenshot,
                        person_slot=slot,
                        person_label=label,
                    )
                    db_session.add(new_event)
                    db_session.commit()
                    print(f"[EVENT] {event_type.upper()} - Slot {slot} ({label})")
            
            if unauthorized_boxes:
                now_ts = time.time()
                if now_ts - last_unauth_time >= UNAUTH_COOLDOWN:
                    for idx, (ubox, score) in enumerate(unauthorized_boxes):
                        screenshot = save_screenshot(frame, 'unauthorized', person_label='Unknown', crop_box=ubox)
                        new_event = Event(
                            event_type='unauthorized',
                            zone_name=zone_config['name'],
                            screenshot_path=screenshot,
                            person_slot=None,
                            person_label='Unknown',
                        )
                        db_session.add(new_event)
                        db_session.commit()
                    
                    total_unauthorized_events += len(unauthorized_boxes)
                    last_unauth_time = now_ts
                    print(f"[EVENT] UNAUTHORIZED - {len(unauthorized_boxes)} person(s)")
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"[ERROR] detection_loop: {e}")
            time.sleep(0.05)
    
    db_session.close()

def gen_frames():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        
        if frame is None:
            time.sleep(0.05)
            continue
        
        frame_display = frame.copy()
        
        # Draw main zone (green)
        cv2.polylines(frame_display, [zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # FIX 3: Visualize exit zone (orange)
        exit_poly = zone_poly.copy().astype(np.float32)
        center = exit_poly.mean(axis=0)
        exit_poly = center + (exit_poly - center) * 1.5  # 50% larger
        cv2.polylines(frame_display, [exit_poly.astype(np.int32)], isClosed=True, color=(0, 165, 255), thickness=1)
        
        _, buffer = cv2.imencode('.jpg', frame_display)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def get_status():
    """Dashboard status endpoint for real-time updates"""
    with enroll_lock:
        enrolled_list = [
            {
                'slot': i,
                'label': enrolled_slots[i]['label'],
                'present': slot_present_prev[i]
            }
            for i in range(1, 6) if enrolled_slots[i]['label']
        ]
    
    return jsonify({
        'total_enrolled': len(enrolled_list),
        'enrolled_persons': enrolled_list,
        'entry_events': total_entry_events,
        'exit_events': total_exit_events,
        'unauthorized_events': total_unauthorized_events,
        'active_slots': active_slots
    })

@app.route('/set_slots', methods=['POST'])
def set_slots():
    """Set the number of active monitoring slots (1-5)"""
    global active_slots
    
    body = request.get_json(silent=True) or {}
    new_slots = int(body.get('slots', 5))
    
    if not (1 <= new_slots <= 5):
        return jsonify(success=False, error='Slots must be between 1 and 5'), 400
    
    active_slots = new_slots
    print(f"[INFO] Active slots set to {active_slots}")
    
    return jsonify(success=True, active_slots=active_slots)

@app.route('/events')
def get_events():
    db_session = SessionLocal()
    try:
        events = db_session.query(Event).order_by(Event.id.desc()).limit(50).all()
        return jsonify([{
            'id': e.id,
            'event_type': e.event_type,
            'zone_name': e.zone_name,
            'timestamp': e.timestamp.isoformat(),
            'screenshot_path': e.screenshot_path,
            'person_slot': e.person_slot,
            'person_label': e.person_label,
        } for e in events])
    finally:
        db_session.close()

@app.route('/enrolled_persons')
def get_enrolled():
    with enroll_lock:
        return jsonify({
            'active_slots': active_slots,
            'slots': [{
                'slot': i,
                'label': enrolled_slots[i]['label'],
                'enrolled': enrolled_slots[i]['label'] is not None
            } for i in range(1, 6)]
        })

@app.route('/enroll', methods=['POST'])
def enroll():
    body = request.get_json(silent=True) or {}
    slot = int(body.get('slot', 1))
    label = body.get('label') or f"Person{slot}"
    
    if not (1 <= slot <= 5):
        return jsonify(success=False, error='Invalid slot'), 400
    
    # FIX 2: Improved enrollment validation with better feedback
    with enroll_lock:
        if enrolled_slots[slot]['label'] is not None:
            existing_label = enrolled_slots[slot]['label']
            print(f"[WARN] Slot {slot} occupied by {existing_label} - blocking enrollment")
            return jsonify(
                success=False,
                error=f'Slot {slot} occupied by "{existing_label}". Use Remove Person first.',
                already_enrolled=True
            ), 400
    
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    
    if frame is None:
        return jsonify(success=False, error='Camera not ready'), 503
    
    results = yolo_model(frame, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
    
    # FIX 2: Single person validation
    zone_boxes = [b for b in boxes if is_in_zone(b, zone_poly, margin_px=ZONE_MARGIN)]
    
    if len(zone_boxes) == 0:
        return jsonify(success=False, error='No person in monitored zone'), 400
    
    if len(zone_boxes) > 1:
        return jsonify(success=False, error=f'{len(zone_boxes)} persons detected. Ensure only one person is in frame'), 400
    
    target_box = zone_boxes[0]
    
    quality_ok, quality_msg = check_enrollment_quality(frame, target_box)
    if not quality_ok:
        return jsonify(success=False, error=quality_msg), 400
    
    if USE_REID:
        # Try normal extraction first
        embedding = extract_person_embedding(frame, target_box)
        
        # If failed, try enhanced crop
        if embedding is None:
            print("[INFO] Normal extraction failed, trying enhanced image...")
            enhanced_crop = enhance_for_enrollment(frame, target_box)
            if enhanced_crop is not None:
                x1, y1, x2, y2 = [int(v) for v in target_box]
                frame[y1:y2, x1:x2] = enhanced_crop
                embedding = extract_person_embedding(frame, target_box)
        
        if embedding is None:
            return jsonify(success=False, error='Image too poor - retry with better lighting'), 500
        
        screenshot = save_screenshot(frame, 'enrollment', person_label=label, person_slot=slot, crop_box=target_box)
        
        with enroll_lock:
            enrolled_slots[slot]['embedding'] = embedding
            enrolled_slots[slot]['label'] = label
            enrolled_slots[slot]['photo_path'] = screenshot
            # FIX 4: Reset tracking state on enrollment
            slot_present_prev[slot] = False
            present_counts[slot] = 0
            absent_counts[slot] = 0
            grace_counters[slot] = 0
            last_entry_time[slot] = 0
            last_exit_time[slot] = 0
        
        print(f"[INFO] ✅ Enrolled Slot {slot} ({label})")
    else:
        hist = compute_person_histogram(frame, target_box)
        if hist is None:
            return jsonify(success=False, error='Failed to compute histogram'), 500
        
        with enroll_lock:
            enrolled_slots[slot]['hist'] = hist
            enrolled_slots[slot]['label'] = label
    
    return jsonify(success=True, slot=slot, label=label)

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_from_directory('screenshots', filename)

@app.route('/enrollment_photo/<int:slot>')
def enrollment_photo(slot):
    if not (1 <= slot <= 5):
        return jsonify(error='Invalid slot'), 400
    
    with enroll_lock:
        slot_data = enrolled_slots.get(slot, {})
        label = slot_data.get('label')
        photo_path = slot_data.get('photo_path')
    
    if not label:
        return jsonify(error='Not enrolled'), 404
    
    if photo_path and os.path.exists(photo_path):
        dir_path, file_name = os.path.split(photo_path)
        return send_from_directory(dir_path, file_name)
    
    db_session = SessionLocal()
    try:
        event = db_session.query(Event).filter(
            Event.person_slot == slot,
            Event.screenshot_path.isnot(None)
        ).order_by(Event.timestamp.desc()).first()
        
        if event and event.screenshot_path:
            dir_path, file_name = os.path.split(event.screenshot_path)
            return send_from_directory(dir_path, file_name)
    finally:
        db_session.close()
    
    return jsonify(error='No photo'), 404

@app.route('/remove_person', methods=['POST'])
def remove_person():
    body = request.get_json(silent=True) or {}
    slot = int(body.get('slot', 0))
    
    if not (1 <= slot <= 5):
        return jsonify(success=False, error='Invalid slot'), 400
    
    with enroll_lock:
        if enrolled_slots[slot]['embedding'] is None:
            return jsonify(success=False, error='Slot not enrolled'), 400
        
        enrolled_slots[slot]['embedding'] = None
        enrolled_slots[slot]['label'] = None
        enrolled_slots[slot]['photo_path'] = None
        # FIX 4: Reset tracking state on removal
        slot_present_prev[slot] = False
        present_counts[slot] = 0
        absent_counts[slot] = 0
        grace_counters[slot] = 0
        last_entry_time[slot] = 0
        last_exit_time[slot] = 0
    
    print(f"[INFO] Removed Slot {slot}")
    return jsonify(success=True, slot=slot)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_unauthorized', methods=['POST'])
def clear_unauthorized():
    """Clear only unauthorized events, keep entry/exit history."""
    global total_unauthorized_events
    
    db_session = SessionLocal()
    try:
        deleted_count = db_session.query(Event).filter(
            Event.event_type == 'unauthorized'
        ).delete()
        db_session.commit()
        
        total_unauthorized_events = 0
        print(f"[INFO] Cleared {deleted_count} unauthorized events")
        return jsonify(success=True, cleared=deleted_count)
    finally:
        db_session.close()

@app.route('/reset', methods=['POST'])
def reset():
    global total_entry_events, total_exit_events, total_unauthorized_events
    
    db_session = SessionLocal()
    try:
        db_session.query(Event).delete()
        db_session.commit()
    finally:
        db_session.close()
    
    # Reset counters
    total_entry_events = 0
    total_exit_events = 0
    total_unauthorized_events = 0
    
    shots_dir = os.path.join(os.getcwd(), 'screenshots')
    try:
        with enroll_lock:
            enrollment_photos = set()
            for i in range(1, 6):
                photo_path = enrolled_slots[i].get('photo_path')
                if photo_path:
                    enrollment_photos.add(os.path.basename(photo_path))
        
        for name in os.listdir(shots_dir):
            if name not in enrollment_photos:
                path = os.path.join(shots_dir, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"[WARN] Failed to remove {path}: {e}")
    except FileNotFoundError:
        pass
    
    print(f"[INFO] ✅ Reset complete")
    return jsonify(success=True)

if __name__ == '__main__':
    init_db()
    
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')
    
    grabber_thread = threading.Thread(target=frame_grabber, daemon=True)
    grabber_thread.start()
    
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask server on port {port}")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


=======
import ultralytics_patch
import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'false'
import cv2
import json
import threading
import time
from datetime import datetime
import numpy as np
import base64

try:
    import face_recognition
    USE_FACE_RECOGNITION = True
    print("[INFO] ✅ Face recognition enabled (clothing-invariant matching)")
except ImportError:
    USE_FACE_RECOGNITION = False
    print("[WARN] ⚠️ Face recognition not available")

from flask import Flask, jsonify, render_template, send_from_directory, Response, request
from ultralytics import YOLO
from models import Event, SessionLocal, init_db
import torch
import torchreid
from sklearn.metrics.pairwise import cosine_similarity

torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

app = Flask(__name__)
yolo_model = YOLO('yolov8n.pt')

with open('config.json', 'r') as f:
    config = json.load(f)

# Detect application mode
app_mode = config.get('mode', 'server')

# Global locks and variables
frame_lock = threading.Lock()
enroll_lock = threading.Lock()
latest_frame = None
current_frame = None
cap = None
active_slots = 5
enrolled_slots = {i: {'label': None, 'embedding': None, 'photo_path': None} for i in range(1, 6)}

zone_config = config['zones'][0]
zone_poly = np.array(zone_config['points'], dtype=np.int32)

# Re-ID initialization
USE_REID = True
try:
    reid_model_name = config.get('reid_model', 'osnet_x1_0')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reid_extractor = torchreid.utils.FeatureExtractor(
        model_name=reid_model_name,
        model_path=None,
        device=device
    )
    if torch.__version__.startswith('2.') and device == 'cuda':
        print(f"[INFO] PyTorch {torch.__version__} detected")
    print(f"[INFO] ✅ Re-ID model loaded: {reid_model_name}, Device: {device}, Threshold: {config.get('reid_threshold', 0.70)}")
except Exception as e:
    print(f"[ERROR] Re-ID failed: {e}")
    USE_REID = False

# Debouncing
ENTRY_FRAMES = config.get('entry_debounce_frames', 8)
EXIT_FRAMES = config.get('exit_debounce_frames', 12)
ZONE_MARGIN = config.get('zone_margin_px', 20)
ABSENCE_GRACE_FRAMES = 5  # FIX 3: Allow 5 consecutive misses before counting toward exit
EXIT_ZONE_MARGIN = 50  # FIX 3: Larger margin for exit detection

slot_present_prev = {i: False for i in range(1, 6)}
present_counts = {i: 0 for i in range(1, 6)}
absent_counts = {i: 0 for i in range(1, 6)}
grace_counters = {i: 0 for i in range(1, 6)}  # FIX 3: Grace period counters
last_entry_time = {i: 0 for i in range(1, 6)}  # FIX 4: Track last event times
last_exit_time = {i: 0 for i in range(1, 6)}  # FIX 4: Track last event times

# Global counters for dashboard
total_entry_events = 0
total_exit_events = 0
total_unauthorized_events = 0

def is_in_zone(box, zone_polygon, margin_px=0):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    test_pt = np.array([[cx, cy]], dtype=np.int32)
    
    if margin_px > 0:
        expanded = zone_polygon.copy().astype(np.float32)
        center = expanded.mean(axis=0)
        expanded = center + (expanded - center) * (1 + margin_px / 100.0)
        expanded = expanded.astype(np.int32)
        return cv2.pointPolygonTest(expanded, (cx, cy), False) >= 0
    
    return cv2.pointPolygonTest(zone_polygon, (cx, cy), False) >= 0

def save_screenshot(frame, event_type, person_label='Unknown', person_slot=None, crop_box=None):
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{event_type}_{timestamp_str}.jpg"
    path = os.path.join('screenshots', filename)
    
    if crop_box is not None:
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(path, crop)
    else:
        cv2.imwrite(path, frame)
    
    return path

def compute_person_histogram(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def extract_person_embedding(frame, box):
    if not USE_REID:
        return None
    
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop_strategy = config.get('reid_crop_strategy', 'full_body')
    height = y2 - y1
    width = x2 - x1
    
    if crop_strategy == 'upper_body':
        mid_y = y1 + height // 2
        crop = frame[y1:mid_y, x1:x2]
    elif crop_strategy == 'upper_torso':
        crop_y = y1 + int(height * 0.6)
        crop = frame[y1:crop_y, x1:x2]
    else:
        crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    target_aspect = 0.5
    current_aspect = crop.shape[1] / crop.shape[0] if crop.shape[0] > 0 else 0
    
    if current_aspect < target_aspect * 0.8:
        target_width = int(crop.shape[0] * target_aspect)
        pad_width = max(0, (target_width - crop.shape[1]) // 2)
        crop = cv2.copyMakeBorder(crop, 0, 0, pad_width, pad_width, cv2.BORDER_REPLICATE)
    
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    body_emb = reid_extractor([crop_rgb])
    
    if hasattr(body_emb, 'detach'):
        body_emb = body_emb.detach().cpu().numpy()
    
    body_emb = np.asarray(body_emb)
    if body_emb.ndim == 1:
        body_emb = body_emb.reshape(1, -1)
    body_emb = body_emb / (np.linalg.norm(body_emb) + 1e-8)
    
    face_emb = None
    if USE_FACE_RECOGNITION:
        try:
            face_locations = face_recognition.face_locations(crop_rgb, model='hog')
            if face_locations:
                face_encodings = face_recognition.face_encodings(crop_rgb, face_locations)
                if face_encodings:
                    face_emb = np.asarray(face_encodings[0]).reshape(1, -1)
                    face_emb = face_emb / (np.linalg.norm(face_emb) + 1e-8)
        except Exception as e:
            print(f"[WARN] Face extraction failed: {e}")
    
    return {'body': body_emb, 'face': face_emb}

def match_slot_for_box(frame, box):
    with enroll_lock:
        candidates = {k: v for k, v in enrolled_slots.items() if 1 <= k <= active_slots and v['embedding'] is not None}
    
    if not candidates:
        return None, 0.0
    
    if USE_REID:
        embedding = extract_person_embedding(frame, box)
        if embedding is None:
            return None, 0.0
        
        best_slot, best_score = None, -1.0
        
        for slot, data in candidates.items():
            ref_emb = data['embedding']
            
            if isinstance(ref_emb, dict):
                ref_body = ref_emb.get('body')
                ref_face = ref_emb.get('face')
            else:
                ref_body = np.asarray(ref_emb)
                if ref_body.ndim == 1:
                    ref_body = ref_body.reshape(1, -1)
                ref_face = None
            
            body_score = cosine_similarity(embedding['body'], ref_body)[0][0]
            face_score = 0.0
            has_face_match = False
            
            if embedding.get('face') is not None and ref_face is not None:
                face_score = cosine_similarity(embedding['face'], ref_face)[0][0]
                has_face_match = True
            
            combined_score = 0.6 * body_score + 0.4 * face_score if has_face_match else body_score
            
            if has_face_match:
                print(f"[DEBUG] Slot {slot}: body={body_score:.3f}, face={face_score:.3f}, combined={combined_score:.3f}")
            else:
                print(f"[DEBUG] Slot {slot}: body_only={body_score:.3f}")
            
            if combined_score > best_score:
                best_slot, best_score = slot, combined_score
        
        reid_threshold = config.get('reid_threshold', 0.60)
        if best_score >= reid_threshold:
            print(f"[DEBUG] ✅ MATCHED Slot {best_slot} (score={best_score:.3f})")
            return (best_slot, float(best_score))
        else:
            print(f"[DEBUG] ❌ NO MATCH (best={best_score:.3f} < {reid_threshold})")
            return (None, float(best_score))
    else:
        hist = compute_person_histogram(frame, box)
        if hist is None:
            return None, 0.0
        
        best_slot, best_score = None, -1.0
        for slot, data in candidates.items():
            if data.get('hist') is None:
                continue
            score = cv2.compareHist(hist, data['hist'], cv2.HISTCMP_CORREL)
            if score > best_score:
                best_slot, best_score = slot, score
        
        return (best_slot, float(best_score)) if best_score >= 0.6 else (None, float(best_score))

def enhance_for_enrollment(frame, box):
    """
    Enhance image quality for enrollment in low-light/blur conditions.
    Returns enhanced crop or None.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    # Denoise
    crop_enhanced = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    crop_enhanced = cv2.filter2D(crop_enhanced, -1, kernel)
    
    # Contrast improve via CLAHE on L channel
    lab = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    crop_enhanced = cv2.merge([l, a, b])
    crop_enhanced = cv2.cvtColor(crop_enhanced, cv2.COLOR_LAB2BGR)
    
    return crop_enhanced

def check_enrollment_quality(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return False, "Invalid crop"
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    
    if brightness < 30:
        return False, "❌ Too dark"
    if brightness > 220:
        return False, "❌ Too bright"
    
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 15:  # Reduced from 30 to 15
        return False, "❌ Extremely blurry - try again"
    
    height = y2 - y1
    width = x2 - x1
    if height < 60 or width < 30:
        return False, "❌ Too small - move closer"
    
    # Face detection is OPTIONAL - does not block enrollment
    if USE_FACE_RECOGNITION:
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(crop_rgb)
            if not faces:
                print("[INFO] No face detected - will use body Re-ID only")
            else:
                face_height = faces[0][2] - faces[0][0]
                if face_height < height * 0.15:
                    print("[WARN] Small face detected but allowing enrollment")
        except Exception as e:
            print(f"[INFO] Face check skipped: {e}")
    
    contrast = float(np.std(gray))
    if contrast < 30:
        return False, "❌ Low contrast"
    
    return True, "✅ Quality OK"

def frame_grabber():
    global latest_frame, cap
    camera_config = config['cameras'][0]
    source = camera_config['source']
    
    print(f"[INFO] Opening camera: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera")
        return
    
    print(f"[INFO] ✅ Camera opened")
    
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.1)
            cap.release()
            cap = cv2.VideoCapture(source)
        time.sleep(0.01)

def detection_loop():
    global total_entry_events, total_exit_events, total_unauthorized_events
    
    db_session = SessionLocal()
    last_unauth_time = 0.0
    UNAUTH_COOLDOWN = 4.0
    frame_counter = 0
    
    while True:
        try:
            frame_counter += 1
            
            # FIX 3: Enhanced status logging
            if frame_counter % 90 == 0:
                with enroll_lock:
                    enrolled = {i: enrolled_slots[i]['label'] for i in range(1, 6) if enrolled_slots[i]['label']}
                print(f"[STATUS] Frame {frame_counter} | Enrolled: {enrolled}")
                print(f"[STATUS] Absence counts: {absent_counts}")
                print(f"[STATUS] Grace counters: {grace_counters}")
            
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            results = yolo_model(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
            
            slots_now = {i: False for i in range(1, 6)}
            unauthorized_boxes = []
            
            # FIX 3: Dual-zone detection (main zone + exit zone)
            for box in boxes:
                in_main_zone = is_in_zone(box, zone_poly, margin_px=ZONE_MARGIN)
                in_exit_zone = is_in_zone(box, zone_poly, margin_px=EXIT_ZONE_MARGIN)
                
                if not in_exit_zone:
                    continue  # Person completely outside monitoring area
                
                best_slot, score = match_slot_for_box(frame, box)
                
                if not in_main_zone:
                    # Person in exit zone but not main zone - mark as potentially exiting
                    if best_slot:
                        slots_now[best_slot] = False  # Allow exit counting to proceed
                    continue
                
                if best_slot is None:
                    unauthorized_boxes.append((box, float(score)))
                else:
                    slots_now[best_slot] = True
            
            with enroll_lock:
                # FIX 1: Only monitor ENROLLED slots, not all active slots
                monitored = [i for i in range(1, active_slots + 1) if enrolled_slots[i]['label'] is not None]
                labels = {i: enrolled_slots[i]['label'] for i in monitored}
            
            for slot in monitored:
                now = slots_now[slot]
                prev = slot_present_prev[slot]
                
                # FIX 3: Hysteresis logic with grace period
                if now:
                    present_counts[slot] = min(ENTRY_FRAMES, present_counts[slot] + 1)
                    absent_counts[slot] = 0
                    grace_counters[slot] = 0  # Reset grace period
                else:
                    grace_counters[slot] += 1
                    if grace_counters[slot] >= ABSENCE_GRACE_FRAMES:
                        absent_counts[slot] = min(EXIT_FRAMES, absent_counts[slot] + 1)
                        present_counts[slot] = 0
                    # Otherwise keep current counts (don't increment/decrement)
                
                event_type = None
                if not prev and present_counts[slot] >= ENTRY_FRAMES:
                    event_type = 'entry'
                    slot_present_prev[slot] = True
                    total_entry_events += 1
                elif prev and absent_counts[slot] >= EXIT_FRAMES:
                    event_type = 'exit'
                    slot_present_prev[slot] = False
                    total_exit_events += 1
                    # FIX 3: Exit debug logging
                    print(f"[EXIT-DEBUG] Slot {slot}: absent_counts={absent_counts[slot]}, EXIT_FRAMES={EXIT_FRAMES}")
                
                if event_type:
                    label = labels.get(slot) or f"Person{slot}"
                    screenshot = save_screenshot(frame, event_type, person_label=label, person_slot=slot)
                    
                    new_event = Event(
                        event_type=event_type,
                        zone_name=zone_config['name'],
                        screenshot_path=screenshot,
                        person_slot=slot,
                        person_label=label,
                    )
                    db_session.add(new_event)
                    db_session.commit()
                    print(f"[EVENT] {event_type.upper()} - Slot {slot} ({label})")
            
            if unauthorized_boxes:
                now_ts = time.time()
                if now_ts - last_unauth_time >= UNAUTH_COOLDOWN:
                    for idx, (ubox, score) in enumerate(unauthorized_boxes):
                        screenshot = save_screenshot(frame, 'unauthorized', person_label='Unknown', crop_box=ubox)
                        new_event = Event(
                            event_type='unauthorized',
                            zone_name=zone_config['name'],
                            screenshot_path=screenshot,
                            person_slot=None,
                            person_label='Unknown',
                        )
                        db_session.add(new_event)
                        db_session.commit()
                    
                    total_unauthorized_events += len(unauthorized_boxes)
                    last_unauth_time = now_ts
                    print(f"[EVENT] UNAUTHORIZED - {len(unauthorized_boxes)} person(s)")
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"[ERROR] detection_loop: {e}")
            time.sleep(0.05)
    
    db_session.close()

def gen_frames():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        
        if frame is None:
            time.sleep(0.05)
            continue
        
        frame_display = frame.copy()
        
        # Draw main zone (green)
        cv2.polylines(frame_display, [zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # FIX 3: Visualize exit zone (orange)
        exit_poly = zone_poly.copy().astype(np.float32)
        center = exit_poly.mean(axis=0)
        exit_poly = center + (exit_poly - center) * 1.5  # 50% larger
        cv2.polylines(frame_display, [exit_poly.astype(np.int32)], isClosed=True, color=(0, 165, 255), thickness=1)
        
        _, buffer = cv2.imencode('.jpg', frame_display)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    if app_mode == 'hybrid':
        return render_template('mode_selector.html')
    elif app_mode == 'browser':
        return render_template('browser.html')
    else:
        return render_template('index.html')

@app.route('/server')
def server_mode():
    return render_template('index.html')

@app.route('/browser')
def browser_mode():
    return render_template('browser.html')

@app.route('/status')
def get_status():
    """Dashboard status endpoint for real-time updates"""
    with enroll_lock:
        enrolled_list = [
            {
                'slot': i,
                'label': enrolled_slots[i]['label'],
                'present': slot_present_prev[i]
            }
            for i in range(1, 6) if enrolled_slots[i]['label']
        ]
    
    return jsonify({
        'total_enrolled': len(enrolled_list),
        'enrolled_persons': enrolled_list,
        'entry_events': total_entry_events,
        'exit_events': total_exit_events,
        'unauthorized_events': total_unauthorized_events,
        'active_slots': active_slots
    })

@app.route('/set_slots', methods=['POST'])
def set_slots():
    """Set the number of active monitoring slots (1-5)"""
    global active_slots
    
    body = request.get_json(silent=True) or {}
    new_slots = int(body.get('slots', 5))
    
    if not (1 <= new_slots <= 5):
        return jsonify(success=False, error='Slots must be between 1 and 5'), 400
    
    active_slots = new_slots
    print(f"[INFO] Active slots set to {active_slots}")
    
    return jsonify(success=True, active_slots=active_slots)

@app.route('/events')
def get_events():
    db_session = SessionLocal()
    try:
        events = db_session.query(Event).order_by(Event.id.desc()).limit(50).all()
        return jsonify([{
            'id': e.id,
            'event_type': e.event_type,
            'zone_name': e.zone_name,
            'timestamp': e.timestamp.isoformat(),
            'screenshot_path': e.screenshot_path,
            'person_slot': e.person_slot,
            'person_label': e.person_label,
        } for e in events])
    finally:
        db_session.close()

@app.route('/enrolled_persons')
def get_enrolled():
    with enroll_lock:
        return jsonify({
            'active_slots': active_slots,
            'slots': [{
                'slot': i,
                'label': enrolled_slots[i]['label'],
                'enrolled': enrolled_slots[i]['label'] is not None
            } for i in range(1, 6)]
        })

@app.route('/enroll', methods=['POST'])
def enroll():
    body = request.get_json(silent=True) or {}
    slot = int(body.get('slot', 1))
    label = body.get('label') or f"Person{slot}"
    mode = body.get('mode', 'server')  # Check if browser or server mode
    
    if not (1 <= slot <= 5):
        return jsonify(success=False, error='Invalid slot'), 400
    
    # FIX 2: Improved enrollment validation with better feedback
    with enroll_lock:
        if enrolled_slots[slot]['label'] is not None:
            existing_label = enrolled_slots[slot]['label']
            print(f"[WARN] Slot {slot} occupied by {existing_label} - blocking enrollment")
            return jsonify(
                success=False,
                error=f'Slot {slot} occupied by "{existing_label}". Use Remove Person first.',
                already_enrolled=True
            ), 400
    
    # BROWSER MODE: Decode image from request
    if mode == 'browser':
        image_b64 = body.get('image')
        if not image_b64:
            return jsonify(success=False, error='No image provided'), 400
        
        try:
            # Decode base64 image
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            image_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify(success=False, error='Failed to decode image'), 400
        except Exception as e:
            return jsonify(success=False, error=f'Image decode error: {str(e)}'), 500
    
    # SERVER MODE: Use camera frame
    else:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        
        if frame is None:
            return jsonify(success=False, error='Camera not ready'), 503
    
    # Run YOLO detection on the frame
    results = yolo_model(frame, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
    
    # FIX 2: Single person validation
    zone_boxes = [b for b in boxes if is_in_zone(b, zone_poly, margin_px=ZONE_MARGIN)]
    
    if len(zone_boxes) == 0:
        return jsonify(success=False, error='No person in monitored zone'), 400
    
    if len(zone_boxes) > 1:
        return jsonify(success=False, error=f'{len(zone_boxes)} persons detected. Ensure only one person is in frame'), 400
    
    target_box = zone_boxes[0]
    
    # Check enrollment quality
    quality_ok, quality_msg = check_enrollment_quality(frame, target_box)
    if not quality_ok:
        return jsonify(success=False, error=quality_msg), 400
    
    # Extract Re-ID embedding
    if USE_REID:
        # Try normal extraction first
        embedding = extract_person_embedding(frame, target_box)
        
        # If failed, try enhanced crop
        if embedding is None:
            print("[INFO] Normal extraction failed, trying enhanced image...")
            enhanced_crop = enhance_for_enrollment(frame, target_box)
            if enhanced_crop is not None:
                x1, y1, x2, y2 = [int(v) for v in target_box]
                frame[y1:y2, x1:x2] = enhanced_crop
                embedding = extract_person_embedding(frame, target_box)
        
        if embedding is None:
            return jsonify(success=False, error='Image too poor - retry with better lighting'), 500
        
        # Save enrollment screenshot
        screenshot = save_screenshot(frame, 'enrollment', person_label=label, person_slot=slot, crop_box=target_box)
        
        # Store enrollment
        with enroll_lock:
            enrolled_slots[slot]['embedding'] = embedding
            enrolled_slots[slot]['label'] = label
            enrolled_slots[slot]['photo_path'] = screenshot
            
            # FIX 4: Reset tracking state on enrollment
            slot_present_prev[slot] = False
            present_counts[slot] = 0
            absent_counts[slot] = 0
            grace_counters[slot] = 0
            last_entry_time[slot] = 0
            last_exit_time[slot] = 0
        
        print(f"[INFO] ✅ Enrolled Slot {slot} ({label})")
    else:
        # Fallback: histogram matching
        hist = compute_person_histogram(frame, target_box)
        if hist is None:
            return jsonify(success=False, error='Failed to compute histogram'), 500
        
        with enroll_lock:
            enrolled_slots[slot]['hist'] = hist
            enrolled_slots[slot]['label'] = label
    
    return jsonify(success=True, slot=slot, label=label)

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_from_directory('screenshots', filename)

@app.route('/enrollment_photo/<int:slot>')
def enrollment_photo(slot):
    if not (1 <= slot <= 5):
        return jsonify(error='Invalid slot'), 400
    
    with enroll_lock:
        slot_data = enrolled_slots.get(slot, {})
        label = slot_data.get('label')
        photo_path = slot_data.get('photo_path')
    
    if not label:
        return jsonify(error='Not enrolled'), 404
    
    if photo_path and os.path.exists(photo_path):
        dir_path, file_name = os.path.split(photo_path)
        return send_from_directory(dir_path, file_name)
    
    db_session = SessionLocal()
    try:
        event = db_session.query(Event).filter(
            Event.person_slot == slot,
            Event.screenshot_path.isnot(None)
        ).order_by(Event.timestamp.desc()).first()
        
        if event and event.screenshot_path:
            dir_path, file_name = os.path.split(event.screenshot_path)
            return send_from_directory(dir_path, file_name)
    finally:
        db_session.close()
    
    return jsonify(error='No photo'), 404

@app.route('/remove_person', methods=['POST'])
def remove_person():
    body = request.get_json(silent=True) or {}
    slot = int(body.get('slot', 0))
    
    if not (1 <= slot <= 5):
        return jsonify(success=False, error='Invalid slot'), 400
    
    with enroll_lock:
        if enrolled_slots[slot]['embedding'] is None:
            return jsonify(success=False, error='Slot not enrolled'), 400
        
        enrolled_slots[slot]['embedding'] = None
        enrolled_slots[slot]['label'] = None
        enrolled_slots[slot]['photo_path'] = None
        # FIX 4: Reset tracking state on removal
        slot_present_prev[slot] = False
        present_counts[slot] = 0
        absent_counts[slot] = 0
        grace_counters[slot] = 0
        last_entry_time[slot] = 0
        last_exit_time[slot] = 0
    
    print(f"[INFO] Removed Slot {slot}")
    return jsonify(success=True, slot=slot)

@app.route('/video_feed')
def video_feed():
    if app_mode == 'browser':
        return jsonify(error='Video feed not available in browser mode'), 400
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/match_frame', methods=['POST'])
def match_frame():
    """Browser mode: receive base64 image and run Re-ID matching on server"""
    if app_mode not in ['browser', 'hybrid']:
        return jsonify(error='Endpoint only available in browser mode'), 403
    
    try:
        data = request.get_json()
        image_b64 = data.get('image')
        box = data.get('box')  # [x1, y1, x2, y2]
        
        if not image_b64 or not box:
            return jsonify(error='Missing image or box data'), 400
        
        # Decode base64 to OpenCV frame
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify(error='Failed to decode image'), 400
        
        # Use existing match_slot_for_box function
        best_slot, score = match_slot_for_box(frame, box)
        
        with enroll_lock:
            label = enrolled_slots[best_slot]['label'] if best_slot else None
        
        return jsonify({
            'slot': best_slot,
            'score': float(score),
            'label': label
        })
    
    except Exception as e:
        print(f"[ERROR] match_frame: {e}")
        return jsonify(error=str(e)), 500

@app.route('/report_event', methods=['POST'])
def report_event():
    """Browser mode: receive detection events from client"""
    global total_entry_events, total_exit_events, total_unauthorized_events
    
    if app_mode not in ['browser', 'hybrid']:
        return jsonify(error='Endpoint only available in browser mode'), 403
    
    try:
        data = request.get_json()
        event_type = data.get('event_type')
        person_slot = data.get('person_slot')
        person_label = data.get('person_label', 'Unknown')
        image_b64 = data.get('screenshot')
        
        if not event_type:
            return jsonify(error='Missing event_type'), 400
        
        filepath = None
        if image_b64:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{event_type}_{timestamp_str}.jpg"
            filepath = os.path.join('screenshots', filename)
            
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            image_bytes = base64.b64decode(image_b64)
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
        
        db_session = SessionLocal()
        try:
            new_event = Event(
                event_type=event_type,
                zone_name=zone_config['name'],
                screenshot_path=filepath,
                person_slot=person_slot,
                person_label=person_label
            )
            
            db_session.add(new_event)
            db_session.commit()
            
            # Update global counters
            if event_type == 'entry':
                total_entry_events += 1
            elif event_type == 'exit':
                total_exit_events += 1
            elif event_type == 'unauthorized':
                total_unauthorized_events += 1
            
            print(f"[EVENT-BROWSER] {event_type.upper()} - {person_label}")
            return jsonify(success=True, event_id=new_event.id)
        finally:
            db_session.close()
    
    except Exception as e:
        print(f"[ERROR] report_event: {e}")
        return jsonify(error=str(e)), 500

@app.route('/clear_unauthorized', methods=['POST'])
def clear_unauthorized():
    """Clear only unauthorized events, keep entry/exit history."""
    global total_unauthorized_events
    
    db_session = SessionLocal()
    try:
        deleted_count = db_session.query(Event).filter(
            Event.event_type == 'unauthorized'
        ).delete()
        db_session.commit()
        
        total_unauthorized_events = 0
        print(f"[INFO] Cleared {deleted_count} unauthorized events")
        return jsonify(success=True, cleared=deleted_count)
    finally:
        db_session.close()

@app.route('/reset', methods=['POST'])
def reset():
    global total_entry_events, total_exit_events, total_unauthorized_events
    
    db_session = SessionLocal()
    try:
        db_session.query(Event).delete()
        db_session.commit()
    finally:
        db_session.close()
    
    # Reset counters
    total_entry_events = 0
    total_exit_events = 0
    total_unauthorized_events = 0
    
    shots_dir = os.path.join(os.getcwd(), 'screenshots')
    try:
        with enroll_lock:
            enrollment_photos = set()
            for i in range(1, 6):
                photo_path = enrolled_slots[i].get('photo_path')
                if photo_path:
                    enrollment_photos.add(os.path.basename(photo_path))
        
        for name in os.listdir(shots_dir):
            if name not in enrollment_photos:
                path = os.path.join(shots_dir, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"[WARN] Failed to remove {path}: {e}")
    except FileNotFoundError:
        pass
    
    print(f"[INFO] ✅ Reset complete")
    return jsonify(success=True)

if __name__ == '__main__':
    init_db()
    
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')
    
    # Only start camera threads in server mode
    if app_mode == 'server':
        grabber_thread = threading.Thread(target=frame_grabber, daemon=True)
        grabber_thread.start()
        
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        
        print(f"[INFO] Running in SERVER mode - CCTV/USB camera processing enabled")
    else:
        print(f"[INFO] Running in BROWSER mode - Client-side webcam detection")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
>>>>>>> 3b85ec0 ( v2.0 - Major Update: Exit Detection, Browser Mode, Action Buttons)
