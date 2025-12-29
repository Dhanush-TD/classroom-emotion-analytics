
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify, send_file

import csv
import cv2
import base64
import numpy as np
from datetime import datetime
import torch
import timm
from facenet_pytorch import MTCNN
from torchvision import transforms
from collections import deque, defaultdict
from PIL import Image
import math

from src.emotion_engine import predict_frame
from src.config import EMOTION_MAP
import subprocess
import threading

app = Flask(__name__)

# ======================================================
# TORCH OPTIMIZATION
# ======================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[dashboard] device: {device}")

# ======================================================
# LOAD FINE-TUNED MODEL
# ======================================================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=7
)

checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints/swin_t_rafdb_finetuned.pth")
print(f"[dashboard] loading model checkpoint: {checkpoint_path}")
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    # If running on CUDA, convert model to half precision for faster inference
    if device.type == 'cuda':
        try:
            model.half()
            print("[dashboard] model converted to half precision")
        except Exception:
            pass
    print("[dashboard] model loaded and set to eval()")
except Exception as e:
    print(f"[dashboard] failed loading model: {e}")
    import traceback
    traceback.print_exc()

# ======================================================
# FACE DETECTOR (MTCNN) - Returns boxes and tensors
# ======================================================
mtcnn = MTCNN(
    image_size=224,
    margin=10,
    keep_all=True,
    device=device,
    post_process=False  # Get raw boxes
)
print("[dashboard] MTCNN initialized")

# ======================================================
# TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ======================================================
# EMOTION & COLORS
# ======================================================
EMOTION_COLORS = {
    "Happy": (0, 255, 0),      # Green
    "Neutral": (128, 128, 128), # Gray
    "Surprise": (255, 0, 255),  # Magenta
    "Sad": (255, 0, 0),        # Blue (BGR)
    "Angry": (0, 0, 255),      # Red (BGR)
    "Fear": (128, 0, 128),     # Purple
    "Disgust": (0, 128, 128)   # Teal
}

# ======================================================
# EDUCATION LEARNING STATES
# ======================================================
LEARNING_STATE_MAP = {
    "Happy": "Engaged",
    "Neutral": "Attentive",
    "Surprise": "Interested",
    "Sad": "Confused",
    "Angry": "Frustrated",
    "Fear": "Anxious",
    "Disgust": "Disengaged"
}

ENGAGEMENT_WEIGHT = {
    "Engaged": 1.0,
    "Interested": 0.8,
    "Attentive": 0.6,
    "Confused": 0.3,
    "Frustrated": 0.2,
    "Anxious": 0.2,
    "Disengaged": 0.0
}

# ======================================================
# FACE TRACKING
# ======================================================
next_face_id = 0
face_centers = {}
emotion_queues = defaultdict(lambda: deque(maxlen=3))
MAX_DIST = 60
last_emotion_per_id = {}
last_conf_per_id = {}

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_face_id(center):
    """Return (face_id, distance_to_matched_center)."""
    global next_face_id

    if not face_centers:
        face_id = next_face_id
        next_face_id += 1
        face_centers[face_id] = center
        return face_id, 0.0

    distances = {fid: euclidean(center, fc) for fid, fc in face_centers.items()}
    closest_id = min(distances, key=distances.get)
    dist = distances[closest_id]

    if dist < MAX_DIST:
        face_centers[closest_id] = center
        return closest_id, dist
    else:
        face_id = next_face_id
        next_face_id += 1
        face_centers[face_id] = center
        return face_id, float('inf')

def predict_emotion_and_state(frame):
    """Predict emotion and learning state for detected faces with bounding boxes"""
    try:
        # Convert BGR numpy array to PIL Image (MTCNN requires PIL Image)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Detect faces using MTCNN
        # boxes: [x1, y1, x2, y2, conf] for each face
        boxes, probs = mtcnn.detect(pil_image, landmarks=False)
        
        # Also get face tensors for emotion prediction
        face_tensors = mtcnn(pil_image)
        
        if boxes is None or len(boxes) == 0:
            print("⚠️ No faces detected in this frame")
            return [], [], 0, frame, []
        
        if face_tensors is None:
            print(f"⚠️ No face tensors extracted")
            return [], [], 0, frame, []
        
        if not isinstance(face_tensors, torch.Tensor):
            print(f"⚠️ Unexpected face type: {type(face_tensors)}")
            return [], [], 0, frame, []
        
        # Ensure proper dimensions
        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)
        
        emotions = []
        states = []
        engagement_scores = []
        face_boxes = []
        
        print(f"✓ Detected {face_tensors.shape[0]} face(s)")
        
        with torch.no_grad():
            for i in range(face_tensors.shape[0]):
                try:
                    # Get face tensor
                    face_tensor = face_tensors[i:i+1].to(device)

                    # Resize to model input
                    face_tensor_resized = torch.nn.functional.interpolate(
                        face_tensor,
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )

                    # Ensure dtype matches model on CUDA
                    if device.type == 'cuda' and face_tensor_resized.dtype != torch.float16:
                        face_tensor_resized = face_tensor_resized.half()

                    # Single forward pass
                    logits = model(face_tensor_resized)
                    emotion_id = int(torch.argmax(logits, dim=1).item())
                    emotion = EMOTION_MAP.get(emotion_id, "Neutral")
                    emotion_conf = float(torch.softmax(logits, dim=1)[0, emotion_id].item())

                    # Detector confidence (reduce false positives)
                    det_conf = None
                    try:
                        if probs is not None and len(probs) > i:
                            det_conf = float(probs[i])
                    except Exception:
                        det_conf = None
                    if det_conf is not None and det_conf < 0.55:
                        continue

                    # Get box coordinates
                    raw_box = boxes[i]
                    if isinstance(raw_box, np.ndarray):
                        b = [int(float(x)) for x in raw_box[:4]]
                    else:
                        b = [int(float(x)) for x in list(raw_box)[:4]]

                    # Compute center and get face id
                    cx = int((b[0] + b[2]) / 2)
                    cy = int((b[1] + b[3]) / 2)
                    face_id, moved_dist = get_face_id((cx, cy))

                    # Append to short history for this face
                    try:
                        emotion_queues[face_id].append((emotion, emotion_conf))
                    except Exception:
                        emotion_queues[face_id] = deque(maxlen=5)
                        emotion_queues[face_id].append((emotion, emotion_conf))

                    # Compute dominant emotion from history and average confidence
                    history = list(emotion_queues[face_id])
                    emotions_only = [e for e, c in history]
                    confs_only = [c for e, c in history]
                    from collections import Counter
                    dominant_emotion, count = Counter(emotions_only).most_common(1)[0]
                    avg_conf = float(np.mean(confs_only)) if confs_only else emotion_conf

                    # Temporal smoothing / confidence-aware mapping
                    # Rules:
                    # - If recent dominant emotion appears in >=60% of history and avg_conf >=0.65, trust dominant.
                    # - If avg_conf is low, fallback to last known emotion or 'Attentive'.
                    trusted = (count >= max(1, int(0.6 * len(history))) and avg_conf >= 0.65)

                    if trusted:
                        final_emotion = dominant_emotion
                        final_conf = avg_conf
                    else:
                        # If face hasn't moved and we have last emotion, reuse it
                        reuse_threshold = 8.0
                        if face_id in last_emotion_per_id and moved_dist < reuse_threshold:
                            final_emotion = last_emotion_per_id[face_id]
                            final_conf = last_conf_per_id.get(face_id, emotion_conf)
                        else:
                            final_emotion = emotion
                            final_conf = emotion_conf

                    # Map emotion -> learning state with confidence adjustment
                    # Base mapping
                    base_state = LEARNING_STATE_MAP.get(final_emotion, "Attentive")
                    base_weight = ENGAGEMENT_WEIGHT.get(base_state, 0.5)

                    # Adjust weight slightly by confidence (push towards 1.0 if confident)
                    weight_adj = (final_conf - 0.5) * 0.6  # scale factor
                    final_weight = float(min(1.0, max(0.0, base_weight + weight_adj)))

                    # Save last emotion for face id
                    last_emotion_per_id[face_id] = final_emotion
                    last_conf_per_id[face_id] = final_conf

                    # Append outputs
                    emotions.append(final_emotion)
                    states.append(base_state)
                    engagement_scores.append(final_weight)

                    face_boxes.append({
                        "box": b,
                        "emotion": final_emotion,
                        "confidence": round(final_conf, 3)
                    })

                except Exception as face_error:
                    print(f"❌ Error processing face {i}: {face_error}")
                    continue
        
        avg_engagement = float(np.mean(engagement_scores)) if engagement_scores else 0.0
        
        return emotions, states, avg_engagement, frame, face_boxes
        
    except Exception as e:
        print(f"❌ Error in predict_emotion_and_state: {e}")
        import traceback
        traceback.print_exc()
        return [], [], 0, frame, []

# ======================================================
# GLOBAL SESSION DATA
# ======================================================
SESSION_ID = None
CSV_PATH = None
ANALYTICS_DATA = []
ALL_EMOTIONS = []
ALL_STATES = []

@app.route("/")
def index():
    plot = None
    plot_timestamp = None
    plots_dir = os.path.join(os.path.dirname(__file__), "../../analytics/plots")

    if os.path.exists(plots_dir):
        files = sorted(os.listdir(plots_dir), reverse=True)
        if files:
            plot = files[0]
            # derive timestamp from filename like 'session_YYYYMMDD_HHMMSS_stylish.png'
            base = os.path.basename(plot)
            name, _ext = os.path.splitext(base)
            if name.startswith("session_"):
                # remove leading 'session_'
                rest = name[len("session_"):]
                # strip any suffix like '_stylish'
                plot_timestamp = rest.split("_stylish")[0]

    return render_template("index.html", plot_file=plot, plot_timestamp=plot_timestamp)

@app.route("/start-session", methods=["POST"])
def start_session():
    global SESSION_ID, CSV_PATH, ANALYTICS_DATA

    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_PATH = os.path.join(os.path.dirname(__file__), f"../../analytics/session_{SESSION_ID}.csv")
    ANALYTICS_DATA = []

    # Create analytics directory if it doesn't exist
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Write CSV header
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Learning_State"])

    return jsonify({"status": "started", "session_id": SESSION_ID})

@app.route("/predict", methods=["POST"])
def predict():
    global CSV_PATH, ANALYTICS_DATA, ALL_EMOTIONS, ALL_STATES

    try:
        data = request.json["image"]
        
        # Decode BASE64 image
        if "," in data:
            data = data.split(",")[1]
        
        img_bytes = base64.b64decode(data)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("⚠️ Frame decoding failed")
            return jsonify({
                "emotions": [], "states": [], "engagement": 0,
                "success": False, "error": "Frame decoding failed", "faces": []
            }), 400
        
        # Ensure frame is BGR
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Predict emotions and get face boxes
        emotions, states, engagement, _, face_boxes = predict_emotion_and_state(frame)

        # Save to analytics
        timestamp = datetime.now().strftime("%H:%M:%S")
        if CSV_PATH and states:
            for state in states:
                ANALYTICS_DATA.append([timestamp, state])
                ALL_STATES.append(state)
        
        if emotions:
            ALL_EMOTIONS.extend(emotions)

        return jsonify({
            "emotions": emotions,
            "states": states,
            "engagement": float(engagement),
            "faces": face_boxes,  # Send face boxes for drawing
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
            "success": True
        })
    except Exception as e:
        print(f"❌ Error in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "emotions": [], "states": [], "engagement": 0,
            "success": False, "error": str(e), "faces": []
        }), 400



@app.route("/end-session", methods=["POST"])
def end_session():
    global CSV_PATH, ANALYTICS_DATA, ALL_EMOTIONS, ALL_STATES

    try:
        if CSV_PATH and ANALYTICS_DATA:
            # Write all collected data to CSV
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(ANALYTICS_DATA)

            print(f"Session analytics saved to: {CSV_PATH}")

            # Calculate statistics
            stats = {
                "total_frames": len(ANALYTICS_DATA),
                "total_emotions": len(ALL_EMOTIONS),
                "unique_states": list(set(ALL_STATES))
            }

            # Generate plot asynchronously to avoid blocking the HTTP response
            def generate_plot_async(csv_path):
                try:
                    plot_script = os.path.join(PROJECT_ROOT, "src/plot_analytics.py")
                    result = subprocess.run([
                        sys.executable,
                        plot_script,
                        csv_path
                    ], capture_output=True, text=True, cwd=PROJECT_ROOT)
                    if result.returncode != 0:
                        print(f"Plot generation warning: {result.stderr}")
                    else:
                        print(f"Plot generated successfully")
                except Exception as e:
                    print(f"Error in async plot generation: {e}")

            threading.Thread(target=generate_plot_async, args=(CSV_PATH,), daemon=True).start()

            # Reset data
            ANALYTICS_DATA.clear()
            ALL_EMOTIONS.clear()
            ALL_STATES.clear()

            return jsonify({
                "status": "ended",
                "message": "Session analytics saved and plot generated",
                "stats": stats
            })
        else:
            return jsonify({
                "status": "ended",
                "message": "No data to save"
            })
    except Exception as e:
        print(f"Error in /end-session: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route("/sessions", methods=["GET"])
def get_sessions():
    """Get list of all saved sessions with their files"""
    try:
        analytics_dir = os.path.join(PROJECT_ROOT, "analytics")
        sessions = []
        
        # Find all CSV files
        for filename in os.listdir(analytics_dir):
            if filename.startswith("session_") and filename.endswith(".csv"):
                # Extract timestamp from filename
                timestamp_str = filename.replace("session_", "").replace(".csv", "")
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Check if corresponding plot exists
                    plot_files = [f for f in os.listdir(os.path.join(analytics_dir, "plots"))
                                if f.startswith(f"session_{timestamp_str}")]
                    
                    sessions.append({
                        "id": timestamp_str,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "csv_file": filename,
                        "plot_file": plot_files[0] if plot_files else None,
                        "csv_path": f"/download-csv/{timestamp_str}",
                        "plot_path": f"/download-plot/{timestamp_str}" if plot_files else None
                    })
                except ValueError:
                    continue
        
        # Sort by timestamp descending
        sessions.sort(key=lambda x: x["id"], reverse=True)
        
        return jsonify({"sessions": sessions, "success": True})
    except Exception as e:
        print(f"Error getting sessions: {e}")
        return jsonify({"sessions": [], "success": False, "error": str(e)})

@app.route("/download-csv/<timestamp>", methods=["GET"])
def download_csv(timestamp):
    """Download CSV file for a session"""
    try:
        csv_path = os.path.join(PROJECT_ROOT, f"analytics/session_{timestamp}.csv")
        
        if not os.path.exists(csv_path):
            return {"error": "File not found"}, 404
        
        return send_file(
            csv_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"session_{timestamp}.csv"
        )
    except Exception as e:
        print(f"Error downloading CSV: {e}")
        return {"error": str(e)}, 500

@app.route("/download-plot/<timestamp>", methods=["GET"])
def download_plot(timestamp):
    """Download plot PNG for a session"""
    try:
        plots_dir = os.path.join(PROJECT_ROOT, "analytics/plots")
        
        # Find plot file for this timestamp
        plot_files = [f for f in os.listdir(plots_dir) if f.startswith(f"session_{timestamp}")]
        
        if not plot_files:
            return {"error": "Plot not found"}, 404
        
        plot_path = os.path.join(plots_dir, plot_files[0])
        
        return send_file(
            plot_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=plot_files[0]
        )
    except Exception as e:
        print(f"Error downloading plot: {e}")
        return {"error": str(e)}, 500
@app.route("/plot/<timestamp>", methods=["GET"])
def serve_plot(timestamp):
    try:
        plots_dir = os.path.join(PROJECT_ROOT, "analytics/plots")
        plot_files = [f for f in os.listdir(plots_dir) if f.startswith(f"session_{timestamp}")]

        if not plot_files:
            return {"error": "Plot not found"}, 404

        plot_path = os.path.join(plots_dir, plot_files[0])

        return send_file(
            plot_path,
            mimetype="image/png",
            as_attachment=False
        )
    except Exception as e:
        print(f"Error serving plot: {e}")
        return {"error": str(e)}, 500


if __name__ == "__main__":
    # Enable threaded server to handle concurrent requests (dev only)
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
