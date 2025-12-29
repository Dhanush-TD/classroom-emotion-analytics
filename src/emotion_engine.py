import os
import cv2
import torch
import timm
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
from collections import deque, defaultdict
import math
from datetime import datetime

from src.config import EMOTION_MAP

# ======================================================
# PATH SETUP (CRITICAL FOR WEB APPS)
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "swin_t_rafdb_finetuned.pth"
)

# ======================================================
# DEVICE
# ======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# LOAD MODEL (ONCE)
# ======================================================

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=7
)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ======================================================
# FACE DETECTOR (MTCNN)
# ======================================================

mtcnn = MTCNN(
    image_size=160,
    margin=10,
    keep_all=True,
    device=device
)

# ======================================================
# TRANSFORM
# ======================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ======================================================
# TRACKING + SMOOTHING (MULTI-FACE SAFE)
# ======================================================

next_face_id = 0
face_centers = {}
emotion_queues = defaultdict(lambda: deque(maxlen=3))
MAX_DIST = 60


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_face_id(center):
    global next_face_id

    if not face_centers:
        face_centers[next_face_id] = center
        fid = next_face_id
        next_face_id += 1
        return fid

    distances = {
        fid: euclidean(center, prev)
        for fid, prev in face_centers.items()
    }

    fid, dist = min(distances.items(), key=lambda x: x[1])

    if dist < MAX_DIST:
        face_centers[fid] = center
        return fid
    else:
        face_centers[next_face_id] = center
        fid = next_face_id
        next_face_id += 1
        return fid


def smooth_prediction(fid, pred):
    q = emotion_queues[fid]
    q.append(pred)
    return max(set(q), key=q.count)

# ======================================================
# LEARNING STATE MAPPING (EDUCATION MODE)
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
# MAIN PREDICTION FUNCTION (USED BY WEBSITE)
# ======================================================

def predict_frame(frame_bgr, use_learning_states=True):
    """
    Takes a single BGR frame (from webcam / browser)
    Returns:
        - annotated_frame (BGR)
        - list of detected learning states or emotions
        - engagement percentage
    """

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    detected_states = []
    engagement = 0

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            h, w, _ = frame_bgr.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Mouth-focused crop (important for sad/confused)
            y1 = int(y1 + 0.15 * (y2 - y1))
            face = rgb[y1:y2, x1:x2]

            if face.shape[0] < 30 or face.shape[1] < 30:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            fid = get_face_id((cx, cy))

            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_tensor)

            pred = torch.argmax(outputs, dim=1).item()
            emotion_id = smooth_prediction(fid, pred)
            emotion = EMOTION_MAP[emotion_id]

            # Convert to learning state if requested
            if use_learning_states:
                state = LEARNING_STATE_MAP[emotion]
            else:
                state = emotion

            detected_states.append(state)

            # Draw on frame
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                state,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Calculate engagement percentage
        if detected_states:
            scores = [ENGAGEMENT_WEIGHT[s] for s in detected_states]
            engagement = int((sum(scores) / len(scores)) * 100)

    return frame_bgr, detected_states, engagement
