import cv2
import torch
import timm
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
from collections import deque, defaultdict
from PIL import Image
import math

from config import EMOTION_MAP

# ===============================
# TORCH OPTIMIZATION
# ===============================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# LOAD FINE-TUNED MODEL
# ===============================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=7
)

model.load_state_dict(
    torch.load("checkpoints/swin_t_rafdb_finetuned.pth", map_location=device)
)

model.to(device)
model.eval()

# ===============================
# FACE DETECTOR (MTCNN)
# ===============================
mtcnn = MTCNN(
    image_size=160,
    margin=10,
    keep_all=True,
    device=device
)

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ===============================
# TRACKING STATE (INTERNAL)
# ===============================
next_face_id = 0
face_centers = {}  # face_id -> (cx, cy)
emotion_queues = defaultdict(lambda: deque(maxlen=3))

MAX_DIST = 60  # tracking sensitivity (pixels)

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_face_id(center):
    global next_face_id

    if not face_centers:
        face_centers[next_face_id] = center
        fid = next_face_id
        next_face_id += 1
        return fid

    distances = {
        fid: euclidean(center, prev_center)
        for fid, prev_center in face_centers.items()
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

# ===============================
# WEBCAM (DROIDCAM)
# ===============================
cap = cv2.VideoCapture(1)  # change to 0 if needed

if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Press ESC to exit")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Focus more on mouth region
            y1 = int(y1 + 0.15 * (y2 - y1))

            face = rgb[y1:y2, x1:x2]
            if face.shape[0] < 30 or face.shape[1] < 30:
                continue

            # Tracking center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            face_id = get_face_id((cx, cy))

            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil)\
                .unsqueeze(0)\
                .to(device)

            with torch.no_grad():
                outputs = model(face_tensor)

            pred = torch.argmax(outputs, dim=1).item()
            emotion_id = smooth_prediction(face_id, pred)
            emotion = EMOTION_MAP[emotion_id]

            # DRAW (CLEAN UI)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(
                frame,
                emotion,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Multi-Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
