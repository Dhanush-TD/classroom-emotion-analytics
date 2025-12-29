# 🎯 Integration Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    WEB BROWSER (Frontend)                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Dashboard UI (HTML/CSS)                  │ │
│  │                                                            │ │
│  │  [Start] [End] | Engagement: 72% | Faces: 2             │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  📹 Live Video Feed (640x480)                        │ │ │
│  │  │  Shows webcam stream in real-time                   │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │  Learning States: [Engaged] [Attentive]                 │ │
│  │  Emotions: [Happy: 45%] [Neutral: 35%] [Sad: 20%]      │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  📊 Last Session Analytics Plot                      │ │ │
│  │  │  (Displays after session ends)                       │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  JavaScript Layer:                                              │
│  • Canvas frame capture (500ms interval)                        │
│  • BASE64 image encoding                                        │
│  • Fetch API calls to backend                                   │
│  • Real-time DOM updates                                        │
│  • Statistics accumulation                                      │
└──────────────┬───────────────────────────────────────────────────┘
               │
               │ HTTP/JSON
               │ (POST /predict, /start-session, /end-session)
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                   FLASK BACKEND (app.py)                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Session Management:                                       │ │
│  │  • /start-session → Initialize CSV, reset counters        │ │
│  │  • /predict → Process frame, return emotions/states       │ │
│  │  • /end-session → Save CSV, generate plot, reload page    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Image Processing Pipeline:                               │ │
│  │                                                            │ │
│  │  Input: BASE64 Image String                              │ │
│  │     ↓                                                     │ │
│  │  BASE64 Decode → NumPy Array → OpenCV Frame            │ │
│  │     ↓                                                     │ │
│  │  Color Space Conversion (RGBA → BGR if needed)          │ │
│  │     ↓                                                     │ │
│  │  Output: OpenCV BGR Image                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  AI Model Pipeline:                                        │ │
│  │                                                            │ │
│  │  Frame Input                                              │ │
│  │     ↓                                                     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ MTCNN Face Detection                               │ │ │
│  │  │ (Detects all faces in frame)                       │ │ │
│  │  │ Outputs: Face tensors (batch)                      │ │ │
│  │  └────┬─────────────────────────────────────────────────┘ │ │
│  │       ↓                                                     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Swin Transformer Model                             │ │ │
│  │  │ (7 emotion classification)                          │ │ │
│  │  │ Inputs: Face tensors                                │ │ │
│  │  │ Outputs: Emotion logits for each face               │ │ │
│  │  └────┬─────────────────────────────────────────────────┘ │ │
│  │       ↓                                                     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Emotion → Learning State Mapping                    │ │ │
│  │  │ Happy → Engaged                                     │ │ │
│  │  │ Neutral → Attentive                                 │ │ │
│  │  │ Surprise → Interested                               │ │ │
│  │  │ Sad → Confused                                      │ │ │
│  │  │ Angry → Frustrated                                  │ │ │
│  │  │ Fear → Anxious                                      │ │ │
│  │  │ Disgust → Disengaged                                │ │ │
│  │  └────┬─────────────────────────────────────────────────┘ │ │
│  │       ↓                                                     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Engagement Score Calculation                        │ │ │
│  │  │ avg_engagement = mean(ENGAGEMENT_WEIGHT[state])     │ │ │
│  │  │ Output: Engagement % (0-100%)                       │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Analytics Tracking:                                       │ │
│  │  • ANALYTICS_DATA[] → Accumulates per-frame data         │ │
│  │  • ALL_EMOTIONS[] → Tracks all detected emotions         │ │
│  │  • ALL_STATES[] → Tracks all detected states             │ │
│  │  • emotionCounts, stateCounts → Statistics               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  JSON Response (to Browser):                              │ │
│  │  {                                                         │ │
│  │    "success": true,                                       │ │
│  │    "emotions": ["Happy", "Neutral"],                      │ │
│  │    "states": ["Engaged", "Attentive"],                    │ │
│  │    "engagement": 0.72                                     │ │
│  │  }                                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────┬───────────────────────────────────────────────────┘
               │
               │ File I/O
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                  DATA STORAGE & PROCESSING                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  CSV Recording:                                            │ │
│  │  File: analytics/session_YYYYMMDD_HHMMSS.csv             │ │
│  │                                                            │ │
│  │  Time,Learning_State                                      │ │
│  │  11:45:29,Attentive                                       │ │
│  │  11:45:30,Engaged                                         │ │
│  │  11:45:31,Attentive                                       │ │
│  │  ... (one row per detected face per frame)                │ │
│  │                                                            │ │
│  │  Saved when: /end-session is called                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Plot Generation:                                          │ │
│  │  subprocess.run(['python', 'src/plot_analytics.py', csv]) │ │
│  │                                                            │ │
│  │  Reads CSV → Counts states → Creates bar chart           │ │
│  │  Output: analytics/plots/session_..._stylish.png         │ │
│  │                                                            │ │
│  │  Displayed in browser after generation                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Flow

### 1️⃣ Session Start
```
User Click [Start]
    ↓
Browser: fetch('/start-session', {POST})
    ↓
Flask: Initialize CSV file, reset counters
    ↓
Response: {"status": "started", "session_id": "20251222_112345"}
    ↓
Browser: Enable /predict calls every 500ms
```

### 2️⃣ Frame Processing (Repeating every 500ms)
```
Browser: Capture webcam frame
    ↓
Canvas.toDataURL('image/jpeg') → BASE64 string
    ↓
fetch('/predict', {image: base64})
    ↓
Flask: 
  1. Decode BASE64 → NumPy array
  2. MTCNN detect faces
  3. Swin model classify emotions
  4. Map to learning states
  5. Calculate engagement
    ↓
Response: {emotions: [...], states: [...], engagement: 0.72}
    ↓
Browser:
  1. Update DOM with new states
  2. Update engagement percentage
  3. Update emotion percentages
  4. Accumulate statistics
    ↓
Display updates visible in real-time
```

### 3️⃣ Session End
```
User Click [End]
    ↓
Browser: fetch('/end-session', {POST})
    ↓
Flask:
  1. Write all accumulated data to CSV
  2. subprocess.run(plot_analytics.py CSV)
  3. Plot generation runs
  4. Return success response
    ↓
Browser: Reload page
    ↓
New plot displays if generation successful
```

---

## Device & Hardware

```
┌─────────────────────────────────┐
│   Physical Hardware             │
│                                 │
│  🎥 Webcam                      │
│  💾 GPU (Optional, CUDA)        │
│  🧠 CPU (Fallback)              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   PyTorch/CUDA Layer            │
│                                 │
│  torch.backends.cudnn.benchmark │
│  torch.backends.cuda.matmul     │
│  device = cuda or cpu           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Model Inference               │
│                                 │
│  MTCNN (face detection)         │
│  Swin Transformer (emotions)    │
└─────────────────────────────────┘
```

---

## File Dependencies

```
emotion_project/
│
├── analytics/dashboard/
│   ├── app.py                    ← MAIN SERVER
│   │   ├── imports: Flask, torch, cv2, numpy, timm, facenet_pytorch
│   │   ├── loads: checkpoints/swin_t_rafdb_finetuned.pth
│   │   ├── requires: src/config.py
│   │   └── calls: src/plot_analytics.py (subprocess)
│   │
│   ├── templates/index.html      ← FRONTEND
│   │   ├── imports: CSS from static/style.css
│   │   ├── calls: /start-session, /predict, /end-session
│   │   └── displays: real-time metrics & plots
│   │
│   └── static/style.css          ← STYLING
│       └── referenced by: index.html
│
├── src/
│   ├── config.py                 ← EMOTION MAP
│   │   └── used by: app.py
│   │
│   ├── plot_analytics.py         ← PLOT GENERATOR
│   │   └── called by: app.py (/end-session)
│   │
│   └── webcam_emotion_education.py ← ORIGINAL (standalone works too!)
│
└── checkpoints/
    └── swin_t_rafdb_finetuned.pth ← MODEL WEIGHTS
        └── loaded by: app.py on startup
```

---

## Performance Metrics

| Component | Time | Impact |
|-----------|------|--------|
| **Frame Capture** | ~10ms | Video stream (continuous) |
| **MTCNN Face Detection** | ~100-200ms | Bottleneck for multiple faces |
| **Swin Model Inference** | ~50-100ms per face | GPU accelerated |
| **Total per frame** | ~150-300ms | At 500ms interval = OK |
| **Canvas→BASE64** | ~50ms | Depends on compression |
| **Network latency** | ~10-50ms | Local = very fast |
| **Plot generation** | ~2-5s | Called once per session |

**Note**: With GPU, inference is ~10x faster!

---

## Error Handling

```
Browser Error Path:
  Webcam Access Denied
    ↓
  Display: "❌ Webcam not available"
  
API Error Path:
  /predict returns error
    ↓
  Console: "Prediction error: ..."
  ↓
  Display continues without that frame
  
CSV Error Path:
  Cannot write to analytics/
    ↓
  Check: permissions, disk space, path exists
    
Plot Error Path:
  plot_analytics.py fails
    ↓
  Flask logs error but returns success
  ↓
  Browser reloads (plot may not appear)
  ↓
  Check: CSV file exists, folder writable
```

---

## Security Considerations

⚠️ **Development Only** - Not for Production!

Current setup:
- ❌ No HTTPS
- ❌ No authentication
- ❌ No rate limiting
- ❌ No CORS protection
- ✅ Local network only
- ✅ No external data transmission

For production:
1. Use WSGI server (Gunicorn)
2. Add SSL/HTTPS
3. Add authentication
4. Add rate limiting
5. Add proper error logging
6. Validate all inputs
7. Use database instead of CSV

---

## Scaling Considerations

Current implementation handles:
- ✅ 1-10 concurrent faces
- ✅ 2 FPS frame rate
- ✅ Single browser session
- ✅ Single user

For scaling:
- Add database (SQLite → PostgreSQL)
- Add multiple sessions/users
- Add real-time WebSocket updates
- Add background task queue (Celery)
- Add caching (Redis)
- Deploy on production server

---

## Monitoring & Debugging

### Check Flask Server:
```powershell
# Watch logs in terminal
# Error messages appear in real-time
# Look for: {status_code} in HTTP requests
```

### Check Browser Console:
```javascript
F12 → Console tab
// Look for: "Prediction error", "Frame capture error"
```

### Check Analytics Output:
```bash
# Verify CSV created
dir analytics\session_*.csv

# Verify plot generated
dir analytics\plots\*.png

# Check CSV contents
type analytics\session_20251222_112345.csv
```

### Performance:
```javascript
// In browser console:
// Monitor network tab for /predict latency
// Monitor CPU usage while running
// Check memory usage in Task Manager
```

---

## Summary

```
Old Workflow:        Command-line → webcam_emotion_education.py → CSV/Plot
New Workflow:        Browser → Flask app → AI backend → CSV/Plot

Same Model:          ✅ Swin Transformer (unchanged)
Same Emotions:       ✅ 7 emotions (unchanged)  
Same Data:           ✅ CSV format (unchanged)
Same Plots:          ✅ Same visualization (unchanged)

New Capability:      🌐 Web-based interface
Better UX:           📊 Real-time dashboard
Multi-session:       🔄 Easier to manage
Beautiful UI:        🎨 Professional looking
```

---

**Integration Status**: ✅ Complete & Functional!
