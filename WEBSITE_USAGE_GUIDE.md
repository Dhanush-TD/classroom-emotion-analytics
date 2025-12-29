# 🎓 Education Emotion Analytics Website - Usage Guide

## Overview
Your `webcam_emotion_education.py` functionality has been successfully integrated into a Flask web dashboard! The website now runs real-time emotion detection and learning state analysis directly from your browser.

## ✨ Features

### Real-Time Emotion Detection
- Detects 7 emotions in real-time from webcam feed:
  - 😊 Happy → **Engaged**
  - 😐 Neutral → **Attentive**
  - 😮 Surprise → **Interested**
  - 😢 Sad → **Confused**
  - 😠 Angry → **Frustrated**
  - 😨 Fear → **Anxious**
  - 🤢 Disgust → **Disengaged**

### Live Analytics Dashboard
- **Engagement Score**: Real-time engagement percentage (0-100%)
- **Face Detection**: Number of faces detected in the frame
- **Learning States**: Current learning state of each detected person
- **Emotion Percentages**: Percentage of each emotion detected in the session

### Session Recording
- Automatically saves session data to CSV file
- Generates visualization plots after session ends
- Stores analytics in `analytics/` directory

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment activated: `fer_env`
- Dependencies installed (PyTorch, OpenCV, Flask, etc.)

### Step 1: Start the Flask Server

**Option A: Using PowerShell (Windows)**
```powershell
cd "D:\Git Hub Data\emotion_project"
.\fer_env\Scripts\Activate.ps1
python analytics/dashboard/app.py
```

**Option B: Using Command Prompt (Windows)**
```bash
cd D:\Git Hub Data\emotion_project
fer_env\Scripts\activate.bat
python analytics/dashboard/app.py
```

**Option C: Using the provided batch files**
```bash
run_dashboard.bat        # For Command Prompt
run_dashboard.ps1        # For PowerShell
```

### Step 2: Open in Browser

Once the Flask server is running, open your web browser and go to:
```
http://127.0.0.1:5000
```

You should see the **Education Emotion Analytics Dashboard** with:
- Start/End Session buttons
- Live webcam feed
- Real-time metrics
- Previous session plots

---

## 📊 How to Use

### Starting a Session
1. Click the **▶ Start Session** button
2. Allow browser access to your webcam when prompted
3. The system will begin detecting emotions and learning states
4. You'll see live updates of:
   - Engagement percentage
   - Number of detected faces
   - Current learning states
   - Emotion distribution

### During Session
- The dashboard captures frames every 500ms
- Each frame is analyzed for emotions and learning states
- Data is accumulated in real-time
- You can see live metrics updating

### Ending a Session
1. Click the **⏹ End Session** button
2. The system will:
   - Save all session data to CSV (with timestamp)
   - Generate a bar chart visualization
   - Display the plot on the dashboard
   - Automatically reload the page

### Viewing Results
- Last session plot displays automatically
- CSV files stored in: `analytics/session_YYYYMMDD_HHMMSS.csv`
- Plot images stored in: `analytics/plots/`

---

## 🎨 Dashboard Sections

### 1. Controls
```
[▶ Start Session] [⏹ End Session] [Session Status]
```
- Start/End buttons to control recording
- Status indicator shows session state

### 2. Video Feed
- Live webcam display (640x480)
- Shows in real-time what the AI is analyzing

### 3. Metrics (Top)
- **Engagement Score**: Overall engagement percentage
- **Face Count**: Number of people detected

### 4. Learning States Section
- Shows current learning states of detected faces
- Color-coded badges:
  - 🟢 Green = Engaged (1.0)
  - 🔵 Blue = Interested (0.8)
  - 🟡 Yellow = Attentive (0.6)
  - 🟠 Orange = Confused (0.3)
  - 🔴 Red = Frustrated (0.2)
  - 🟣 Purple = Anxious (0.2)
  - ⚪ Gray = Disengaged (0.0)

### 5. Emotions Section
- Shows distribution of detected emotions
- Percentage of each emotion throughout session
- Color-coded by emotion type

### 6. Analytics Plot
- Bar chart showing learning state distribution
- Generated after session ends
- Shows percentage of time in each state

---

## 📁 File Structure

```
emotion_project/
├── analytics/
│   ├── dashboard/
│   │   ├── app.py                 # Flask server (UPDATED)
│   │   ├── templates/
│   │   │   └── index.html         # Dashboard UI (UPDATED)
│   │   └── static/
│   │       └── style.css          # Styling (UPDATED)
│   ├── session_YYYYMMDD_HHMMSS.csv  # Session recordings
│   └── plots/
│       └── session_YYYYMMDD_HHMMSS_stylish.png  # Generated plots
├── src/
│   ├── emotion_engine.py          # Emotion detection model
│   ├── config.py                  # Configuration
│   ├── plot_analytics.py          # Plot generation
│   └── webcam_emotion_education.py  # Original script (still works!)
└── checkpoints/
    └── swin_t_rafdb_finetuned.pth   # Fine-tuned model weights
```

---

## 🔧 Technical Details

### Backend (Flask)
- **Framework**: Flask 3.1.2
- **Model**: Swin Transformer (swin_t_rafdb_finetuned)
- **Face Detection**: MTCNN
- **Device**: GPU (if available) or CPU

### Frontend (JavaScript)
- Canvas-based frame capture
- Real-time fetch API calls
- Dynamic DOM updates
- Responsive design

### Data Flow
```
Webcam → Canvas Capture → Base64 Encoding
    ↓
/predict endpoint → Model Inference
    ↓
Emotion + Learning State Detection
    ↓
CSV Storage + Real-time Display
    ↓
/end-session endpoint → Plot Generation
```

---

## ⚙️ Configuration

### Frame Capture Rate
Default: Every 500ms (2 frames per second)
To change, edit `index.html` line with:
```javascript
frameInterval = setInterval(sendFrame, 500);  // Change 500 to desired milliseconds
```

### Model Checkpoint
Location: `checkpoints/swin_t_rafdb_finetuned.pth`
Automatically loaded when Flask server starts

### Engagement Weights
Edit `app.py` ENGAGEMENT_WEIGHT dictionary:
```python
ENGAGEMENT_WEIGHT = {
    "Engaged": 1.0,
    "Interested": 0.8,
    "Attentive": 0.6,
    "Confused": 0.3,
    "Frustrated": 0.2,
    "Anxious": 0.2,
    "Disengaged": 0.0
}
```

---

## 🐛 Troubleshooting

### Issue: Webcam not working
**Solution:**
- Check browser permissions (Settings → Privacy)
- Try a different browser (Chrome/Firefox/Edge)
- Restart the Flask server
- Check device has working webcam

### Issue: Model loading error
**Solution:**
- Ensure `checkpoints/swin_t_rafdb_finetuned.pth` exists
- Check CUDA is properly installed (for GPU)
- Virtual environment is activated

### Issue: Slow performance
**Solution:**
- Reduce frame rate (increase interval in JavaScript)
- Close other applications
- Use GPU if available
- Reduce video resolution

### Issue: Plot not generating
**Solution:**
- Check `src/plot_analytics.py` exists
- Ensure `analytics/` directory is writable
- Check console for error messages
- Try ending session again

### Issue: Port 5000 already in use
**Solution:**
```powershell
# Change port in app.py
app.run(debug=True, host="127.0.0.1", port=5001)
```

---

## 📊 Understanding the Output CSV

Example session file: `session_20251219_114954.csv`

| Time | Learning_State |
|------|---|
| 11:49:55 | Attentive |
| 11:49:56 | Engaged |
| 11:49:57 | Attentive |

### Analysis Options
```bash
# View raw CSV
type analytics\session_20251219_114954.csv

# Generate plot from existing CSV
python src/plot_analytics.py analytics/session_20251219_114954.csv
```

---

## 🔐 Security Notes
- This is a **local development server** only
- Not suitable for production use
- Webcam access is requested per browser session
- No data is uploaded to external servers (except your system)
- For production, use proper WSGI server (Gunicorn, etc.)

---

## 📈 Next Steps

### Enhancements you can add:
1. **Multiple User Support**: Track individual faces over time
2. **Export Reports**: Generate PDF reports with analytics
3. **Real-time Notifications**: Alert when engagement drops
4. **Database Integration**: Store sessions in SQLite/PostgreSQL
5. **Advanced Visualizations**: More detailed charts and graphs
6. **Time-series Analysis**: Track engagement trends
7. **Mobile Support**: Responsive design for tablets/phones

---

## 📞 Support

For issues or questions:
1. Check the console (F12 → Console tab in browser)
2. Check terminal output where Flask is running
3. Review error messages for debugging
4. Ensure all dependencies are installed

---

## ✅ Quick Test Checklist

- [ ] Flask server starts without errors
- [ ] Browser can open `http://127.0.0.1:5000`
- [ ] Webcam access permission granted
- [ ] Face is detected in video feed
- [ ] Emotion badges appear in real-time
- [ ] Engagement % changes with different expressions
- [ ] Session can be started and ended
- [ ] CSV file is created after session
- [ ] Plot is generated after session
- [ ] New plot displays on dashboard

---

## 🎉 Congratulations!

You've successfully integrated your emotion detection system into a web application! 

Now you can:
- ✅ Run emotion analysis through a web browser
- ✅ Record and analyze classroom sessions
- ✅ Generate visual reports automatically
- ✅ Share dashboard with others on local network

Enjoy your Education Emotion Analytics platform! 📊😊
