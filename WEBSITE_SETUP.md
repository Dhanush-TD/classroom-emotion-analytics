# Education Emotion Analytics Dashboard - Web Setup Guide

## Overview

The `webcam_emotion_education.py` functionality has been successfully integrated into a Flask web dashboard. The system now provides real-time emotion detection and learning state analysis through a web interface.

## What's Changed

### 1. **Enhanced `emotion_engine.py`**
   - Added `LEARNING_STATE_MAP` to convert emotions to education learning states:
     - Happy → Engaged
     - Neutral → Attentive
     - Surprise → Interested
     - Sad → Confused
     - Angry → Frustrated
     - Fear → Anxious
     - Disgust → Disengaged
   
   - Added `ENGAGEMENT_WEIGHT` to calculate engagement percentage
   - Updated `predict_frame()` to return learning states and engagement score

### 2. **Improved Flask App (`app.py`)**
   - POST-based endpoints for better HTTP practices
   - Proper session management with CSV file handling
   - Real-time data collection and storage
   - Automatic plot generation using `plot_analytics.py`
   - Error handling and logging

### 3. **Enhanced User Interface**
   - Modern, responsive dashboard design
   - Real-time engagement percentage display
   - Live learning state detection badges
   - Session status indicator
   - Plot visualization of past sessions
   - Mobile-friendly layout

## Running the Dashboard

### Prerequisites
Ensure you have activated the virtual environment:
```bash
# On Windows PowerShell
cd "d:\Git Hub Data\emotion_project"
.\fer_env\Scripts\Activate.ps1

# Or on Command Prompt
.\fer_env\Scripts\activate
```

### Start the Web Server
```bash
cd analytics\dashboard
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## Using the Dashboard

### Start a Session
1. Click the **▶ Start Session** button
2. Allow the browser to access your webcam when prompted
3. The system will begin capturing frames and detecting emotions

### During Session
- **Engagement Percentage**: Shows real-time engagement score (0-100%)
- **Current States Detected**: Displays detected learning states in real-time with color-coded badges
- **Learning States**:
  - 🟢 **Engaged** (Green) - Highest engagement
  - 🔵 **Interested** (Blue) - High engagement
  - 🟡 **Attentive** (Yellow) - Medium engagement
  - 🟠 **Confused** (Orange) - Lower engagement
  - 🔴 **Frustrated** (Red) - Low engagement
  - 🟣 **Anxious** (Purple) - Low engagement
  - ⚫ **Disengaged** (Gray) - No engagement

### End Session
1. Click the **⏹ End Session** button
2. The system will:
   - Save all collected data to a CSV file
   - Generate a statistical plot
   - Automatically reload the page to show the new plot

## Output Files

### CSV Files
Located in: `analytics/session_YYYYMMDD_HHMMSS.csv`

Format:
```
Time,Learning_State
14:30:45,Attentive
14:30:46,Attentive
14:30:47,Interested
...
```

### Plot Files
Located in: `analytics/plots/session_YYYYMMDD_HHMMSS_stylish.png`

Shows:
- Bar chart of learning state distribution
- Percentage breakdown for each state
- Color-coded visualization

## System Architecture

```
Browser (Webcam Input)
    ↓
Flask App (/predict endpoint)
    ↓
emotion_engine.py (Face detection + Emotion prediction)
    ↓
Learning State Conversion
    ↓
Engagement Calculation
    ↓
CSV Storage + Real-time Update
    ↓
(On Session End) Plot Generation
    ↓
Dashboard Display
```

## Key Features

✅ **Real-time Detection**: Processes frames every 500ms
✅ **Multi-face Support**: Tracks multiple people simultaneously
✅ **Smoothing Algorithm**: Reduces false detections with 3-frame averaging
✅ **Educational Mapping**: Converts emotions to learning states
✅ **Engagement Scoring**: Calculates engagement based on detected states
✅ **CSV Logging**: Maintains detailed session records
✅ **Plot Generation**: Creates visual analytics automatically
✅ **Responsive UI**: Works on desktop and mobile devices

## Troubleshooting

### Webcam Not Accessible
- Check browser permissions for camera access
- Ensure no other application is using the webcam
- Try a different browser (Chrome recommended)

### No Faces Detected
- Ensure adequate lighting
- Get closer to the camera (within 2-3 feet)
- Position face directly toward camera
- Clear any obstructions

### Plot Not Generated
- Check if `analytics/plots` directory exists
- Verify plot_analytics.py is in the correct location
- Check console for error messages

### CSV File Not Saved
- Verify write permissions for the `analytics` directory
- Check disk space availability
- Ensure session was properly ended

## Performance Tips

1. **Lighting**: Use good lighting conditions for better face detection
2. **Distance**: Keep 2-3 feet from camera for optimal detection
3. **Frame Rate**: System processes frames every 500ms for balance between accuracy and responsiveness
4. **Multi-face**: The system can track up to 10+ faces in a single frame

## Development Notes

- Model: Swin Transformer (Tiny) fine-tuned on RAF-DB dataset
- Face Detection: MTCNN (Multi-task Cascaded Convolutional Networks)
- Smoothing: 3-frame moving average for emotion predictions
- Device: Automatically uses GPU if available, otherwise CPU

## File Structure
```
emotion_project/
├── analytics/
│   ├── dashboard/
│   │   ├── app.py                 (Main Flask app)
│   │   ├── static/
│   │   │   └── style.css          (Dashboard styling)
│   │   └── templates/
│   │       └── index.html         (Web interface)
│   ├── session_*.csv              (Session data)
│   └── plots/
│       └── session_*_stylish.png  (Generated plots)
├── src/
│   ├── emotion_engine.py          (Core prediction logic)
│   ├── config.py                  (Emotion mappings)
│   ├── plot_analytics.py          (Plot generation)
│   └── ... (other scripts)
└── checkpoints/
    └── swin_t_rafdb_finetuned.pth (Model weights)
```

## Next Steps

- Customize the engagement weight formula in `emotion_engine.py`
- Adjust frame capture frequency (currently 500ms)
- Implement database storage instead of CSV
- Add historical data comparison features
- Create teacher/admin dashboard for multiple sessions

---

**Version**: 1.0
**Last Updated**: December 2025
