# 🚀 QUICK START GUIDE

## Start the Website in 3 Steps

### Step 1: Open PowerShell/Command Prompt
Navigate to your project folder:
```powershell
cd "D:\Git Hub Data\emotion_project"
```

### Step 2: Run this command
```powershell
.\fer_env\Scripts\Activate.ps1
python analytics/dashboard/app.py
```

Or simply double-click:
```
run_dashboard.ps1
```

### Step 3: Open in Browser
Go to: **http://127.0.0.1:5000**

---

## Using the Dashboard

### 🎬 Start Recording
1. Click **▶ Start Session** button
2. Allow webcam access when prompted
3. Watch the real-time emotion detection!

### 📊 See Results
1. Click **⏹ End Session** to finish
2. Your session gets saved to CSV
3. A plot chart is generated automatically
4. Refresh page to see your analytics plot

---

## 📁 Where Are My Files?

- **Session Data**: `analytics/session_YYYYMMDD_HHMMSS.csv`
- **Charts**: `analytics/plots/session_YYYYMMDD_HHMMSS_stylish.png`

---

## 🎨 Dashboard Shows:

✅ **Live Video** - Webcam feed  
✅ **Engagement %** - Overall engagement level  
✅ **Face Count** - How many people detected  
✅ **Learning States** - Engaged/Attentive/Confused etc  
✅ **Emotion % Chart** - Happy/Sad/Angry distribution  
✅ **Analytics Plot** - Final session visualization  

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam not working | Check browser permissions (Chrome Settings → Privacy) |
| Port 5000 in use | Change port in `app.py` line: `app.run(..., port=5001)` |
| Model not loading | Make sure `checkpoints/` folder exists |
| Slow/Laggy | Close other apps, or reduce frame rate |

---

## 📝 Important Files Modified

```
✅ app.py - Updated with full emotion detection
✅ index.html - Enhanced UI with emotion display
✅ style.css - Added emotion badge styling
```

All original functionality from `webcam_emotion_education.py` is now in the website!

---

Enjoy your emotion analytics dashboard! 🎓😊
