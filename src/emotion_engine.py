from datetime import datetime
import os

def run_emotion_session():
    # ✅ CREATE SESSION ID FIRST
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    csv_path = os.path.join("analytics", f"{session_id}.csv")
    plot_path = os.path.join("analytics", "plots", f"{session_id}_stylish.png")

    # ---- your webcam + emotion logic here ----
    # save csv to csv_path
    # generate plot to plot_path
    # -----------------------------------------

    return session_id
