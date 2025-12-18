from flask import Flask, render_template
import csv
import os
from collections import defaultdict

app = Flask(__name__)

ANALYTICS_DIR = "analytics"

def get_latest_csv():
    files = [f for f in os.listdir(ANALYTICS_DIR) if f.endswith(".csv")]
    files.sort(reverse=True)
    return os.path.join(ANALYTICS_DIR, files[0]) if files else None

@app.route("/")
def index():
    csv_file = get_latest_csv()
    state_counts = defaultdict(int)
    total = 0

    if csv_file:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = row["Learning_State"]
                state_counts[state] += 1
                total += 1

    engagement_weights = {
        "Engaged": 1.0,
        "Interested": 0.8,
        "Attentive": 0.6,
        "Confused": 0.3,
        "Frustrated": 0.2,
        "Anxious": 0.2,
        "Disengaged": 0.0
    }

    engagement_score = 0
    if total:
        score_sum = sum(state_counts[s] * engagement_weights.get(s, 0)
                        for s in state_counts)
        engagement_score = int((score_sum / total) * 100)

    return render_template(
        "index.html",
        counts=dict(state_counts),
        engagement=engagement_score
    )

if __name__ == "__main__":
    app.run(debug=True)
