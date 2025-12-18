import csv
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ===============================
# INPUT CHECK
# ===============================
if len(sys.argv) < 2:
    print("Usage: python plot_analytics_stylish.py analytics/session_xxx.csv")
    sys.exit(1)

csv_file = sys.argv[1]

# ===============================
# LOAD DATA
# ===============================
states = []

with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        states.append(row["Learning_State"])

# ===============================
# COUNT STATES
# ===============================
state_counts = defaultdict(int)
for s in states:
    state_counts[s] += 1

total = sum(state_counts.values())

# ===============================
# SORT FOR NICE DISPLAY
# ===============================
order = [
    "Engaged",
    "Interested",
    "Attentive",
    "Confused",
    "Frustrated",
    "Anxious",
    "Disengaged"
]

counts = [state_counts.get(k, 0) for k in order]
percents = [(c / total * 100) if total else 0 for c in counts]

# ===============================
# COLOR THEME
# ===============================
colors = [
    "#2ecc71",  # Engaged - green
    "#3498db",  # Interested - blue
    "#f1c40f",  # Attentive - yellow
    "#e67e22",  # Confused - orange
    "#e74c3c",  # Frustrated - red
    "#9b59b6",  # Anxious - purple
    "#95a5a6"   # Disengaged - grey
]

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(10, 6))
bars = plt.bar(order, counts, color=colors, edgecolor="black")

plt.title("Learning State Distribution", fontsize=16, fontweight="bold")
plt.ylabel("Number of Observations", fontsize=12)
plt.xlabel("Learning State", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# ===============================
# LABEL BARS
# ===============================
for bar, p in zip(bars, percents):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{p:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )


plt.tight_layout()

# ===============================
# SAVE PLOT
# ===============================
os.makedirs("analytics/plots", exist_ok=True)

plot_path = os.path.join(
    "analytics/plots",
    os.path.basename(csv_file).replace(".csv", "_stylish.png")
)


plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Stylish plot saved to: {plot_path}")
