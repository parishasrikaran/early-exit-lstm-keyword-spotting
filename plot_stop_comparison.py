import json
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("results/learned_stop_comparison.json")
data = json.loads(path.read_text())

existing = data["existing_policy"]
learned = data["learned_stop_head"]

metrics = [
    ("early_keyword_recall", "Early Keyword Recall", 100),
    ("early_miss_rate", "Early Miss Rate", 100),
    ("early_negative_trigger_rate", "Negative Trigger Rate", 100),
    ("mean_emission_ms", "Mean Emission Time", 1)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (key, title, scale) in zip(axes.flat, metrics):
    values = [existing[key] * scale, learned[key] * scale]

    bars = ax.bar(
        ["Existing Policy", "Learned StopHead"],
        values,
        color=["steelblue", "darkorange"]
    )

    ax.set_title(title)
    ax.set_ylabel("Milliseconds" if scale == 1 else "Percentage (%)")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig("results/stop_head_comparison.png", dpi=300)

print("Saved: results/stop_head_comparison.png")
plt.show()