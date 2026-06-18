import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def _ensure_directory(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_emotion_pie(history, output_path):
    _ensure_directory(output_path)
    counts = {}
    for row in history:
        emotion = row[1] or "Unknown"
        counts[emotion] = counts.get(emotion, 0) + 1

    if not counts:
        counts = {"Unknown": 1}

    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = plt.get_cmap("tab20")(range(len(labels)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors, startangle=140)
    ax.set_title("Emotion Distribution")
    ax.axis("equal")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_head_pose_pie(history, output_path):
    _ensure_directory(output_path)
    counts = {}
    for row in history:
        head_pose = row[6] or "Unknown"
        counts[head_pose] = counts.get(head_pose, 0) + 1

    if not counts:
        counts = {"Center": 1}

    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ["#42c26a" if label == "Center" else "#f1c40f" if label in {"Left", "Right"} else "#9bb0cf" for label in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors, startangle=140)
    ax.set_title("Head Pose Distribution")
    ax.axis("equal")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_age_histogram(history, output_path):
    _ensure_directory(output_path)
    ages = []
    for row in history:
        try:
            ages.append(int(row[2]))
        except (TypeError, ValueError):
            continue

    if not ages:
        ages = [25]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ages, bins=range(min(ages), max(ages) + 2, 2), color="#4f8cff", edgecolor="#ffffff")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
