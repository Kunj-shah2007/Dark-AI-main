"""
plot_results.py
Generates confusion matrix and samples-per-intent bar chart for Dark AI.
Run evaluate_model.py first to generate the required data files.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "results"

def load_results():
    cm = np.load(os.path.join(RESULTS_DIR, "confusion_matrix.npy"))
    with open(os.path.join(RESULTS_DIR, "class_labels.json"), "r") as f:
        labels = json.load(f)
    with open(os.path.join(RESULTS_DIR, "samples_per_intent.json"), "r") as f:
        samples = json.load(f)
    return cm, labels, samples


def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(14, 11))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    threshold = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=7)

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Dark AI — Intent Classification Confusion Matrix", fontsize=13, pad=15)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_samples_per_intent(samples):
    intents = list(samples.keys())
    counts = list(samples.values())

    sorted_pairs = sorted(zip(counts, intents), reverse=True)
    counts, intents = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(intents, counts, color="#4C72B0", edgecolor="white", linewidth=0.6)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Intent", fontsize=11)
    ax.set_ylabel("Number of Patterns", fontsize=11)
    ax.set_title("Dark AI — Training Samples per Intent", fontsize=13, pad=15)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "samples_per_intent.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Samples per intent chart saved: {save_path}")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        print("ERROR: 'results/' directory not found.")
        print("Please run 'python evaluate_model.py' first.")
        exit(1)

    required = ["confusion_matrix.npy", "class_labels.json", "samples_per_intent.json"]
    for fname in required:
        if not os.path.exists(os.path.join(RESULTS_DIR, fname)):
            print(f"ERROR: Missing file: results/{fname}")
            print("Please run 'python evaluate_model.py' first.")
            exit(1)

    cm, labels, samples = load_results()
    plot_confusion_matrix(cm, labels)
    plot_samples_per_intent(samples)
    print("\nAll charts generated successfully.")
