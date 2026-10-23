"""
evaluate_model.py
Trains the Dark AI intent classifier and generates a full classification report.
Saves results to the 'results/' directory.
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nlp_utils import preprocess_text

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open("intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)["intents"]

X, y = [], []
for intent in intents_data:
    for pattern in intent["patterns"]:
        X.append(preprocess_text(pattern))
        y.append(intent["tag"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=200, C=1.5)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print(f"\n{'='*50}")
print(f"  Dark AI — Model Evaluation Results")
print(f"{'='*50}")
print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
print(f"{'='*50}\n")
print("Classification Report:\n")
print(report)

report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Dark AI — Classification Report\n{'='*50}\n")
    f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(report)

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.npy")
np.save(cm_path, cm)

labels_path = os.path.join(RESULTS_DIR, "class_labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(list(model.classes_), f)

samples_path = os.path.join(RESULTS_DIR, "samples_per_intent.json")
from collections import Counter
counts = Counter(y)
with open(samples_path, "w", encoding="utf-8") as f:
    json.dump(dict(counts), f, indent=2)

print(f"Results saved to '{RESULTS_DIR}/' directory.")
print(f"  - {report_path}")
print(f"  - {cm_path}")
print(f"  - {labels_path}")
print(f"  - {samples_path}")
print("\nRun 'python plot_results.py' to generate charts.")

from evaluate_model import evaluate_model
evaluate_model()