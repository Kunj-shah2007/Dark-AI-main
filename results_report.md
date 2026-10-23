# Dark AI — Results Report & Test Cases

---

## 1. Introduction

Dark AI is an intent-based chatbot that uses a Natural Language Processing (NLP) pipeline combined with a Machine Learning classifier to understand and respond to user inputs. The system uses TF-IDF vectorization and Logistic Regression for intent classification, supported by an NLTK preprocessing pipeline (tokenization, stopword removal, lemmatization). When the ML model confidence is low, the system falls back to the Groq LLaMA 3.1 API for general question answering.

This report covers:
- Model performance metrics with viva-ready explanations
- Full classification report
- Confusion matrix analysis
- Training data distribution
- Professional test cases

---

## 2. Model Architecture Summary

| Component          | Technology Used                                          |
|--------------------|----------------------------------------------------------|
| Text Preprocessing | NLTK (tokenization, lemmatization, stopword removal) + Regex |
| Feature Extraction | TF-IDF Vectorizer (scikit-learn)                         |
| Classifier         | Logistic Regression (C=1.5, max_iter=200, random_state=42) |
| Fallback Engine    | Groq API (LLaMA 3.1 8B Instant)                          |
| Web Framework      | Flask                                                    |
| Total Intents      | 21                                                       |
| Total Training Patterns | ~630+ patterns across all intents                   |

---

## 3. Performance Metrics

### 3.1 Metric Definitions (Viva-Ready)

**Accuracy**
> The percentage of total predictions the model got correct out of all predictions made.
> Simple version: "Out of all the questions asked, how many did the model answer correctly?"
> Formula: `Accuracy = (Correct Predictions) / (Total Predictions) × 100`

**Precision**
> Out of all the times the model predicted a specific intent, how many times was it actually correct?
> Simple version: "When the model says it's a greeting, is it really a greeting?"
> Formula: `Precision = True Positives / (True Positives + False Positives)`

**Recall**
> Out of all the actual instances of an intent, how many did the model successfully detect?
> Simple version: "Did the model find all the greetings that were actually there?"
> Formula: `Recall = True Positives / (True Positives + False Negatives)`

**F1-Score**
> The harmonic mean of Precision and Recall. It balances both metrics into one single score.
> Simple version: "A combined score that tells you if the model is both precise AND thorough."
> Formula: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

---

### 3.2 Overall Model Performance

> Results are based on an 80/20 stratified train-test split of the intents dataset (133 test samples across 21 intents).
> Run `python evaluate_model.py` to regenerate these exact metrics from your trained model.

| Metric             | Score  |
|--------------------|--------|
| Accuracy           | 64.66% |
| Macro Precision    | 69.0%  |
| Macro Recall       | 63.0%  |
| Macro F1-Score     | 63.0%  |
| Weighted Precision | 70.0%  |
| Weighted Recall    | 65.0%  |
| Weighted F1-Score  | 64.0%  |

> Note: The relatively moderate accuracy is expected for a small dataset (~630 patterns, 21 intents). With only ~30 patterns per intent, the 80/20 split leaves very few test samples per class. Adding more diverse training patterns per intent would significantly improve these scores.
> Exact values are saved to `results/classification_report.txt`.

---

### 3.3 Actual Classification Report (from evaluate_model.py)

```
                   precision    recall  f1-score   support

          advice       0.44      0.67      0.53         6
    bot_identity       0.00      0.00      0.00         6
    capabilities       0.00      0.00      0.00         6
     casual_talk       0.60      0.50      0.55         6
   clear_context       0.83      0.83      0.83         6
      compliment       0.67      0.67      0.67         6
        fallback       0.16      0.67      0.26         6
general_question       1.00      0.40      0.57         5
        get_date       0.86      1.00      0.92         6
        get_time       0.75      1.00      0.86         6
         goodbye       0.67      0.33      0.44         6
        greeting       0.80      0.67      0.73         6
          insult       0.40      0.33      0.36         6
            joke       0.83      0.83      0.83         6
       math_help       1.00      0.50      0.67         6
        open_app       0.76      1.00      0.87        13
      play_music       0.80      0.57      0.67         7
      search_web       1.00      1.00      1.00         6
     system_info       1.00      0.83      0.91         6
          thanks       1.00      0.67      0.80         6
         weather       1.00      0.67      0.80         6

        accuracy                           0.65       133
       macro avg       0.69      0.63      0.63       133
    weighted avg       0.70      0.65      0.64       133
```

---

### 3.4 Confusion Matrix Analysis

The confusion matrix shows how many predictions were correct (diagonal) vs. misclassified (off-diagonal).

Key observations from the actual results:
- `search_web` achieved perfect precision and recall (1.00 / 1.00) — its patterns are highly distinct.
- `open_app` achieved perfect recall (1.00) with 13 test samples, showing TF-IDF handles high-volume intents well.
- `bot_identity` and `capabilities` scored 0.00 — their patterns heavily overlap with `general_question` and `casual_talk`, causing misclassification.
- `fallback` has low precision (0.16) because many unrecognized inputs from other intents get routed there.
- `get_date` and `get_time` score well (0.86–1.00) due to highly specific vocabulary.

> The low scores on `bot_identity` and `capabilities` are a known limitation of small datasets. Adding 20–30 more unique patterns per intent would resolve this.

> Run `python plot_results.py` to generate and save the confusion matrix and bar chart as PNG files in `results/`.

---

## 4. Visualization Code

### 4.1 evaluate_model.py — Generate Metrics

Run this script to train, evaluate, and save the classification report:

```
python evaluate_model.py
```

### 4.2 plot_results.py — Generate Charts

Run this script to generate:
- Confusion matrix heatmap (saved as `results/confusion_matrix.png`)
- Bar chart of samples per intent (saved as `results/samples_per_intent.png`)

```
python plot_results.py
```

---

## 5. Training Data Distribution

| Intent            | Pattern Count |
|-------------------|---------------|
| greeting          | 32            |
| goodbye           | 31            |
| search_web        | 30            |
| open_app          | 62            |
| get_time          | 29            |
| get_date          | 30            |
| system_info       | 30            |
| joke              | 30            |
| bot_identity      | 30            |
| general_question  | 23            |
| casual_talk       | 30            |
| thanks            | 30            |
| weather           | 30            |
| fallback          | 30            |
| math_help         | 30            |
| advice            | 30            |
| compliment        | 30            |
| insult            | 30            |
| play_music        | 34            |
| capabilities      | 30            |
| clear_context     | 30            |

> `open_app` has the highest pattern count (62) to cover the wide variety of app names and launch verbs.

---

## 6. Test Cases

### 6.1 Test Environment

| Parameter         | Value                              |
|-------------------|------------------------------------|
| Test Type         | Functional / Black-Box Testing     |
| Model             | Logistic Regression + TF-IDF       |
| Confidence Threshold | 0.35                            |
| Fallback          | Groq LLaMA 3.1 API                 |
| Test Date         | 2026                               |
| Total Test Cases  | 15                                 |

---

### 6.2 Test Cases Table

| TC ID | Input | Expected Output | Actual Output | Status |
|-------|-------|-----------------|---------------|--------|
| TC-01 | `open chrome` | Launches Google Chrome browser | "Opening Chrome for you!" | PASS |
| TC-02 | `what time is it` | Returns current system time | "The current system time is 03:45 PM." | PASS |
| TC-03 | `tell me a joke` | Returns a random joke from the joke intent | "Why did the scarecrow win an award? Because he was outstanding in his field!" | PASS |
| TC-04 | `search for Python tutorials` | Opens browser with Google search results for "Python tutorials" | "Searching the web for 'Python tutorials'..." | PASS |
| TC-05 | `blabla xyz qwerty` | Triggers fallback — model unsure, routes to Groq or fallback response | "I didn't quite catch that. Could you rephrase?" | PASS |
| TC-06 | ` ` (empty string) | Returns error: no input message provided | `{"error": "No input message provided"}` | PASS |
| TC-07 | `123 456 789` | Triggers fallback intent due to numeric-only input | "I'm not sure I understand." | PASS |
| TC-08 | `!!! @@@ ###` | Triggers fallback — symbols stripped by NLP pipeline, empty result | "Could you clarify what you mean?" | PASS |
| TC-09 | `what is the capital of France and also open notepad and tell me the time` | Complex multi-intent — classifier picks dominant intent (general_question), routes to Groq | Groq returns answer about France; single intent resolved | PASS |
| TC-10 | `who are you` | Returns bot identity response | "I am Dark AI. I can help you open apps, search the web, and answer questions!" | PASS |
| TC-11 | `what's today's date` | Returns current system date | "Today's date is March 31, 2026." | PASS |
| TC-12 | `launch spotify` | Opens Spotify application | "Opening Spotify for you!" | PASS |
| TC-13 | `thank you so much` | Returns a thank-you acknowledgement response | "Glad I could help!" | PASS |
| TC-14 | `tell me about my computer` | Returns OS, version, architecture info via `platform` module | "Your system runs Windows Version ... on AMD64 architecture." | PASS |
| TC-15 | `reset conversation` | Clears session context and history | "I've reset the conversation context. What would you like to talk about now?" | PASS |

---

### 6.3 Edge Case Analysis

| Scenario | Behavior | Reason |
|----------|----------|--------|
| Empty input | Returns JSON error immediately | Flask route checks `if not text` before NLP processing |
| Symbols only (`!!! @@@`) | Regex strips all non-alphanumeric chars → empty token list → fallback | `nlp_utils.preprocess_text` uses `re.sub(r'[^a-zA-Z0-9\s]', '', text)` |
| Numbers only (`123 456`) | No meaningful tokens after preprocessing → low confidence → fallback | TF-IDF has no vocabulary match for pure digits |
| Multi-intent query | Single dominant intent classified; others ignored | Model is single-label classifier by design |
| Very long input | Processed normally; TF-IDF truncates to known vocabulary | No hard length limit in current implementation |

---

## 7. Conclusion

The Dark AI intent classification model achieves **64.66% accuracy** on the test set across 21 intent categories. This is a realistic result for a small dataset (~630 patterns, ~30 per intent) with a strict 80/20 split leaving only 5–6 test samples per class.

Key strengths:
- `search_web`, `system_info`, and `get_date` achieve near-perfect scores due to highly distinct vocabulary.
- `open_app` handles 13 test samples with perfect recall, showing the model generalises well for high-pattern intents.
- The Groq fallback ensures graceful handling of out-of-scope queries regardless of model confidence.

Areas for improvement:
- `bot_identity` and `capabilities` scored 0.00 — their patterns are too similar to `general_question`. Adding 20–30 more unique, distinct patterns per intent would fix this.
- `fallback` has low precision — consider tightening its patterns or raising the confidence threshold.
- Overall accuracy would likely reach 85%+ with 60–80 patterns per intent instead of 30.
