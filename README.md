# Dark AI - Advanced NLP Chatbot

A production-ready, locally-driven AI assistant named **Dark AI**. It uses a hybrid architecture that prioritizes classical Machine Learning for performance and speed, while using Large Language Models (Groq API) for realistic, conversational refinement and fallbacks.

## 🚀 Key Features

*   **Hybrid Intelligence**: Uses a fast Scikit-learn (TF-IDF + Logistic Regression) model for intent classification and Groq (LLama 3.1) for conversational paraphrasing and deep knowledge.
*   **Realistic Responses**: Every system action handled by Dark AI is paraphrased by an LLM to provide a "premium," realistic conversational experience.
*   **Zero-Dependency NLP**: Completely removed heavy libraries like `spaCy` and `torch` to ensure 100% stability on Windows and avoid DLL initialization crashes (`c10.dll`).
*   **System Integration**: 
    *   **Open Apps**: Native support for Chrome, Notepad, Spotify, VS Code, and many more.
    *   **Music Control**: Intelligent routing to Spotify for music requests or YouTube for specific song searches.
    *   **Web Searching**: Google search integration with dynamic query extraction.
*   **Advanced Decision Engine**: Tiered confidence scoring ensures that simple commands are processed locally (~10ms) while complex questions are routed to the cloud.

---

## 📂 Project Architecture

```
advanced_chatbot/
├── app.py                # Main Flask Server & Decision Engine
├── model.py              # ML Intent Classifier (Training & Prediction)
├── nlp_utils.py          # Robust NLTK + Regex Preprocessing (No Torch/SpaCy)
├── actions.py            # System/Plugin Action Implementation
├── groq_api.py           # Fallback & Paraphraser LLM Wrapper
├── intents.json          # 20+ Intent Dataset with 400+ Patterns
├── model_cache/          # Saved ML model weights and vectorizers
├── requirements.txt      # Lightweight dependencies
├── static/               # CSS & JavaScript for the Chat UI
└── templates/            # HTML base for the Web Interface
```

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. This system is optimized for Windows.

```bash
pip install -r requirements.txt
```

### 2. Configure Groq API
Open `groq_api.py` and ensure the `GROQ_API_KEY` is set. This is required for the "Realistic" responses and factual fallbacks.

### 3. (Optional) Manual Training
The system automatically trains on its first run, but you can force-retrain the ML model if you update `intents.json`:

```python
# Open Python console or create a script
from model import IntentClassifier
c = IntentClassifier()
c.train()
```

### 4. Run Dark AI
Start the Flask application:

```bash
python app.py
```

Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛡️ Developer Note
This version is developed by **Kunj Shah** (distributed as **Dark AI**). It was built to solve the fragmentation of modern AI assistants by combining local ML speed with LLM intelligence.
