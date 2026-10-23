# Dark AI - Modern Architecture & Quickstart
**Developer:** Kunj Shah

The system applies a hybrid ML-First approach, leveraging local performance with LLM-based realistic refinement.

## 🛠️ Installation & Setup
1. **Python**: Ensure Python 3.9+ is installed.
2. **Library Installation**:
   ```bash
   pip install -r requirements.txt
   ```
   *(The system will automatically download necessary NLTK data on first run).*
3. **Groq API**:
   Open `groq_api.py` and ensure the API key is present: `GROQ_API_KEY="..."`. This is required for our Realistic Paraphrasing and LLM fallback feature to work.

## 🚀 Running the Bot!
Simply execute `app.py`. The system will detect if a model exists—if not, it trains automatically (~3 seconds) and starts the server:
```bash
python app.py
```
**Web Interface**: Access the chat at `http://127.0.0.1:5000`.

---

## 🏗️ Architecture: Hybrid Confidence Scoring
Dark AI uses a logic-gated decision engine in `app.py`:

*   **Zone 1: Local ML Execution (High Confidence >= 0.35)**
    *   The bot identifies your **intent** (e.g., `open_app`, `play_music`, `search_web`).
    *   It extracts relevant **entities** (e.g., `"Chrome"`, `"Spotify"`, `"quantum physics"`) using regex and NLTK.
    *   It executes the **native system action** (via `actions.py`).
    *   **REALISM**: Finally, it passes the action result to the **Groq LLM** to generate a natural, conversational response confirming success.

*   **Zone 2: Direct Fallback (Low Confidence < 0.35 or Unknown)**
    *   If the local model doesn't recognize the input (e.g., *"Explain quantum entanglement"*), it routes the query directly to Groq.
    *   All deep knowledge questions and factual requests are handled by the **LLama 3.1 8B** model.

## 📁 Key File Mapping
- `nlp_utils.py`: Stable NLTK-based preprocessor (No Torch or SpaCy dependencies).
- `actions.py`: System commands for Windows (Apps, Music, Time, Web).
- `intents.json`: The training dataset for the Local ML model.
- `groq_api.py`: Clean wrapper for high-speed LLM fallback.
- `app.py`: The brain of the bot, managing sessions, memory, and routing.
