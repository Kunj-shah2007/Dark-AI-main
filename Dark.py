import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
import random
import re
from flask import Flask, request, jsonify, render_template
import json

from nlp_utils import extract_entities
from model import IntentClassifier
from actions import execute_action
from groq_api import ask_groq

app = Flask(__name__, template_folder='templates', static_folder='static')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load intents mapping
try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        INTENTS_DATA = json.load(f)['intents']
except FileNotFoundError:
    INTENTS_DATA = []

def get_intent_metadata(tag):
    for intent in INTENTS_DATA:
        if intent['tag'] == tag:
            return intent
    return {}

# Initialize Machine Learning System
logger.info("Initializing NLP Model architecture...")
classifier = IntentClassifier()

try:
    if not classifier.load_model():
        logger.warning("No pre-trained model found. Initiating Training Phase.")
        classifier.train()
except Exception as e:
    logger.error(f"Error loading model: {e}")

# Context Management Storage
SESSION_CONTEXT = {}

def extract_app_name(text):
    """Intelligently extract the app name from user text."""
    text_lower = text.lower().strip()
    OPEN_VERBS = ["please open", "please launch", "can you open", "can you launch",
                  "could you open", "open", "launch", "start", "run",
                  "fire up", "bring up", "boot up"]
    for verb in sorted(OPEN_VERBS, key=len, reverse=True):
        if text_lower.startswith(verb + " "):
            return text_lower[len(verb)+1:].strip()
    # If it's just 1-2 words like "notepad", return it directly
    if len(text_lower.split()) <= 2:
        return text_lower
    return None

def extract_song_name(text):
    """Intelligently extract the song/music query from user text."""
    text_lower = text.lower().strip()
    PLAY_VERBS = ["search youtube for ", "can you play ", "put on some ",
                  "put on ", "listen to ", "play some ", "play "]
    for verb in sorted(PLAY_VERBS, key=len, reverse=True):
        if text_lower.startswith(verb):
            result = text_lower[len(verb):].strip()
            if result:
                return result
    # For phrases like "i want to listen to X"
    match = re.search(r'(?:listen to|hear|play)\s+(.+)', text_lower)
    if match:
        return match.group(1).strip()
    return None

def extract_search_query(text):
    """Intelligently extract the search query from user text."""
    text_lower = text.lower().strip()
    SEARCH_VERBS = ["search the web for ", "search the internet for ", "search google for ",
                    "google ", "search for ", "look up ", "search online for ",
                    "find information about ", "find me details on ", "search "]
    for verb in sorted(SEARCH_VERBS, key=len, reverse=True):
        if text_lower.startswith(verb):
            result = text_lower[len(verb):].strip()
            if result:
                return result
    return None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    text = data.get('message', '')
    session_id = data.get('session_id', 'default_user')

    if not text:
        return jsonify({'error': 'No input message provided'})

    try:
        # Initialize session context
        if session_id not in SESSION_CONTEXT:
            SESSION_CONTEXT[session_id] = {
                "last_intent": None,
                "history": [],
                "entities": {}
            }
        session = SESSION_CONTEXT[session_id]

        # CLEAR old entities every new message to prevent loops
        session["entities"] = {}

        # 1. NLP Processing & ML Classification
        predicted_tag, confidence = classifier.predict(text)
        logger.info(f"Classified Intent: {predicted_tag} (Confidence: {confidence:.2f})")

        # 2. Entity Extraction (regex-based)
        entities = extract_entities(text)
        session["entities"].update(entities)
        logger.info(f"Extracted Entities: {entities}")

        # 3. Get intent metadata
        intent_meta = get_intent_metadata(predicted_tag)

        # 4. Smart Entity Extraction based on predicted intent
        if predicted_tag == "open_app" or (intent_meta.get("action") == "open_app"):
            app_name = extract_app_name(text)
            if app_name:
                session["entities"]["app_name"] = app_name

        elif predicted_tag == "play_music" or (intent_meta.get("action") == "play_music"):
            song = extract_song_name(text)
            if song:
                session["entities"]["song_name"] = song

        elif predicted_tag == "search_web" or (intent_meta.get("action") == "search_web"):
            query = extract_search_query(text)
            if query:
                session["entities"]["query"] = query

        # ========================================
        # 5. DECISION ENGINE
        # ========================================
        ML_THRESHOLD = 0.35
        raw_result = ""
        is_action_executed = False

        # Route 1: ML handling with Groq paraphraser for realism
        if confidence >= ML_THRESHOLD and predicted_tag not in ["fallback", "general_question"]:
            logger.info(f"ML intent match: {predicted_tag}")
            session["last_intent"] = predicted_tag
            action_name = intent_meta.get('action')

            if action_name == "ask_groq":
                raw_result = ask_groq(text, context=None)
                is_action_executed = True
            elif action_name:
                logger.info(f"Firing system action: {action_name}")
                raw_result = execute_action(action_name, session["entities"])
                is_action_executed = True
            else:
                responses = intent_meta.get('responses', ["I understand."])
                raw_result = random.choice(responses)
            
            # Generate a realistic response using Groq if possible
            if is_action_executed:
                 bot_response = ask_groq(f"The user said '{text}'. My intent classifier matched this to '{predicted_tag}'. The system action result was: '{raw_result}'. Generate a very brief (1 sentence), confident, and premium conversational response confirming this success.")
            else:
                 bot_response = raw_result

            if predicted_tag == "clear_context":
                session["entities"] = {}
                session["last_intent"] = None

        # Route 2: Unknown or Complex logic -> Direct Fallback to Groq
        else:
            logger.info(f"ML unsure. Fallback to Groq.")
            history_str = " ".join([h["text"] for h in session["history"][-3:]]) if session["history"] else None
            # Here it's a direct LLM response
            bot_response = ask_groq(text, context=history_str)
            session["last_intent"] = "groq_fallback"

        # Update session history
        session["history"].append({"role": "user", "text": text})

        return jsonify({
            "response": bot_response,
            "intent": predicted_tag,
            "confidence": round(confidence, 3),
            "entities": session["entities"]
        })

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
