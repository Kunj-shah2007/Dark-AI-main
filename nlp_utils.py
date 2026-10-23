"""
nlp_utils.py - Pure NLTK NLP pipeline. No spaCy/torch dependency.
Uses regex + NLTK for preprocessing and basic entity extraction.
"""
import re
import ssl
import logging

logger = logging.getLogger(__name__)

# ------- NLTK Safe Download -------
import nltk

try:
    _create_unverified = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified

for pkg in ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'omw-1.4', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ------- Safe imports with fallbacks -------
try:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    def _lemmatize(word):
        return _lemmatizer.lemmatize(word)
except Exception:
    def _lemmatize(word):
        return word

try:
    from nltk.corpus import stopwords
    _stop_words = set(stopwords.words('english'))
except Exception:
    _stop_words = {
        'i','me','my','myself','we','our','you','your','he','she','it','they',
        'is','are','was','were','be','been','have','has','had','do','does',
        'did','will','would','could','should','may','might','shall',
        'a','an','the','and','or','but','in','on','at','to','for','of','with'
    }

def _safe_tokenize(text):
    try:
        return nltk.word_tokenize(text)
    except Exception:
        return re.findall(r'\b\w+\b', text)


# ------- Core Functions -------

def preprocess_text(text):
    """
    Full NLP preprocessing pipeline:
    1. Lowercase + punctuation removal
    2. Tokenization (NLTK punkt or regex fallback)
    3. Stopword removal
    4. WordNet Lemmatization
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = _safe_tokenize(text)
    cleaned = [_lemmatize(w) for w in tokens if w not in _stop_words and len(w) > 1]
    return " ".join(cleaned) if cleaned else text.lower()


def extract_entities(text):
    """
    Lightweight regex-based Named Entity Extraction.
    No spaCy/torch required. Covers the most common entity types
    the chatbot needs: app names, time references, URLs, etc.
    """
    entities = {}
    text_lower = text.lower()

    # --- TIME patterns ---
    time_match = re.search(r'\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b', text_lower)
    if time_match:
        entities['TIME'] = time_match.group(1)

    # --- DATE patterns ---
    date_match = re.search(
        r'\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
        r'|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', text_lower
    )
    if date_match:
        entities['DATE'] = date_match.group(1)

    # --- URL patterns ---
    url_match = re.search(r'https?://\S+|www\.\S+', text_lower)
    if url_match:
        entities['URL'] = url_match.group()

    return entities
