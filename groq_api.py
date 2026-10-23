import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Groq API Configuration
GROQ_API_KEY = "gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your actual Groq API key

def ask_groq(prompt, context=None):
    """
    Calls the Groq API as a fallback when NLP model confidence is low.
    Only triggered for genuinely unknown queries.
    """
    if not GROQ_API_KEY:
        return "My Groq fallback is not configured. Please set your API key."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Build a single clean system message (Groq only allows one system role)
    system_content = "You are Dark AI, a helpful assistant. Give concise, direct answers in 1-3 sentences."
    if context and context.strip() and context != "None":
        system_content += f" Prior context: {context[:300]}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 256
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 400:
            logger.error(f"Groq 400 error: {response.text}")
            return "I couldn't find a confident answer in my knowledge base for that."
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "The fallback service timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "Cannot reach the fallback service. Check your internet connection."
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return "I encountered an issue with the fallback service. Please try again."
