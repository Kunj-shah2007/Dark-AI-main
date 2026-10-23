import json
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nlp_utils import preprocess_text

class IntentClassifier:
    def __init__(self, data_path="intents.json", model_dir="model_cache"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression(random_state=42, max_iter=200, C=1.5)
        self.intents_data = None
        self.is_trained = False
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.intents_data = json.load(f)['intents']

    def train(self):
        print("Loading intents data and creating NLP pipeline...")
        self.load_data()
        X = []
        y = []
        
        for intent in self.intents_data:
            tag = intent['tag']
            for pattern in intent['patterns']:
                preprocessed = preprocess_text(pattern)
                X.append(preprocessed)
                y.append(tag)
                
        print("Vectorizing Text data using TF-IDF...")
        X_vectors = self.vectorizer.fit_transform(X)
        
        print("Training Logistic Regression Model...")
        self.classifier.fit(X_vectors, y)
        self.is_trained = True
        
        # Save models
        with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.model_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
            
        print("NLP Model trained and cached successfully.")

    def load_model(self):
        vec_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        clf_path = os.path.join(self.model_dir, 'classifier.pkl')
        
        if os.path.exists(vec_path) and os.path.exists(clf_path):
            with open(vec_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(clf_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.is_trained = True
            return True
        return False
        
    def predict(self, text):
        if not self.is_trained:
            if not self.load_model():
                raise Exception("Model weights not found. You must train the model first.")
                
        # 1. Preprocess the incoming text using NLTK
        preprocessed = preprocess_text(text)
        
        # 2. Vectorize the string using TF-IDF 
        vec = self.vectorizer.transform([preprocessed])
        
        # 3. Predict mapping and confidence
        probabilities = self.classifier.predict_proba(vec)[0]
        max_prob_idx = np.argmax(probabilities)
        
        confidence = probabilities[max_prob_idx]
        predicted_tag = self.classifier.classes_[max_prob_idx]
        
        return predicted_tag, float(confidence)
