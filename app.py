# app.py
import time
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from collections import Counter

app = Flask(__name__)

# OpenRouter API Setup
API_KEY = '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'
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Adjusted confidence thresholds with much stricter requirements
CONFIDENCE_THRESHOLDS = {
    'very_high': 0.85,  # Increased from 0.90
    'high': 0.75,       # Increased from 0.80
    'medium': 0.60,     # Decreased from 0.65
    'low': 0.35,        # Decreased from 0.40
    'very_low': 0.15    # Decreased from 0.20
}

WEIGHTS = {
    'probability': 0.45,
    'similarity': 0.30,
    'entropy': 0.15,
    'diversity': 0.10
}

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and prepare data with explicit encoding
faq_df = pd.read_csv("faq_dataset.csv", encoding='utf-8')
queries_df = pd.read_csv("farmer_queries.csv", encoding='utf-8')

# Drop any rows with NaN values
faq_df = faq_df.dropna()

# Reset index after cleaning
faq_df = faq_df.reset_index(drop=True)

print(f"Loaded {len(faq_df)} questions for training")

# Clean text for all rows
faq_df['question'] = faq_df['question'].apply(clean_text)
queries_df['questions'] = queries_df['questions'].apply(clean_text)

# Train Local Model with verbose output
vectorizer = TfidfVectorizer(min_df=1)  # Allow terms that appear in single documents
X = vectorizer.fit_transform(faq_df['question'])
y = faq_df['answers']

print(f"Training model with {X.shape[0]} examples and {X.shape[1]} features")

model = MultinomialNB()
model.fit(X, y)

# Function to retrain the model
def retrain_model(new_question, new_answer):
    global faq_df, X, y, model, vectorizer
    
    try:
        # Create new DataFrame with single row
        new_row = pd.DataFrame({
            'question': [clean_text(new_question)], 
            'answers': [new_answer]
        })
        
        # Concatenate with existing data
        faq_df = pd.concat([faq_df, new_row], ignore_index=True)
        
        # Save the updated dataset immediately
        faq_df.to_csv("faq_dataset.csv", index=False, encoding='utf-8')
        
        # Retrain the model with all data
        X = vectorizer.fit_transform(faq_df['question'])
        y = faq_df['answers']
        model.fit(X, y)
        
        print(f"Model retrained successfully with {len(faq_df)} examples")
        return True
        
    except Exception as e:
        print(f"Error during retraining: {str(e)}")
        return False

def calculate_entropy_score(probabilities):
    """Calculate normalized entropy of prediction probabilities"""
    entropy_val = entropy(probabilities)
    max_entropy = entropy([1/len(probabilities)] * len(probabilities))
    return 1 - (entropy_val / max_entropy) if max_entropy != 0 else 0

def get_prediction_diversity(pred_probas):
    """Calculate how diverse/concentrated the predictions are"""
    above_threshold = (pred_probas > 0.2).sum()
    return 1 - (above_threshold / len(pred_probas))

def analyze_confidence(pred_proba, similarity_score, pred_probas):
    """Enhanced confidence analysis with stricter thresholds"""
    # Calculate base confidence metrics
    entropy_score = calculate_entropy_score(pred_probas)
    diversity_score = get_prediction_diversity(pred_probas)
    
    # Weighted confidence calculation with adjusted weights
    confidence_score = (
        pred_proba * 0.5 +          # Increased weight on probability
        similarity_score * 0.3 +     # Kept same weight on similarity
        entropy_score * 0.1 +        # Reduced weight on entropy
        diversity_score * 0.1        # Reduced weight on diversity
    )

    # Strict validation checks
    if pred_proba < 0.2 or similarity_score < 0.15:
        return 'very_low', False, confidence_score

    # Use higher thresholds for local model
    for level, threshold in CONFIDENCE_THRESHOLDS.items():
        if confidence_score >= threshold:
            use_local = (level in ['very_high', 'high'] or 
                        (level == 'medium' and pred_proba > 0.65))
            return level, use_local, confidence_score

    return 'very_low', False, confidence_score

def get_similarity_score(query_vector, corpus_vectors):
    """Calculate cosine similarity between query and closest match in corpus"""
    similarities = cosine_similarity(query_vector, corpus_vectors)
    return np.max(similarities)

def get_top_k_predictions(model, vector, k=3):
    """Get top k predictions and their probabilities"""
    probas = model.predict_proba(vector)[0]
    top_k_indices = probas.argsort()[-k:][::-1]
    return [
        (model.classes_[i], probas[i])
        for i in top_k_indices
    ]

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    user_message_cleaned = clean_text(user_message)
    
    # Handle greetings first
    greetings = ['hi', 'hello', 'hey', 'how are you']
    if user_message_cleaned.lower() in greetings:
        return jsonify({
            'local_answer': "Hello! I'm your farming assistant. How can I help you today? 🌱",
            'local_confidence': "1.0",
            'confidence_level': 'very_high',
            'deepseek_answer': "Not needed - greeting"
        })

    # Check if exact match exists in dataset
    exact_match = faq_df[faq_df['question'].str.lower() == user_message_cleaned.lower()]
    if not exact_match.empty:
        return jsonify({
            'local_answer': exact_match.iloc[0]['answers'],
            'local_confidence': "1.0",
            'confidence_level': 'very_high',
            'deepseek_answer': "Not needed - exact match"
        })

    # Get predictions and metrics
    user_vector = vectorizer.transform([user_message_cleaned])
    pred_probas = model.predict_proba(user_vector)[0]
    local_pred_proba = pred_probas.max()
    similarity_score = get_similarity_score(user_vector, X)

    # Get confidence analysis
    confidence_level, use_local, confidence_score = analyze_confidence(
        local_pred_proba, similarity_score, pred_probas
    )

    try:
        # Always get API response for potential retraining
        response = make_api_call(user_message)
        deepseek_answer = ""
        if response and response.get('choices'):
            deepseek_answer = response['choices'][0]['message']['content']

        if use_local:
            local_answer = model.predict(user_vector)[0]
            # Retrain if API response is substantially different
            if deepseek_answer and len(deepseek_answer) > 50 and confidence_score < 0.9:
                retrain_model(user_message, deepseek_answer)
            return jsonify({
                'local_answer': local_answer,
                'local_confidence': f"{confidence_score:.2f}",
                'confidence_level': confidence_level,
                'deepseek_answer': deepseek_answer if confidence_score < 0.9 else "Not needed - high confidence local answer"
            })
        
        # For low confidence, always use API response and retrain
        if deepseek_answer:
            if retrain_model(user_message, deepseek_answer):
                print(f"Added new Q&A pair to training data: {user_message}")
            return jsonify({
                'local_answer': "Based on my latest information:",
                'local_confidence': f"{confidence_score:.2f}",
                'confidence_level': 'API_response',
                'deepseek_answer': deepseek_answer
            })
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        # Fallback to local model if API fails
        local_answer = model.predict(user_vector)[0]
        return jsonify({
            'local_answer': local_answer,
            'local_confidence': f"{confidence_score:.2f}",
            'confidence_level': 'fallback',
            'deepseek_answer': f"Error: {str(e)}"
        })

def make_api_call(message):
    """Separated API call logic with retry mechanism"""
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": message}]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=data, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    
    return None

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__': 
    app.run(debug=True)