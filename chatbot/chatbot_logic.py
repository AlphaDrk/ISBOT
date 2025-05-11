import os
import requests
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from whoosh.qparser import QueryParser
from chatbot.data_processing import (
    ix, responses, urls, preprocess_text, vectorizer, 
    tfidf_matrix, file_paths, fasttext_question_vectors
)
from chatbot.models import nb_classifier, knn_classifier
from chatbot.config import shortcuts, shortcut_urls
from chatbot.embeddings_utils import get_best_match_with_fasttext
from dotenv import load_dotenv
import re

# Load environment variables for API key
load_dotenv()
API_KEY = os.getenv('OPENROUTER_API_KEY')
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Add a mapping for method names
METHOD_NAME_MAP = {
    'exact_match': 'correspondance exacte 🧠',
    'tfidf 🧠': 'Recherche TF-IDF 🧠',
    'fasttext 🧠': 'FastText 🧠',
    'knn 🧠': 'Recherche KNN 🧠',
    'index_search 🧠': 'Recherche Index 🧠',
    'External Chatbot 👾': 'Chatbot Externe 👾'
}

def map_method_name(method):
    return METHOD_NAME_MAP.get(method, method)

def call_openrouter_api(query):
    """Call OpenRouter API to generate a response."""
    try:
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:8080',
            'X-Title': 'ISBOT'
        }
        payload = {
            'model': 'meta-llama/llama-3.1-8b-instruct:free',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': query}
            ]
        }
        response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        answer = data['choices'][0]['message']['content'].strip()
        return answer
    except requests.RequestException as e:
        print(f"OpenRouter API request failed: {e}")
        return "Sorry, I could not connect to the OpenRouter API."

def search_in_index(query):
    """Search the Whoosh index for a matching question."""
    with ix.searcher() as searcher:
        query_obj = QueryParser("question", ix.schema).parse(query)
        results = searcher.search(query_obj, limit=1)
        if results:
            return {
                "answer": results[0]['answer'],
                "url": results[0]['url'],
                "similarity": results[0].score
            }
        return None

def get_shortcut_url(shortcut):
    """Get the URL for a shortcut command."""
    path = shortcut_urls.get(shortcut)
    return f"https://isetsf.rnu.tn{path}" if path else None

def check_new_questions(user_input, user_id):
    """Check if the question exists in new_questions.json for the user."""
    try:
        if not os.path.exists('data/new_questions.json'):
            return None
        
        with open('data/new_questions.json', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry['question'].lower() == user_input.lower() and entry['user_id'] == user_id:
                        return entry['response']
        return None
    except Exception as e:
        print(f"Error checking new_questions.json: {e}")
        return None

def format_url(url):
    """Format URL to ensure it's absolute."""
    if not url:
        return None
    if url.startswith("http"):
        return url
    return f"https://isetsf.rnu.tn{url}"

def is_placeholder_or_stupid_answer(answer):
    """Check if an answer is a placeholder or not useful."""
    if not answer or len(answer.strip()) < 10:
        return True
    # Detect if answer is just a list of names/titles
    if re.match(r'^(\w+\s+)+$', answer.strip()):
        return True
    # Detect if answer is just a list (no punctuation, no sentences)
    if len(answer.split()) < 8 and answer.count('.') == 0:
        return True
    # Detect repetitive content
    if len(set(answer.split())) < 3:
        return True
    return False

def is_relevant_answer(question, answer):
    """Check if an answer is relevant to the question."""
    # Allow generic positive/neutral answers for short/greeting queries
    greetings = {"good", "thanks", "thank you", "ok", "okay", "fine", "great", "bonjour", "salut", "hello", "hi", "merci"}
    if question.strip().lower() in greetings:
        return True
    
    # Simple keyword overlap check
    question_words = set(preprocess_text(question).split())
    answer_words = set(preprocess_text(answer).split())
    overlap = question_words & answer_words
    
    if not overlap:
        return False
    
    # Check for common irrelevant patterns
    if re.match(r'^(rang|nom|prénom|score|groupe)', answer.lower()):
        return False
        
    return True

def get_response(user_input, user_id):
    """Process user input and return the best matching response."""
    # Check shortcuts first
    if user_input.lower() in ['hello', 'hi', 'hey'] or user_input in shortcuts:
        return {
            "answer": shortcuts.get(user_input, "Hello! How can I assist you today?"),
            "url": get_shortcut_url(user_input),
            "similarity": 1.0,
            "category": "shortcut",
            "is_shortcut": True,
            "method": map_method_name("exact_match"),
            "source": "local"
        }

    # Check saved responses
    saved_response = check_new_questions(user_input, user_id)
    if saved_response:
        return {
            "answer": saved_response,
            "url": None,
            "similarity": 1.0,
            "category": "saved_data",
            "is_shortcut": False,
            "method": map_method_name("exact_match"),
            "source": "local"
        }

    # Process input
    processed_input = preprocess_text(user_input)
    input_tfidf = vectorizer.transform([processed_input])
    input_dense = input_tfidf.toarray()

    # Try TF-IDF matching
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    best_match_idx = similarities.argmax()
    max_similarity = similarities[0, best_match_idx]

    # Format similarity as a percentage
    max_similarity_percentage = round(max_similarity * 100, 2)

    if max_similarity > 0.65:
        answer = responses[best_match_idx]
        if not is_placeholder_or_stupid_answer(answer) and is_relevant_answer(user_input, answer):
            return {
                "answer": answer,
                "url": format_url(urls[best_match_idx]) if urls[best_match_idx] else None,
                "file_path": file_paths[best_match_idx] if file_paths[best_match_idx] else None,
                "similarity": max_similarity_percentage,  # Use formatted percentage
                "category": nb_classifier.predict(input_tfidf)[0],
                "is_shortcut": False,
                "method": map_method_name("tfidf 🧠"),
                "source": "local"
            }

    # Try FastText matching
    ft_idx, ft_sim = get_best_match_with_fasttext(user_input)
    if ft_sim > 0.8:
        answer = responses[ft_idx]
        if not is_placeholder_or_stupid_answer(answer) and is_relevant_answer(user_input, answer):
            return {
                "answer": answer,
                "url": format_url(urls[ft_idx]) if urls[ft_idx] else None,
                "file_path": file_paths[ft_idx] if file_paths[ft_idx] else None,
                "similarity": float(ft_sim),
                "category": nb_classifier.predict(input_tfidf)[0],
                "is_shortcut": False,
                "method": map_method_name("fasttext 🧠"),
                "source": "local"
             }

    # Try KNN matching
    distances, indices = knn_classifier.best_estimator_.kneighbors(input_dense, n_neighbors=1)  # Use best estimator
    if distances[0][0] < 0.7:
        idx = indices[0][0]
        answer = responses[idx]
        if not is_placeholder_or_stupid_answer(answer) and is_relevant_answer(user_input, answer):
            return {
                "answer": answer,
                "url": format_url(urls[idx]) if urls[idx] else None,
                "file_path": file_paths[idx] if file_paths[idx] else None,
                "similarity": float(1.0 - distances[0][0]),
                "category": nb_classifier.predict(input_tfidf)[0],
                "is_shortcut": False,
                "method": map_method_name("knn 🧠"),
                "source": "local"
            }

    # Try Whoosh index search
    search_result = search_in_index(user_input)
    if search_result and not is_placeholder_or_stupid_answer(search_result['answer']) and is_relevant_answer(user_input, search_result['answer']):
        return {
            "answer": search_result['answer'],
            "url": format_url(search_result['url']) if search_result['url'] else None,
            "similarity": float(search_result['similarity']),
            "category": nb_classifier.predict(input_tfidf)[0],
            "is_shortcut": False,
            "method": map_method_name("index_search 🧠"),
            "source": "local"
        }

    # Fallback to OpenRouter API
    api_response = call_openrouter_api(user_input)
    response_dict = {
        "answer": api_response,
        "url": None,
        "similarity": 0.0,
        "category": "external_api",
        "is_shortcut": False,
        "method": map_method_name("External Chatbot 👾"),
        "source": "local"
    }
    save_new_question(user_input, response_dict, user_id=user_id)
    return response_dict

def save_new_question(user_input, response, rating=None, user_id=None):
    """Save new questions and responses to a file."""
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Check if question already exists
        if os.path.exists('data/new_questions.json'):
            with open('data/new_questions.json', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry['question'].lower() == user_input.lower() and entry['user_id'] == user_id:
                            return True
        
        # Prepare new entry
        answer = response.get('answer') if isinstance(response, dict) else response
        entry = {
            "question": user_input,
            "response": answer,
            "rating": rating,
            "user_id": user_id,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save to file
        with open('data/new_questions.json', 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
        return True
    except Exception as e:
        print(f"Error saving question: {e}")
        return False