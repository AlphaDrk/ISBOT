import json
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
import os
from gensim.models import FastText
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stop words
stemmer_fr = SnowballStemmer('french')
stop_words_fr = set(stopwords.words('french'))

# Whoosh schema
schema = Schema(question=TEXT(stored=True), answer=TEXT(stored=True), url=TEXT(stored=True))

# Initialize Whoosh index
if not os.path.exists("indexdir"):
    os.makedirs("indexdir")
ix = create_in("indexdir", schema)

# Global variables
questions = []
responses = []
urls = []
file_paths = []
categories = []
vectorizer = None
tfidf_matrix = None
fasttext_model = None
fasttext_question_vectors = []

def get_document_vector_fasttext(doc, model):
    """Generate document vector by averaging FastText word vectors."""
    if not doc or not model:
        return np.zeros(200)  # Default vector size
    words = preprocess_text(doc).split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def preprocess_text(text):
    """Preprocess text for analysis."""
    if not text:
        return ""
        
    try:
        lang = detect(text)
    except:
        lang = 'fr'
    
    stemmer = stemmer_fr if lang == 'fr' else SnowballStemmer('english')
    stop_words = stop_words_fr if lang == 'fr' else set(stopwords.words('english'))
    
    # Normalize text
    text = text.lower().strip()
    
    # Replace special characters and numbers
    text = re.sub(r'[0-9]+', ' ', text)  # Replace numbers with space
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace special chars with space
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize and stem
    tokens = []
    for word in word_tokenize(text):
        # Skip if word is too short or just numbers
        if len(word) < 2 or word.isdigit():
            continue
        # Skip stop words
        if word in stop_words:
            continue
        # Stem the word
        stemmed = stemmer.stem(word)
        if stemmed:
            tokens.append(stemmed)
            
    return ' '.join(tokens)

def process_data_entries(entries, source_file):
    """Process entries from a JSON file and add them to global lists."""
    global questions, responses, urls, file_paths, categories
    for entry in entries:
        # Get answer from either 'answer' or 'response' field
        answer = entry.get('answer', '') or entry.get('response', '')
        if not answer or all(c in '+\n ' for c in answer):
            continue
            
        main_question = entry['question']
        variations = entry.get('question_variations', [])
        
        # Add main question and variations
        questions.extend([main_question] + variations)
        responses.extend([answer] * (len(variations) + 1))
        urls.extend([entry.get('url', '')] * (len(variations) + 1))
        file_paths.extend([entry.get('file_path', '')] * (len(variations) + 1))
        categories.extend([entry.get('category', 'general')] * (len(variations) + 1))

def load_data():
    """Load data from all JSON files and index them."""
    global questions, responses, urls, file_paths, categories
    questions = []
    responses = []
    urls = []
    file_paths = []
    categories = []
    
    # Load from data.json
    try:
        with open('data/data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            process_data_entries(data, 'data.json')
    except Exception as e:
        print(f"Error loading data.json: {e}")

    # Load from new_questions.json
    try:
        if os.path.exists('data/new_questions.json'):
            with open('data/new_questions.json', 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        entry = json.loads(line)
                        process_data_entries([entry], 'new_questions.json')
    except Exception as e:
        print(f"Error loading new_questions.json: {e}")

    # Load from raw_data.json
    try:
        if os.path.exists('data/raw_data.json'):
            with open('data/raw_data.json', 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
                process_data_entries(raw_data, 'raw_data.json')
    except Exception as e:
        print(f"Error loading raw_data.json: {e}")

    # Update Whoosh index
    writer = ix.writer()
    for i in range(len(questions)):
        answer = responses[i].strip()
        if not answer or all(c in '+\n ' for c in answer):
            continue
        writer.add_document(
            question=questions[i],
            answer=answer,
            url=urls[i] if i < len(urls) else ''
        )
    writer.commit()

def initialize_models():
    """Initialize and train all models."""
    global vectorizer, tfidf_matrix, fasttext_model, fasttext_question_vectors
    
    # Initialize TF-IDF vectorizer with updated parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        analyzer='word',
        tokenizer=word_tokenize,
        preprocessor=preprocess_text,
        token_pattern=None  # Disable token_pattern since we're using custom tokenizer
    )
    
    # Process questions and create TF-IDF matrix
    processed_questions = [preprocess_text(q) for q in questions]
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    
    # Train or load FastText model with updated parameters
    fasttext_model_path = 'models/fasttext.model'
    if os.path.exists(fasttext_model_path):
        fasttext_model = FastText.load(fasttext_model_path)
    else:
        tokenized_questions = [preprocess_text(q).split() for q in questions]
        fasttext_model = FastText(
            tokenized_questions,
            vector_size=200,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=50,
            min_n=2,  # Minimum length of char n-grams
            max_n=5   # Maximum length of char n-grams
        )
        if not os.path.exists('models'):
            os.makedirs('models')
        fasttext_model.save(fasttext_model_path)
    
    # Pre-calculate FastText vectors
    fasttext_question_vectors = np.array([get_document_vector_fasttext(q, fasttext_model) for q in questions])

# Initialize everything
load_data()
initialize_models()