import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chatbot.data_processing import fasttext_model, fasttext_question_vectors, preprocess_text

def get_best_match_with_fasttext(query):
    """Find the best matching question using FastText embeddings."""
    if not query or not fasttext_model or len(fasttext_question_vectors) == 0:
        return -1, 0.0
        
    # Get query vector
    query_vector = get_document_vector_fasttext(query, fasttext_model)
    if query_vector is None:
        return -1, 0.0
        
    # Calculate similarities
    similarities = cosine_similarity([query_vector], fasttext_question_vectors)[0]
    
    # Get best match
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    return best_idx, best_similarity

def get_document_vector_fasttext(doc, model):
    """Generate document vector by averaging FastText word vectors."""
    if not doc or not model:
        return None
        
    # Preprocess and tokenize
    words = preprocess_text(doc).split()
    if not words:
        return None
        
    # Get word vectors
    word_vectors = []
    for word in words:
        try:
            if word in model.wv:
                word_vectors.append(model.wv[word])
        except:
            continue
            
    if not word_vectors:
        return None
        
    # Average word vectors
    return np.mean(word_vectors, axis=0) 