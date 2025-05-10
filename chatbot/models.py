import os
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from chatbot.data_processing import vectorizer, tfidf_matrix, categories

# Initialize Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(tfidf_matrix, categories)

# Initialize KNN classifier
knn_classifier = NearestNeighbors(n_neighbors=5, metric='cosine')
knn_classifier.fit(tfidf_matrix.toarray())

# Save models
if not os.path.exists("models"):
    os.makedirs("models")
with open('models/nb_classifier.pkl', 'wb') as f:
    pickle.dump(nb_classifier, f)
with open('models/knn_classifier.pkl', 'wb') as f:
    pickle.dump(knn_classifier, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)