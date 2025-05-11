import os
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from chatbot.data_processing import initialize_models, tfidf_matrix, categories, vectorizer  # Import global variables

# Initialize models and data
initialize_models()

# Ensure tfidf_matrix and categories are initialized
if tfidf_matrix is None or not categories:
    raise ValueError("tfidf_matrix or categories are not initialized. Ensure data is loaded correctly.")

# Ensure vectorizer is initialized
if vectorizer is None:
    raise ValueError("vectorizer is not initialized. Ensure data is loaded correctly.")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, categories, test_size=0.2, random_state=42, stratify=categories
)

# Initialize and tune Naive Bayes classifier
nb_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
nb_classifier = GridSearchCV(
    MultinomialNB(), nb_param_grid, cv=5, scoring='f1_macro', n_jobs=-1
)
nb_classifier.fit(X_train, y_train)

# Initialize and tune KNN classifier
knn_param_grid = {'n_neighbors': [3, 5, 7, 9], 'metric': ['cosine', 'euclidean']}
knn_classifier = GridSearchCV(
    KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
knn_classifier.fit(X_train.toarray(), y_train)

# Save models and vectorizer
if not os.path.exists("models"):
    os.makedirs("models")
with open('models/nb_classifier.pkl', 'wb') as f:
    pickle.dump(nb_classifier.best_estimator_, f)
with open('models/knn_classifier.pkl', 'wb') as f:
    pickle.dump(knn_classifier.best_estimator_, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Load all available models
models = {
    "Naive Bayes": nb_classifier.best_estimator_,
    "KNN": knn_classifier.best_estimator_
}

def get_model_metrics():
    """Compute and return metrics for the models."""
    if X_test.shape[0] == 0 or len(y_test) == 0:
        raise ValueError("Test data is empty. Ensure proper data splitting.")

    # Generate predictions
    y_pred_nb = nb_classifier.predict(X_test)

    # Use majority voting for KNN predictions
    y_pred_knn = []
    for neighbors in knn_classifier.best_estimator_.kneighbors(X_test.toarray(), return_distance=False):
        neighbor_categories = [y_train[neighbor] for neighbor in neighbors]
        most_common_category = Counter(neighbor_categories).most_common(1)[0][0]
        y_pred_knn.append(most_common_category)

    # Compute metrics
    metrics = {
        "Naive Bayes": {
            "f1_score": f1_score(y_test, y_pred_nb, average='macro'),
            "precision": precision_score(y_test, y_pred_nb, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred_nb, average='macro', zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred_nb),
        },
        "KNN": {
            "f1_score": f1_score(y_test, y_pred_knn, average='macro'),
            "precision": precision_score(y_test, y_pred_knn, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred_knn, average='macro', zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred_knn),
        },
    }

    # Print detailed metrics for debugging
    print("Naive Bayes Best Params:", nb_classifier.best_params_)
    print("KNN Best Params:", knn_classifier.best_params_)
    print("Naive Bayes Metrics:", metrics["Naive Bayes"])
    print("KNN Metrics:", metrics["KNN"])

    return metrics