from dotenv import load_dotenv
import os

load_dotenv()

shortcuts = {
    "🕒 Horaires": "Consultez le calendrier des devoirs pour le 2ème semestre 2024-2025 (SEG, TI, GP, GC, GM) sur le site officiel.",
    "📞 Contact": "Contactez l'administration: Email: bassem.jallouli@sfax.r-iset.tn (Directeur), Tél: +216 74 431 495 (Sciences Économiques et Gestion).",
    "📝 Inscription": "La pré-inscription en ligne est disponible via www.inscription.tn. Consultez la page d'inscription pour plus de détails.",
    "📚 Bibliothèque": "La bibliothèque de l'ISET Sfax dispose de 3550 ouvrages. Horaires disponibles auprès de l'administration.",
    "📖 Examens": "Le planning des devoirs pour le 2ème semestre 2024-2025 est disponible. Téléchargez la convocation via le lien.",
}

shortcut_urls = {
    "🕒 Horaires": "/fr/article/704/calendrier-des-devoirs-2s-20242025",
    "📞 Contact": "/fr/institut/message-du-directeur",
    "📝 Inscription": "/fr/inscription-en-ligne",
    "📚 Bibliothèque": "/fr/institut/presentation",
    "📖 Examens": "/fr/article/707/planning-des-devoirs-2s-2425",
}

MODEL_PATHS = {
    "fasttext": "models/fasttext.model",
    "nb_classifier": "models/nb_classifier.pkl",
    "knn_classifier": "models/knn_classifier.pkl",
    "vectorizer": "models/vectorizer.pkl"
}

SIMILARITY_THRESHOLDS = {
    "tfidf": 0.65,
    "fasttext": 0.8,
    "knn": 0.7
}

DATA_PATH = "data/data.json"
NEW_QUESTIONS_PATH = "data/new_questions.json"
INDEX_DIR = "indexdir"

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')