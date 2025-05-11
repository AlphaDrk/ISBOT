# Chatbot ISET

![Chatbot ISET](https://img.shields.io/badge/Version-1.1-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)

This project is an **educational chatbot** developed as part of a Machine Learning mini-project for the DSIR course at ISET. It answers user queries about academic services (schedules, registrations, exams, etc.) using natural language processing (NLP) techniques and machine learning algorithms. The chatbot is integrated into a web interface with Flask and offers advanced features like search, classification, and self-learning.

## Features

- **Natural Language Processing**: Uses NLTK for preprocessing questions (tokenization, stop word removal, stemming).
- **Intent Classification**: Two trained models: Naive Bayes and KNN, with TF-IDF vectorization.
- **Text Search**: Whoosh-based search engine to find answers in a database.
- **Multilingual Support**: Automatic language detection (French/English) with tailored preprocessing.
- **Predefined Shortcuts**: Commands like `/horaires`, `/contact`, `/inscription` for quick responses.
- **Contextual Suggestions**: Provides links or additional information based on the query.
- **Self-Learning**: Saves new questions to improve the database.
- **Response Evaluation**: Feedback system (👍/👎) to measure user satisfaction.
- **Interactive Dashboard**: Visualizes model performance (accuracy, F1-score) and dataset insights.
- **Real-Time Predictions**: Allows users to test predictions directly in the dashboard.

## Prerequisites

Before running the project, ensure you have the following installed:
- **Python 3.8+**
- **Pip** (Python package manager)
- A virtual environment (recommended)

### Dependencies
The required libraries are listed in `requirements.txt`. Key dependencies include:
- `flask`: Web framework
- `dash`: Dashboard framework
- `nltk`: Natural language processing
- `sklearn`: Machine learning (Naive Bayes, KNN, TF-IDF)
- `whoosh`: Search engine
- `langdetect`: Language detection
- `plotly`: Data visualization
- `fasttext`: Word embeddings

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/chatbot-iset.git
   cd chatbot-iset
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources:**
   Run the following script in a Python terminal:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. **Set up data:**
   Ensure the file `data/data.json` contains initial questions, answers, and URLs in the following format:
   ```json
   [
       {"question": "What are the schedules?", "answer": "Here are the schedules...", "url": "/programmes/horaires", "category": "schedules", "question_variations": ["Class hours?", "Schedules please"]}
   ]
   ```

6. **Run the application:**
   ```bash
   python app.py
   ```
   Access the app at `http://localhost:5000` in your browser.

## Usage

### Web Interface
- **Chat**: Ask a question or use a shortcut (e.g., `/help`) in the text box.
- **Dashboard**: View model performance (accuracy, F1-score) and dataset insights.
- **About**: Learn more about the project and its features.

### Shortcuts
- `/horaires`: Class schedules
- `/contact`: Administration contact details
- `/inscription`: Registration procedure
- `/bibliotheque`: Library hours
- `/examens`: Exam calendar
- `/help`: List of commands

### Example
**User Input:** "When are the exams?"  
**Chatbot Response:** "The exam calendar is available via the link below."  
**Link:** `https://isetsf.rnu.tn/programmes/calendrier-examens`

## Project Structure

```
chatbot/
├── __init__.py             # Package initializer
├── chatbot_logic.py        # Core chatbot logic
├── config.py               # Configuration settings
├── data_processing.py      # Data preprocessing and model training
├── database.py             # SQLite database operations
├── embeddings_utils.py     # Embedding utilities
├── models.py               # Machine learning models
├── scraper.py              # Data scraping utilities
├── __pycache__/            # Compiled Python files
│   ├── __init__.cpython-312.pyc
│   ├── chatbot_logic.cpython-312.pyc
│   ├── config.cpython-312.pyc
│   ├── data_processing.cpython-312.pyc
│   ├── database.cpython-312.pyc
│   ├── embeddings_utils.cpython-312.pyc
│   ├── models.cpython-312.pyc

static/
├── about.css               # About page styles
├── assistant-avatar.png    # Assistant avatar image
├── login_bratuha.png       # Login page image
├── logoX.png               # Logo image
├── style.css               # CSS styles
├── files/                  # Static files
│   ├── documents/          # Document files
│   ├── images/             # Image files
│   ├── pdfs/               # PDF files

models/
├── fasttext.model          # Trained FastText model
├── fasttext.model.wv.vectors_ngrams.npy # FastText n-gram vectors
├── knn_classifier.pkl      # Trained KNN model
├── nb_classifier.pkl       # Trained Naive Bayes model
├── vectorizer.pkl          # TF-IDF vectorizer

data/
├── data.json               # Initial question-answer data
├── new_questions.json      # User-submitted questions
├── raw_data.json           # Raw data for processing

indexdir/
├── _MAIN_1.toc             # Whoosh index table of contents
├── MAIN_79qscey1tyj7mc88.seg # Whoosh index segment
├── MAIN_WRITELOCK          # Whoosh index write lock

templates/
├── about.html              # About page
├── base.html               # Base HTML template
├── chat.html               # Chat interface
├── dashboard.html          # Dashboard interface
├── login.html              # Login page
├── shared_chat.html        # Shared chat template

instance/
├── chatbot.db              # SQLite database

env/                        # Virtual environment
├── Scripts/                # Scripts for managing the environment
├── Lib/                    # Installed libraries
├── pyvenv.cfg              # Virtual environment configuration

app.py                      # Flask application and routes
dashboard_setup.py          # Dashboard initialization
README.md                   # Project documentation
requirements.txt            # Dependencies
wsgi.py                     # WSGI entry point
```

## Model Performance

- **Naive Bayes**:
  - Accuracy: ~85%
  - F1-score: ~83%
- **KNN**:
  - Accuracy: ~80%
  - F1-score: ~78%

Performance metrics are calculated using cross-validation and displayed in the dashboard.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributors

- **Your Name** - Initial work
- **Collaborators** - Contributions and feedback

## Acknowledgments

- **ISET** - For providing the project framework.
- **Open-source libraries** - For enabling rapid development.

---

Feel free to contribute to this project by submitting issues or pull requests on GitHub.
