import os
import re
import json
import logging
import requests
import mimetypes
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor
import time

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ISETScraper:
    def __init__(self, base_url="https://isetsf.rnu.tn/fr"):
        self.base_url = base_url
        self.media_dir = os.path.join("static", "files")
        self.data_dir = os.path.join("data")
        self.create_directories()
        self.visited_urls = set()
        self.data = []
        self.current_id = 1
        self.stop_words = set(stopwords.words('french'))
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )

    def clean_question_text(self, text, max_words=10):
        """Nettoie et limite la longueur du texte de la question"""
        # Supprime les caractères spéciaux et les espaces multiples
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limite le nombre de mots
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
        
        return text

    def generate_questions(self, title, content, url):
        """Génère des questions basées sur l'URL avec des variations contextuelles."""
        # Extraire et nettoyer le dernier segment de l'URL
        path = urlparse(url).path.strip('/')
        segments = path.split('/')
        last_segment = next((s for s in reversed(segments) if s), '')
        clean_title = self.clean_url_segment(last_segment)
        
        # Déterminer la catégorie
        category = self.determine_category(url, title, content)
        
        # Générer les questions de base
        main_question = f"Qu'est-ce que {clean_title} ?"
        base_variations = self.generate_question_variations(clean_title)
        
        # Ajouter les questions spécifiques à la catégorie
        category_variations = self.get_category_specific_questions(clean_title, category)
        
        # Combiner toutes les variations et supprimer les doublons
        all_variations = list(dict.fromkeys(base_variations + category_variations))
        
        # Limiter le nombre de variations pour éviter la redondance
        question_variations = all_variations[:7]  # Augmenté à 7 variations
        
        return [main_question], question_variations

    def create_directories(self):
        """Crée les répertoires nécessaires"""
        # Création des dossiers pour les fichiers
        for subdir in ['pdfs', 'images', 'documents']:
            os.makedirs(os.path.join("static", "files", subdir), exist_ok=True)
        
        # Création du dossier data
        os.makedirs(self.data_dir, exist_ok=True)

    def is_valid_url(self, url):
        """Vérifie si l'URL est valide et appartient au domaine ISET"""
        try:
            parsed = urlparse(url)
            return parsed.netloc == "isetsf.rnu.tn" and parsed.path.startswith("/fr")
        except:
            return False

    def clean_text(self, text):
        
        """Nettoie le texte en supprimant les espaces superflus et les caractères spéciaux"""
        if not text:
            return ""
        # Supprime les espaces multiples et les retours à la ligne superflus
        text = re.sub(r'\s+', ' ', text)
        # Garde les retours à la ligne significatifs
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def extract_main_content(self, soup):
        """Extrait le contenu significatif de la page"""
        # Supprime les éléments communs
        for element in soup.find_all(['header', 'footer', 'nav', 'script', 'style']):
            element.decompose()
        
        # 1. Essayer d'abord containerPage
        main_content = soup.find('div', class_='containerPage')
        if not main_content:
            logging.warning("ContainerPage non trouvé, tentative avec container")
            main_content = soup.find('div', class_='container')
        
        content = ""
        if main_content:
            articles = main_content.find_all('article')
            if articles:
                content = []
                for article in articles:
                    text = self.clean_text(article.get_text())
                    if text:
                        content.append(text)
                if content:
                    content = '\n'.join(content)
            else:
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    content = []
                    for p in paragraphs:
                        text = self.clean_text(p.get_text())
                        if text:
                            content.append(text)
                    if content:
                        content = '\n'.join(content)
                else:
                    divs = main_content.find_all('div')
                    for div in divs:
                        text = self.clean_text(div.get_text())
                        if text and len(text.split()) > 10:
                            content = text
                            break
        if not content:
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headers = soup.find_all(tag)
                for header in headers:
                    text = self.clean_text(header.get_text())
                    if text and len(text.split()) > 3:
                        content = text
                        break
                if content:
                    break
        if not content:
            for tag in ['article', 'section', 'main']:
                elements = soup.find_all(tag)
                for element in elements:
                    text = self.clean_text(element.get_text())
                    if text and len(text.split()) > 10:
                        content = text
                        break
                if content:
                    break
        if not content:
            for element in soup.find_all(['div', 'span', 'p']):
                text = self.clean_text(element.get_text())
                if text and len(text.split()) > 5:
                    content = text
                    break
        content = content or ""
        # If content is only + or empty, treat as no content
        if not content or all(c in '+\n ' for c in content):
            return ""

        # In extract_main_content
        licences_section = soup.find('div', class_='licences-list')  # Example class
        if licences_section:
            return self.clean_text(licences_section.get_text())

        table = soup.find('table')
        if table:
            return self.clean_text(table.get_text())

        return content

    def download_media(self, url, media_type):
        """Télécharge un fichier média et retourne son chemin local"""
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                # Détermine l'extension du fichier
                content_type = response.headers.get('content-type', '')
                ext = mimetypes.guess_extension(content_type) or '.bin'
                
                # Génère un nom de fichier unique avec timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{hash(url)}{ext}"
                filepath = os.path.join(self.media_dir, media_type, filename)
                
                # Sauvegarde le fichier
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return filepath
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement de {url}: {str(e)}")
        return None

    def extract_media_links(self, soup, page_url):
        """Extrait les liens vers les médias de la page"""
        media_links = {
            'pdfs': [],
            'images': [],
            'documents': []
        }
        
        # PDFs
        for link in soup.find_all('a', href=re.compile(r'\.pdf$', re.I)):
            url = urljoin(page_url, link['href'])
            local_path = self.download_media(url, 'pdfs')
            if local_path:
                media_links['pdfs'].append({
                    'url': url,
                    'local_path': local_path,
                    'title': link.get_text(strip=True) or 'Document PDF'
                })
        
        # Images
        for img in soup.find_all('img'):
            if img.get('src'):
                url = urljoin(page_url, img['src'])
                local_path = self.download_media(url, 'images')
                if local_path:
                    media_links['images'].append({
                        'url': url,
                        'local_path': local_path,
                        'alt': img.get('alt', '')
                    })
        
        # Autres documents
        doc_extensions = r'\.(doc|docx|xls|xlsx|ppt|pptx|txt|zip|rar)$'
        for link in soup.find_all('a', href=re.compile(doc_extensions, re.I)):
            url = urljoin(page_url, link['href'])
            local_path = self.download_media(url, 'documents')
            if local_path:
                media_links['documents'].append({
                    'url': url,
                    'local_path': local_path,
                    'title': link.get_text(strip=True) or 'Document'
                })
        
        return media_links

    def determine_category(self, url, title, content):
        """Détermine la catégorie de la page en fonction de son contenu et URL"""
        url_lower = url.lower()
        content_lower = content.lower()
        
        categories = {
            'actualites': ['actualites', 'news', 'annonces'],
            'formation': ['formation', 'cours', 'programme', 'etudes'],
            'administration': ['administration', 'direction', 'services'],
            'admission': ['admission', 'inscription', 'concours'],
            'vie_etudiante': ['vie-etudiante', 'etudiants', 'clubs'],
            'recherche': ['recherche', 'laboratoire', 'projets']
        }
        
        for category, keywords in categories.items():
            if any(keyword in url_lower or keyword in content_lower for keyword in keywords):
                return category
        
        return 'autre'

    def remove_duplicates(self, text):
        seen = set()
        result = []
        for line in text.split('\n'):
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                result.append(line)
        return '\n'.join(result)

    def remove_useless_sentences(self, text, keywords=None):
        if keywords is None:
            keywords = [
                "Téléchargez la brochure", "Plan d'étude", "plan d'étude",
                "kamel.jallouli@", "Contactez-nous", "Cliquez ici"
            ]
        result = []
        for line in text.split('\n'):
            if not any(kw.lower() in line.lower() for kw in keywords):
                result.append(line)
        return '\n'.join(result)

    def clean_list_duplicates(self, text):
        lines = text.split('\n')
        seen = set()
        result = []
        for line in lines:
            if line.strip().startswith('-'):
                if line not in seen:
                    seen.add(line)
                    result.append(line)
            else:
                result.append(line)
        return '\n'.join(result)

    def limit_text(self, text, max_chars=2000, max_paragraphs=20):
        paragraphs = text.split('\n')
        limited = []
        total_chars = 0
        for p in paragraphs:
            if total_chars + len(p) > max_chars or len(limited) >= max_paragraphs:
                break
            limited.append(p)
            total_chars += len(p)
        return '\n'.join(limited)

    def scrape_page(self, url, max_retries=3):
        """Scrape une page individuelle et structure les données"""
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        logging.info(f"Scraping: {url}")

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Vérifier si la réponse est en HTML
                if 'text/html' not in response.headers.get('content-type', '').lower():
                    logging.warning(f"La réponse n'est pas en HTML pour {url}")
                    return []

                soup = BeautifulSoup(response.text, 'html.parser')

                # Extraction du titre
                title = soup.find('title').get_text(strip=True) if soup.find('title') else ''

                # Extraction du contenu principal
                text_content = self.extract_main_content(soup)
                if not text_content:
                    logging.warning(f"Aucun contenu significatif trouvé pour l'URL: {url}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Attendre avant de réessayer
                        continue
                    return []

                # Extraction des médias
                media_links = self.extract_media_links(soup, url)

                # Génération des questions
                main_question, question_variations = self.generate_questions(title, text_content, url)

                # Structure des données
                page_data = {
                    'id': self.current_id,
                    'category': self.determine_category(url, title, text_content),
                    'question': main_question[0],
                    'question_variations': question_variations,
                    'answer': self.limit_text(text_content),
                    'url': url,
                    'title': title,
                    'media': media_links
                }

                self.data.append(page_data)
                self.current_id += 1

                # Pause pour éviter de surcharger le serveur
                time.sleep(1)

                # Extraction des liens pour continuer le scraping
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    if self.is_valid_url(full_url):
                        links.append(full_url)

                return links

            except requests.exceptions.RequestException as e:
                logging.error(f"Erreur de requête pour {url} (tentative {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Attendre avant de réessayer
                    continue
            except Exception as e:
                logging.error(f"Erreur lors du scraping de {url}: {str(e)}")
                return []

        return []

    def save_data(self):
        """Sauvegarde les données dans les fichiers JSON"""
        # Création des données au format souhaité
        formatted_data = []
        for entry in self.data:
            # Détermine le file_path si c'est un PDF
            file_path = None
            if entry['media']['pdfs']:
                # Convertir le chemin en chemin relatif pour static/files
                full_path = entry['media']['pdfs'][0]['local_path']
                file_path = os.path.relpath(full_path, "static/files")
            
            formatted_entry = {
                'id': entry['id'],
                'category': entry['category'],
                'question': entry['question'],
                'question_variations': entry['question_variations'],
                'answer': entry['answer'],
                'url': entry['url']
            }
            
            # Ajoute file_path seulement si c'est un PDF
            if file_path:
                formatted_entry['file_path'] = file_path
            
            formatted_data.append(formatted_entry)
        
        # Sauvegarde des données au format souhaité
        with open(os.path.join(self.data_dir, 'data.json'), 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        # Sauvegarde des données brutes pour référence
        with open(os.path.join(self.data_dir, 'raw_data.json'), 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def start_scraping(self):
        """Démarre le processus de scraping"""
        urls_to_visit = [self.base_url]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            while urls_to_visit:
                current_urls = urls_to_visit[:10]  # Traite 10 URLs à la fois
                urls_to_visit = urls_to_visit[10:]
                
                # Scrape les URLs actuelles
                future_to_url = {executor.submit(self.scrape_page, url): url for url in current_urls}
                
                # Collecte les nouveaux liens
                for future in future_to_url:
                    try:
                        new_links = future.result()
                        if new_links:
                            urls_to_visit.extend([link for link in new_links if link not in self.visited_urls])
                    except Exception as e:
                        logging.error(f"Erreur lors du traitement d'une URL: {str(e)}")

        # Sauvegarde des données
        self.save_data()

    def search(self, query):
        """Recherche dans les documents indexés"""
        results = []
        with open(os.path.join(self.data_dir, 'data.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                if query.lower() in entry['answer'].lower():
                    results.append({
                        'title': entry['title'],
                        'url': entry['url'],
                        'score': self.calculate_score(entry['answer'], query)
                    })
        return sorted(results, key=lambda r: r['score'], reverse=True)

    def calculate_score(self, text, query):
        """Calcule le score de pertinence d'un document par rapport à une requête"""
        # Implémentation de la mise en œuvre du score
        return 0.0  # Placeholder, actual implementation needed

    def clean_url_segment(self, segment):
        """Nettoie et formate un segment d'URL de manière plus intelligente."""
        # Remplacer les tirets et underscores par des espaces
        text = segment.replace('-', ' ').replace('_', ' ')
        
        # Supprimer les numéros au début (comme "638" dans l'exemple)
        text = re.sub(r'^\d+\s*', '', text)
        
        # Supprimer les extensions de fichier
        text = re.sub(r'\.(html|php|asp|aspx)$', '', text)
        
        # Mettre en majuscule la première lettre de chaque mot
        text = text.title()
        
        # Supprimer les mots vides (articles, prépositions)
        stop_words = {'de', 'du', 'des', 'le', 'la', 'les', 'et', 'en', 'au', 'aux', 'pour', 'par', 'avec', 'sans'}
        words = [word for word in text.split() if word.lower() not in stop_words]
        
        return ' '.join(words)

    def generate_question_variations(self, clean_title):
        """Génère des variations de questions plus naturelles."""
        variations = [
            f"Qu'est-ce que {clean_title} ?",
            f"Pouvez-vous m'expliquer ce qu'est {clean_title} ?",
            f"Je souhaite en savoir plus sur {clean_title}",
            f"Pourriez-vous me donner des informations sur {clean_title} ?",
            f"Quelles sont les caractéristiques de {clean_title} ?",
            f"Comment fonctionne {clean_title} ?",
            f"Quel est le rôle de {clean_title} ?"
        ]
        
        # Ajouter des variations spécifiques selon la longueur du titre
        if len(clean_title.split()) <= 3:
            variations.extend([
                f"Où se trouve {clean_title} ?",
                f"Quand est-ce que {clean_title} est disponible ?",
                f"Qui peut accéder à {clean_title} ?"
            ])
        
        return variations

    def get_category_specific_questions(self, clean_title, category):
        """Génère des questions spécifiques selon la catégorie."""
        category_questions = {
            'formation': [
                f"Quelles sont les formations disponibles en {clean_title} ?",
                f"Comment s'inscrire à la formation {clean_title} ?",
                f"Quel est le programme de la formation {clean_title} ?",
                f"Quelles sont les conditions d'admission pour {clean_title} ?",
                f"Quelle est la durée de la formation {clean_title} ?"
            ],
            'actualites': [
                f"Quelles sont les dernières actualités concernant {clean_title} ?",
                f"Quand a eu lieu l'événement {clean_title} ?",
                f"Où se déroule {clean_title} ?",
                f"Qui peut participer à {clean_title} ?",
                f"Comment s'inscrire à {clean_title} ?"
            ],
            'administration': [
                f"Qui est responsable de {clean_title} ?",
                f"Comment contacter le service {clean_title} ?",
                f"Quelles sont les procédures administratives pour {clean_title} ?",
                f"Quels sont les horaires d'ouverture de {clean_title} ?",
                f"Où se trouve le bureau de {clean_title} ?"
            ],
            'vie_etudiante': [
                f"Quelles sont les activités proposées par {clean_title} ?",
                f"Comment rejoindre {clean_title} ?",
                f"Quels sont les avantages de {clean_title} ?",
                f"Quand se réunit {clean_title} ?",
                f"Où se déroulent les activités de {clean_title} ?"
            ],
            'recherche': [
                f"Quels sont les projets de recherche de {clean_title} ?",
                f"Comment participer aux recherches de {clean_title} ?",
                f"Quelles sont les publications de {clean_title} ?",
                f"Qui sont les chercheurs de {clean_title} ?",
                f"Quels sont les domaines de recherche de {clean_title} ?"
            ]
        }
        return category_questions.get(category, [])

def main():
    scraper = ISETScraper()
    scraper.start_scraping()

if __name__ == "__main__":
    main() 