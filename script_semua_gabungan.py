"""
SCRIPT GABUNGAN NLP, POS TAGGING, NER

Bagian 1: Text Processing Dasar dengan NLTK
Bagian 2: Named Entity Recognition (NER) dengan SpaCy
Bagian 3: POS Tagging Bahasa Inggris & Indonesia dengan NLTK
Bagian 4: Named Entity Recognition (NER) dengan NLTK
"""

# Bagian 1: Text Processing Dasar (NLTK)
import nltk  # Library Natural Language Toolkit untuk NLP
from nltk.tokenize import word_tokenize  # Tokenisasi kalimat ke kata
from nltk.corpus import stopwords  # Stopwords (kata umum yang diabaikan)
from nltk.stem import PorterStemmer  # Stemming (mengembalikan kata ke bentuk dasar)
import string  # Untuk tanda baca
import spacy  # Library SpaCy untuk NLP modern

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Lowercasing, tokenisasi, filtering stopwords & tanda baca, stemming
txt = "Natural Language Processing (NLP) adalah cabang dari Artificial Intelligence!"
txt = txt.lower()
tokens = word_tokenize(txt)
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in tokens]
print("Tokens:", tokens)
print("Stems:", stems)

# Bagian 2: NER dengan SpaCy
nlp = spacy.load("en_core_web_sm")
txt_spacy = "Dr. Andi Pratama from the University of Indonesia will attend a conference in Jakarta on October 15, 2025."
doc = nlp(txt_spacy)
for ent in doc.ents:
    print(f"Teks: {ent.text}, Label: {ent.label_}")

# Bagian 3: POS Tagging (NLTK)
tagger_models = ['punkt_tab', 'averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_id']
for model in tagger_models:
    nltk.download(model)
sentence_en = "John plays football in the park."
tokens_en = nltk.word_tokenize(sentence_en)
pos_tags_en = nltk.pos_tag(tokens_en)
print(pos_tags_en)
sentence_id = "John bermain sepakbola di taman."
tokens_id = nltk.word_tokenize(sentence_id)
pos_tags_id = nltk.pos_tag(tokens_id)
print(pos_tags_id)

# Bagian 4: NER dengan NLTK
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')
sentence_ner = "Barack Obama was born in Hawaii."
tokens_ner = nltk.word_tokenize(sentence_ner)
tags_ner = nltk.pos_tag(tokens_ner)
entities = nltk.ne_chunk(tags_ner)
print(entities)
