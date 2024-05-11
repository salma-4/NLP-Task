import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Download NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

# Read the data
data = pd.read_csv('twitter_validation.csv')

# Rename the tweet column
data.rename(columns={'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣': 'tweet'}, inplace=True)

def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def stem_words(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def pos_tagging(text):
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)

def chunking(text):
    sentences = sent_tokenize(text)
    chunked_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        chunked_sentence = nltk.ne_chunk(tagged_words)
        chunked_sentences.append(chunked_sentence)
    return chunked_sentences

def named_entity_recognition(text):
    sentences = sent_tokenize(text)
    ner_tags = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        ner_tags.append(nltk.ne_chunk(tagged_words))
    return ner_tags

def spacy_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def spacy_remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def spacy_remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

def spacy_lemmatize_words(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def spacy_pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def spacy_named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply preprocessing functions using NLTK
data['clean_text_nltk'] = data['tweet'].apply(lambda x: x.lower())
data['clean_text_nltk'] = data['clean_text_nltk'].apply(remove_punctuations)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(remove_special_characters)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(remove_stopwords)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(stem_words)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(lemmatize_words)
data['tokens_nltk'] = data['tweet'].apply(word_tokenize)  # Tokenize using NLTK
data['pos_tagging_nltk'] = data['clean_text_nltk'].apply(pos_tagging)
data['chunking_nltk'] = data['clean_text_nltk'].apply(chunking)
data['named_entity_recognition_nltk'] = data['clean_text_nltk'].apply(named_entity_recognition)

# Apply preprocessing functions using spaCy
data['clean_text_spacy'] = data['tweet'].apply(lambda x: x.lower())
data['tokens_spacy'] = data['clean_text_spacy'].apply(spacy_tokenize)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(spacy_remove_punctuations)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(remove_special_characters)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(spacy_remove_stopwords)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(spacy_lemmatize_words)
data['pos_tagging_spacy'] = data['clean_text_spacy'].apply(spacy_pos_tagging)
data['named_entity_recognition_spacy'] = data['clean_text_spacy'].apply(spacy_named_entity_recognition)

print("---------------------------------------- nltk preprocessing-------------------------------------------------------")
#print(data[['tweet', 'clean_text_nltk', 'tokens_nltk', 'pos_tagging_nltk', 'chunking_nltk', 'named_entity_recognition_nltk']])
print(data[['tweet', 'named_entity_recognition_nltk']])

print("---------------------------------------- spacy preprocessing-------------------------------------------------------")
#print(data[['tweet', 'clean_text_spacy', 'tokens_spacy', 'pos_tagging_spacy', 'named_entity_recognition_spacy']])
print(data[['tweet', 'named_entity_recognition_spacy']])
