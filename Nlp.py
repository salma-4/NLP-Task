import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources ==> for first time
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

dataNltk = pd.read_csv('twitter_validation.csv')

# Rename the tweet column
dataNltk.rename(columns={'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣': 'tweet'}, inplace=True)

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

# Preprocess the text
dataNltk['clean_text'] = dataNltk['tweet'].apply(lambda x: x.lower())
dataNltk['clean_text'] = dataNltk['clean_text'].apply(remove_punctuations)
dataNltk['clean_text'] = dataNltk['clean_text'].apply(remove_special_characters)
dataNltk['clean_text'] = dataNltk['clean_text'].apply(remove_stopwords)
dataNltk['clean_text'] = dataNltk['clean_text'].apply(stem_words)
dataNltk['clean_text'] = dataNltk['clean_text'].apply(lemmatize_words)
dataNltk['pos_tagging'] = dataNltk['clean_text'].apply(pos_tagging)
dataNltk['chunking'] = dataNltk['clean_text'].apply(chunking)
dataNltk['named_entity_recognition'] = dataNltk['clean_text'].apply(named_entity_recognition)

# Display the DataFrame
print(dataNltk.head())
