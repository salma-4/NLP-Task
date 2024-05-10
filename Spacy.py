

import spacy
import pandas as pd
import string
import re
# Load the English model
nlp = spacy.load("en_core_web_sm")

dataSpacy = pd.read_csv('twitter_validation.csv')

# Rename the tweet column
dataSpacy.rename(columns={'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣': 'tweet'}, inplace=True)

def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

def lemmatize_words(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Preprocess the text
dataSpacy['clean_text'] = dataSpacy['tweet'].apply(lambda x: x.lower())
dataSpacy['clean_text'] = dataSpacy['clean_text'].apply(remove_punctuations)
dataSpacy['clean_text'] = dataSpacy['clean_text'].apply(remove_special_characters)
dataSpacy['clean_text'] = dataSpacy['clean_text'].apply(remove_stopwords)
dataSpacy['clean_text'] = dataSpacy['clean_text'].apply(lemmatize_words)
dataSpacy['pos_tagging'] = dataSpacy['clean_text'].apply(pos_tagging)
dataSpacy['named_entity_recognition'] = dataSpacy['clean_text'].apply(named_entity_recognition)

print("----------------------------------------spacy preprocessing-------------------------------------------------------")
print(dataSpacy.head())
