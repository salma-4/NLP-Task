import pandas as pd
import string
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

nlp = spacy.load("en_core_web_sm")
data = pd.read_csv('IMDB Dataset.csv')
data = data.sample(frac=0.1, random_state=42)  # sample of data
data.rename(columns={'review,sentiment': 'review'}, inplace=True)

# Define functions for spaCy preprocessing
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
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

# Apply preprocessing functions using spaCy
data['clean_text_spacy'] = data['review'].apply(lambda x: x.lower())
data['tokens_spacy'] = data['clean_text_spacy'].apply(spacy_tokenize)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(spacy_remove_punctuations)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(remove_special_characters)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(spacy_remove_stopwords)
data['clean_text_spacy'] = data['clean_text_spacy'].apply(spacy_lemmatize_words)

print("---------------------------------------- spacy preprocessing-------------------------------------------------------")
print(data[['review', 'clean_text_spacy']])
# Split the data into train and test sets
X = data['clean_text_spacy']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train SVM model
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train_tfidf, y_train)

# Predictions
y_pred = svm_clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
