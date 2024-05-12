import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

data = data.sample(frac=0.1, random_state=42)  # sample of data
data.rename(columns={'review,sentiment': 'review'}, inplace=True)


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

# Apply preprocessing functions using NLTK

data['clean_text_nltk'] = data['review'].apply(lambda x: x.lower())
data['clean_text_nltk'] = data['clean_text_nltk'].apply(remove_punctuations)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(remove_special_characters)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(remove_stopwords)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(stem_words)
data['clean_text_nltk'] = data['clean_text_nltk'].apply(lemmatize_words)

print("---------------------------------------- nltk preprocessing-------------------------------------------------------")
print(data[['review', 'clean_text_nltk']])

# Split the data into train and test sets
X = data['clean_text_nltk']
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
