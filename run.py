import nltk
import random
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.model_selection import train_test_split
from models.random_forest import run_random_forest
from models.svm import run_svm
from typing import List, Tuple


nltk.download("stopwords")
nltk.download("wordnet")


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def clean_data(sentence: str):
    sentence = sentence.lower().strip()
    for chars in ["\n", "\\n"]:  # the text needs both removed
        sentence = sentence.replace(chars, "")
    sentence_no_punc = re.sub(r"[^\w\s]", "", sentence)
    tokens = word_tokenize(sentence_no_punc)
    filtered_words = [w for w in tokens if w not in stopwords.words("english")]
    stem_words = [stemmer.stem(w) for w in filtered_words]
    lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(lemma_words)


def load_data(label: str):
    filepath = f"./data/{label}s.txt"
    with open(filepath, "r") as f:
        return [(clean_data(line), label) for line in f.readlines()]


def split_data(training_data: List[Tuple[str, str]]):
    np.random.seed(500)
    random.shuffle(training_data)

    X, y = [], []
    labels = {"aye": 0, "eye": 1}
    for sentence, label in training_data:
        # transform the labels to integers. alternatively use sklearn LabelEncoder
        y.append(labels[label])
        X.append(sentence.replace(label + " ", ""))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # load and combine our data
    ayes = load_data("aye")
    eyes = load_data("eye")
    data = ayes[: len(eyes)] + eyes
    X_train, X_test, y_train, y_test = split_data(data)

    run_svm(X_train, X_test, y_train, y_test)
    run_random_forest(X_train, X_test, y_train, y_test)
