import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix


class PoemClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = SGDClassifier(loss='modified_huber', class_weight='balanced', n_jobs=-1)

    def train(self, data_set):
        poets = [poet for _, poet in data_set]

        feature_vectors = self.vectorizer.fit_transform([poem for poem, _ in data_set])

        self.model.fit(feature_vectors, poets)

    def evaluate(self, data_set):
        poets = [poet for _, poet in data_set]

        feature_vectors = self.vectorizer.transform([poem for poem, _ in data_set])

        predictions = self.model.predict(feature_vectors)

        report = classification_report(poets, predictions, labels=list(set(poets)))
        print(report)

        confusion_mat = confusion_matrix(poets, predictions, normalize='true')
        print(np.round(confusion_mat, 2))


if __name__ == '__main__':
    with open('../data/train.json', 'r') as json_file:
        train_set = json.load(json_file)
    with open('../data/test.json', 'r') as json_file:
        test_set = json.load(json_file)

    classifier = PoemClassifier()
    classifier.train(train_set)
    with open('data/vectorizer.pkl', 'wb') as pickle_file:
        pickle.dump(classifier.vectorizer, pickle_file)
    with open('data/model.pkl', 'wb') as pickle_file:
        pickle.dump(classifier.model, pickle_file)

    classifier.evaluate(test_set)
