import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix


class PoemClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        # self.feature_selector = SelectKBest(chi2, k=5000)
        self.model = SGDClassifier(loss='modified_huber', class_weight='balanced', n_jobs=-1)
        # self.model = SVC(class_weight='balanced')
        # self.model = RandomForestClassifier(n_estimators=501, class_weight='balanced', n_jobs=-1)
        # self.model = AdaBoostClassifier(n_estimators=51)

    def train(self, data_set):
        poets = [poet for _, poet in data_set]

        feature_vectors = self.vectorizer.fit_transform([poem for poem, _ in data_set])
        # feature_vectors = self.feature_selector.fit_transform(feature_vectors, poets)

        self.model.fit(feature_vectors, poets)

    def evaluate(self, data_set):
        poets = [poet for _, poet in data_set]

        feature_vectors = self.vectorizer.transform([poem for poem, _ in data_set])
        # feature_vectors = self.feature_selector.transform(feature_vectors)

        predictions = self.model.predict(feature_vectors)

        report = classification_report(poets, predictions, labels=list(set(poets)))
        print(report)

        confusion_mat = confusion_matrix(poets, predictions, normalize='true')
        print(np.round(confusion_mat, 2))


if __name__ == '__main__':
    with open('../data/balanced_train.json', 'r') as json_file:
        train_set = json.load(json_file)
    with open('../data/balanced_test.json', 'r') as json_file:
        test_set = json.load(json_file)

    classifier = PoemClassifier()
    classifier.train(train_set)
    with open('data/vectorizer.pkl', 'wb') as pickle_file:
        pickle.dump(classifier.vectorizer, pickle_file)
    with open('data/model.pkl', 'wb') as pickle_file:
        pickle.dump(classifier.model, pickle_file)

    classifier.evaluate(test_set)
