import fasttext
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

if __name__ == '__main__':
    model = fasttext.train_supervised('data/train.txt')

    targets = []
    predictions = []
    with open('data/test.txt', 'r') as txt_file:
        for line in txt_file.readlines():
            targets.append(int(line[9]))
            predictions.append(int(model.predict(line[11:-1])[0][0][9]))

    report = classification_report(targets, predictions)
    print(report)

    confusion_mat = confusion_matrix(targets, predictions, normalize='true')
    print(np.round(confusion_mat, 2))
