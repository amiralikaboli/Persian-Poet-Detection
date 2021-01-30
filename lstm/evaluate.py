import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from classifier import PoetClassifier


class PoetClassifierEvaluator:
    def __init__(self, model_file_path, hyper_parameters):
        self.hyper_parameters = hyper_parameters
        self.classifier = PoetClassifier(7, hyper_parameters['num_layers'], hyper_parameters['hidden_size'])
        self.classifier.load_state_dict(torch.load(model_file_path))

    def evaluate(self, dataset):
        targets = []
        predictions = []
        for text, poet in torch.utils.data.DataLoader(dataset, self.hyper_parameters['batch_size']):
            targets.append(poet)

            output = self.classifier(text)
            predictions.append(np.argmax(output.detach().numpy()[0]))

        report = classification_report(targets, predictions)
        print(report)

        confusion_mat = confusion_matrix(targets, predictions, normalize='true')
        print(np.round(confusion_mat, 2))


if __name__ == '__main__':
    with open('hyper_parameters.json', 'r') as json_file:
        hyper_parameters = json.load(json_file)

    evaluator = PoetClassifierEvaluator('data/checkpoints/2021-01-29 21:58:48.937865.pth', hyper_parameters)

    with open('../data/balanced_test.json', 'r') as json_file:
        test_set = json.load(json_file)
    evaluator.evaluate(test_set)
