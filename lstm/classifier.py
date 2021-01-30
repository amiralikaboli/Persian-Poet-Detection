import json
import math
from datetime import datetime

import fasttext
import numpy as np
import torch
from sklearn.metrics import accuracy_score


class PoetClassifier(torch.nn.Module):
    def __init__(self, num_poets, num_layers, hidden_size):
        super(PoetClassifier, self).__init__()
        self.embedding = fasttext.load_model('../data/cc.fa.300.bin')
        # self.dropout = torch.nn.Dropout(0.2)
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_size, num_poets)
        self.softmax = torch.nn.Softmax()

    def forward(self, poem_texts):
        embedded_vectors = np.array([
            [self.embedding.get_word_vector(word) for word in poem_text.split(' ')]
            for poem_text in poem_texts
        ])
        embedded_vectors = np.moveaxis(embedded_vectors, 0, 1)
        embedded_vectors = torch.from_numpy(embedded_vectors)
        # embedded_vectors = self.dropout(embedded_vectors)
        lstm_out, (ht, ct) = self.lstm(embedded_vectors)
        linear_out = self.linear(ht[-1])
        softmax_out = self.softmax(linear_out)
        return softmax_out


if __name__ == '__main__':
    now_datetime = datetime.now()

    with open('hyper_parameters.json', 'r') as json_file:
        hyper_parameters = json.load(json_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('../data/balanced_train.json', 'r') as json_file:
        train_set = json.load(json_file)

    validation_size = int(len(train_set) * hyper_parameters['val_frac'])
    train_size = len(train_set) - validation_size
    validation_loader = torch.utils.data.DataLoader(train_set[:validation_size], hyper_parameters['batch_size'])
    train_loader = torch.utils.data.DataLoader(train_set[validation_size:], hyper_parameters['batch_size'])

    torch.set_num_threads(7)

    classifier = PoetClassifier(
        num_poets=7,
        num_layers=hyper_parameters['num_layers'],
        hidden_size=hyper_parameters['hidden_size']
    )
    classifier = classifier.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=hyper_parameters['lr'])

    best_validation_loss = math.inf
    for epoch_ind in range(hyper_parameters['num_epochs']):
        training_loss = 0
        classifier.train()
        for data_point in train_loader:
            poem, poet = data_point

            output = classifier(poem)

            loss = loss_func(output, poet)
            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_loss = 0
        targets = []
        predictions = []
        for data_point in validation_loader:
            poem, poet = data_point

            output = classifier(poem)

            loss = loss_func(output, poet)
            validation_loss += loss.item()

            targets.append(poet)
            predictions.append(np.argmax(output.detach().numpy()[0]))

        print(
            f"Epoch: {epoch_ind} \t"
            f"Loss: {training_loss / train_size} \t"
            f"Val Loss: {validation_loss / validation_size} \t"
            f"Val Acc: {accuracy_score(targets, predictions)}"
        )
        if best_validation_loss > validation_loss:
            best_validation_loss = validation_loss
            torch.save(classifier.state_dict(), f'data/checkpoints/{now_datetime}.pth')
