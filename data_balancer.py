import json
import random


def calc_ratios(freqs):
    return {key: min(round(min(freqs.values()) / freq + 0.1, 2), 1) for key, freq in freqs.items()}


def balance_dataset(dataset):
    poets_freq = {}
    for _, indexed_poet in dataset:
        if indexed_poet not in poets_freq:
            poets_freq[indexed_poet] = 0
        poets_freq[indexed_poet] += 1

    ratios = calc_ratios(poets_freq)

    return [unit for unit in dataset if random.random() < ratios[unit[1]]]


if __name__ == '__main__':
    with open('data/train.json', 'r') as json_file:
        train_set = json.load(json_file)
    with open('data/test.json', 'r') as json_file:
        test_set = json.load(json_file)

    balanced_train_set = balance_dataset(train_set)
    balanced_test_set = balance_dataset(test_set)

    with open('data/balanced_train.json', 'w') as json_file:
        json.dump(balanced_train_set, json_file, ensure_ascii=False)
    with open('data/balanced_test.json', 'w') as json_file:
        json.dump(balanced_test_set, json_file, ensure_ascii=False)
