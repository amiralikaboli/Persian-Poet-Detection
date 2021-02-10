import json

if __name__ == '__main__':
    with open('../data/balanced_train.json', 'r') as json_file:
        train_set = json.load(json_file)
    with open('../data/balanced_test.json', 'r') as json_file:
        test_set = json.load(json_file)

    fasttext_train_set = []
    for text, indexed_poet in train_set:
        text = text.replace('\n', '\t')
        fasttext_train_set.append(f"__label__{indexed_poet} {text}")
    fasttext_test_set = []
    for text, indexed_poet in test_set:
        text = text.replace('\n', '\t')
        fasttext_test_set.append(f"__label__{indexed_poet} {text}")

    with open('data/train.txt', 'w+') as txt_file:
        for line in fasttext_train_set:
            txt_file.write(f'{line}\n')
    with open('data/test.txt', 'w+') as txt_file:
        for line in fasttext_test_set:
            txt_file.write(f'{line}\n')
