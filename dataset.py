import json
import random
import re

import pandas as pd
import parsivar


class TextCleaner:
    def __init__(self):
        self.normalizer = parsivar.Normalizer()

    def clean(self, text):
        text = self.normalizer.sub_alphabets(text)
        text = re.sub("[,.،]", '', text)
        return text


class DataProvider:
    def __init__(self):
        self.text_cleaner = TextCleaner()

    def provide_mesras(self, data_csv):
        final_fields = ['poem_id', 'vorder', 'position', 'text', 'cat']
        mesras = []
        for line_ind in range(data_csv.shape[0]):
            try:
                mesra = {field: data_csv[field][line_ind] for field in final_fields}
                mesra['text'] = self.text_cleaner.clean(mesra['text'])
                mesras.append(mesra)
            except:
                pass

        return mesras

    def provide_beits(self, data_csv):
        final_fields = ['poem_id', 'text', 'cat']
        beits = []
        last_beit = {field: data_csv[field][0] for field in final_fields}
        for line_ind in range(1, data_csv.shape[0]):
            if data_csv['poem_id'][line_ind] == data_csv['poem_id'][line_ind - 1] and \
                    data_csv['position'][line_ind] > data_csv['position'][line_ind - 1]:
                last_beit['text'] = f"{last_beit['text']}\t{data_csv['text'][line_ind]}"
            else:
                try:
                    last_beit['text'] = self.text_cleaner.clean(last_beit['text'])
                except:
                    pass
                beits.append(last_beit)
                last_beit = {field: data_csv[field][line_ind] for field in final_fields}
        last_beit['text'] = self.text_cleaner.clean(last_beit['text'])
        beits.append(last_beit)

        return beits

    def provide_multiple_beits(self, data_csv, k):
        beits = self.provide_beits(data_csv)

        multiple_beits = []
        ind = 0
        while ind < len(beits):
            multiple_beit = beits[ind]
            ind += 1

            counter = k - 1
            while ind < len(beits) and counter and beits[ind]['poem_id'] == multiple_beit['poem_id']:
                multiple_beit['text'] = f"{multiple_beit['text']}\n{beits[ind]['text']}"
                counter -= 1
                ind += 1
            multiple_beits.append(multiple_beit)

        return multiple_beits

    def provide_poems(self, data_csv):
        final_fields = ['poem_id', 'text', 'cat', 'title']
        poems = []
        last_poem = {field: data_csv[field][0] for field in final_fields}
        for line_ind in range(1, data_csv.shape[0]):
            if data_csv['poem_id'][line_ind] != data_csv['poem_id'][line_ind - 1]:
                try:
                    last_poem['text'] = self.text_cleaner.clean(last_poem['text'])
                except:
                    pass
                poems.append(last_poem)
                last_poem = {field: data_csv[field][line_ind] for field in final_fields}
            elif data_csv['position'][line_ind] > data_csv['position'][line_ind - 1]:
                last_poem['text'] = f"{last_poem['text']}\t{data_csv['text'][line_ind]}"
            else:
                last_poem['text'] = f"{last_poem['text']}\n{data_csv['text'][line_ind]}"
        last_poem['text'] = self.text_cleaner.clean(last_poem['text'])
        poems.append(last_poem)

        return poems


if __name__ == '__main__':
    data_csv = pd.read_csv('data/ganjoor.csv', delimiter='\t', dtype=str)

    provider = DataProvider()
    units = provider.provide_multiple_beits(data_csv, k=4)

    with open('data/multiple_beits.json', 'w') as json_file:
        json.dump(units, json_file, ensure_ascii=False)

    poets_freq = {}
    for unit in units:
        poet = unit['cat'].split('____')[0]

        if poet not in poets_freq:
            poets_freq[poet] = 0
        poets_freq[poet] += 1

    poets_freq = dict(sorted(poets_freq.items(), key=lambda item: item[1], reverse=True))

    with open('data/poets_freq.json', 'w') as json_file:
        json.dump(poets_freq, json_file, ensure_ascii=False)

    threshold = 7000
    frequent_poets = [poet for poet in poets_freq.keys() if poets_freq[poet] > threshold]
    poet2index = {poet: ind for ind, poet in enumerate(sorted(frequent_poets))}

    with open('data/poet2index.json', 'w') as json_file:
        json.dump(poet2index, json_file, ensure_ascii=False)

    selected_units = []
    for unit in units:
        poet = unit['cat'].split('____')[0]

        if poet in frequent_poets and isinstance(unit['text'], str):
            selected_units.append((unit['text'], poet2index[poet]))

    test_frac = 0.2
    test_set_size = int(len(selected_units) * test_frac)
    random.shuffle(selected_units)
    train_set = selected_units[test_set_size:]
    test_set = selected_units[:test_set_size]

    with open('data/train.json', 'w') as json_file:
        json.dump(train_set, json_file, ensure_ascii=False)
    with open('data/test.json', 'w') as json_file:
        json.dump(test_set, json_file, ensure_ascii=False)
