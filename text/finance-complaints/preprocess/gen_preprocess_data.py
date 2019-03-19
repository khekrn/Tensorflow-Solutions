import re

import pandas as pd
import spacy

__author__ = 'KKishore'

selected = ['product', 'consumer_complaint_narrative']

nlp = spacy.load('en')


def clean_str(x):
    s = x
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*', "xxx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    s = re.sub(r'"', "", s)
    s = s.strip().lower()
    return s


def spacy_preprocess(text):
    doc = nlp(text)
    res = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_digit or token.is_quote:
            pass
        else:
            res.append(token.text)
    return ' '.join(res)


def preprocess_data(file_name):
    print('String Preprocess')
    new_file = file_name + ".csv"
    data_set = pd.read_csv(new_file, header=None, sep=',', skiprows=[1], names=selected)
    data_set['consumer_complaint_narrative'] = data_set[selected[1]].apply(lambda x: clean_str(x)).tolist()
    print(len(data_set))

    print('Spacy Preprocess')
    data_set['consumer_complaint_narrative'] = data_set[selected[1]].apply(lambda x: spacy_preprocess(x)).tolist()

    print('Writing content')
    data_set.to_csv(file_name + 'preprocess' + ".csv", index=False)


train_file = 'dataset/train'
test_file = 'dataset/test'
valid_file = 'dataset/valid'

print('Preprocesing Training Data')
preprocess_data(train_file)

print('Preprocessing Validation Data')
preprocess_data(valid_file)

print('Preprocessing Test Data')
preprocess_data(test_file)
