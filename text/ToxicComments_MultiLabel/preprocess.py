import re

import pandas as pd


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


def clean_data(file_name):
    data_set = pd.read_csv(file_name).fillna("sterby")
    data_set['comment_text'] = data_set['comment_text'].apply(lambda x: clean_str(x))
    data_set['comment_text'] = data_set['comment_text'].str.strip()
    data_set.drop('id', axis=1, inplace=True)
    file_name = file_name.replace('.csv', '')
    data_set.to_csv(file_name + '_preprocess.csv', index=False, sep='\t')


# clean_data('data/train.csv')
clean_data('data/test.csv')
