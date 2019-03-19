__author__ = 'KKishore'

import pandas as pd

from model.commons import FEATURE_COL, PAD_WORD


def build_vocab(file_name):
    data_set = pd.read_csv(file_name, sep='\t')
    sentences = data_set[FEATURE_COL].values
    vocab_set = set()
    for sentence in sentences:
        text = str(sentence)
        words = text.split(' ')
        word_set = set(words)
        vocab_set.update(word_set)
    return list(vocab_set)


vocab_list = build_vocab('data/train_preprocess.csv')

with open('data/vocab.csv', 'w', encoding='utf-8') as vocab_file:
    vocab_file.write("{}\n".format(PAD_WORD))
    for word in vocab_list:
        vocab_file.write("{}\n".format(word))

with open('data/nwords.csv', mode='w') as n_words:
    n_words.write(str(len(vocab_list)))
