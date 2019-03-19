import re

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

nltk.download('stopwords')

from nltk.corpus import stopwords  # Import the stop word list

print(stopwords.words("english"))

data_set = pd.read_csv('D:/DataSet/Sentiment Analysis/BOG/testData.tsv', sep='\t')


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))


for index, row in data_set.iterrows():
    text = row['review']
    text = review_to_words(text)
    print(index)
    data_set.set_value(index, 'review', text)

print('Cleaning is Done')
data_set.drop('id', inplace=True, axis=1)

data_set.to_csv('data/test.tsv', index=False, encoding='utf-8', sep='\t')


'''
msk = np.random.rand(len(data_set)) < 0.8

train = data_set[msk]
test = data_set[~msk]

print(len(train))
print(len(test))

train.to_csv('data/train.tsv', index=False, encoding='utf-8', sep='\t')
test.to_csv('data/dev.tsv', index=False, encoding='utf-8', sep='\t')
'''