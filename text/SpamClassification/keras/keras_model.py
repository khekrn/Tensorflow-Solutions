import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib import lookup

from model import commons

N_words = None

path = 'D:/Git/Tensorflow/Tensorflow-Solutions/Problems-Solutions/text/SpamClassification/data/'


def input_function(features, labels=None, shuffle=False, epochs=1):
    input_fn = tf.estimator.inputs.numpy_input_fn(x={"sms_input": features}, y=labels, shuffle=shuffle,
                                                  num_epochs=epochs)
    return input_fn


data_set = pd.read_csv(path + 'train-data.tsv', sep='\t')
features = data_set['sms'].values
targets = data_set['class'].values

encoder = LabelEncoder()
encoder.fit(targets)

encoded_targets = encoder.transform(targets)

X_train, X_test, y_train, y_test = train_test_split(features, encoded_targets, test_size=0.25)

print(np.shape(X_train))
print(np.shape(X_test))

max_words = 1000
tokenize = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(X_train)
X_train = tokenize.texts_to_matrix(X_train)
X_test = tokenize.texts_to_matrix(X_test)

print(np.shape(X_train))
print(np.shape(X_test))

y_train = tf.keras.utils.to_categorical(y_train, 2).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test, 2).astype(np.float32)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_words, 50, input_length=max_words, name='sms'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(commons.TARGET_SIZE, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model, model_dir='keras_build/')

estimator.train(input_fn=input_function(X_train, y_train, shuffle=True, epochs=5))

evaluated_results = estimator.evaluate(input_fn=input_function(X_test, y_test, shuffle=False))

print("# Evaluated Results: {}".format(evaluated_results))

print(tokenize.word_index)
print(len(tokenize.word_index))

with open('tok.pkl', 'wb') as tok:
    pickle.dump(tokenize, file=tok, protocol=pickle.HIGHEST_PROTOCOL)

with open('vocab.csv', 'w', encoding="utf-8") as f:
    for key in list(tokenize.word_index.keys()):
        key = str(key)
        key = key.strip()
        f.write(key + "\n")


def serving_fn():
    input_string = tf.placeholder(dtype=tf.string, shape=None)

    receiver_tensor = {
        'sms_input': input_string
    }
    # word_id_vector = tf.map_fn(fn=map_serving, elems=input_string)
    vocab_table = lookup.index_table_from_file(vocabulary_file='vocab.csv', num_oov_buckets=1, default_value=-1)
    words = tf.string_split(input_string)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=commons.PAD_WORD)
    word_ids = vocab_table.lookup(dense_words)

    padding = tf.constant([[0, 0], [0, max_words]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, max_words])
    features = {'sms_input': word_id_vector}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


estimator.export_savedmodel(export_dir_base='serving', serving_input_receiver_fn=serving_fn, as_text=True)
