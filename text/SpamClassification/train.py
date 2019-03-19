__author__ = 'KKishore'

import tensorflow as tf
from tensorflow.contrib import training

from model.bag_of_words import model_fn, input_fn, serving_fn

tf.logging.set_verbosity(tf.logging.INFO)

N_WORDS = 0

with open('data/nwords.csv', 'r') as f:
    N_WORDS = int(f.read()) + 2

hparams = training.HParams(
    N_WORDS=N_WORDS
)

estimator = tf.estimator.Estimator(model_fn=model_fn, params=hparams, model_dir='build/')

estimator.train(input_fn=lambda: input_fn('data/train-data.tsv', shuffle=True, repeat_count=5))

evaluated_results = estimator.evaluate(input_fn=lambda: input_fn('data/valid-data.tsv', shuffle=False, repeat_count=1))

print("# Evaluated Results: {}".format(evaluated_results))

estimator.export_savedmodel(export_dir_base='serving', serving_input_receiver_fn=serving_fn, as_text=True)
