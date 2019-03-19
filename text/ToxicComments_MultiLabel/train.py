import tensorflow as tf
from tensorflow.contrib import training

from model.custom_model import custom_fast_text
from utils.input_utils import input_fn, serving_fn

__author__ = 'KKishore'

tf.logging.set_verbosity(tf.logging.INFO)

N_WORDS = 0

with open('data/nwords.csv', 'r') as f:
    N_WORDS = int(f.read()) + 2

hparams = training.HParams(
    N_WORDS=N_WORDS
)

estimator = tf.estimator.Estimator(model_fn=custom_fast_text, params=hparams, model_dir='build/')

estimator.train(input_fn=lambda: input_fn('data/train_preprocess.csv', shuffle=True, repeat_count=8, batch_size=64))

evaluated_results = estimator.evaluate(
    input_fn=lambda: input_fn('data/valid_preprocess.csv', shuffle=False, repeat_count=1, batch_size=64))

print("# Evaluated Results: {}".format(evaluated_results))

estimator.export_savedmodel(export_dir_base='serving', serving_input_receiver_fn=serving_fn, as_text=True)
