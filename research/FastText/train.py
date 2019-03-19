import tensorflow as tf

from model import model_config, fast_text_model
from utils import commons, input_utils

__author__ = 'KKishore'

tf.logging.set_verbosity(tf.logging.INFO)

N_WORDS = 0

with open('dataset/nwords.csv', 'r') as f:
    N_WORDS = int(f.read()) + 2

fast_text_config = model_config.FastTextConfig(vocab_file='dataset/vocab.csv', vocab_len=N_WORDS,
                                               model_dir='build/', max_len=commons.MAX_DOCUMENT_LENGTH,
                                               embed_dim=128, pad_word=commons.PAD_WORD,
                                               feature_col=commons.FEATURE_COL, label_col=commons.LABEL_COL,
                                               target_size=commons.TARGET_SIZE)

estimator = fast_text_model.FastTextEstimator(config=fast_text_config)

estimator.train(input_fn=lambda: input_utils.input_fn('dataset/trainpreprocess.csv', shuffle=True, repeat_count=10,
                                                      batch_size=128))

evaluated_results = estimator.evaluate(
    input_fn=lambda: input_utils.input_fn('dataset/testpreprocess.csv', shuffle=False, repeat_count=1, batch_size=128))

print("# Evaluated Results: {}".format(evaluated_results))

estimator.export_savedmodel(export_dir_base='serving', serving_input_receiver_fn=input_utils.serving_fn, as_text=True)
