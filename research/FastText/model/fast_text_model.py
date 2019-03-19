from itertools import chain

import tensorflow as tf
from tensorflow.contrib import lookup, layers

from model.model_config import FastTextConfig

__author__ = 'KKishore'

tf.logging.set_verbosity(tf.logging.INFO)


class FastTextEstimator(tf.estimator.Estimator):

    def __init__(self, config: FastTextConfig):
        super(FastTextEstimator, self).__init__(
            model_fn=self.fast_text_model_fn,
            model_dir=config.MODEL_DIR,
            config=None)
        self.VOCAB_FILE = config.VOCAB_FILE
        self.VOCAB_LEN = config.VOCAB_LEN
        self.MODEL_DIR = config.MODEL_DIR
        self.MAX_LEN = config.MAX_LEN
        self.PAD_WORD = config.PAD_WORD
        self.FEATURE_COL = config.FEATURE_COL
        self.LABEL_COL = config.LABEL_COL
        self.TARGET_SIZE = config.TARGET_SIZE
        self.EMBED_DIM = config.EMBED_DIM

    def streaming_f1(self, labels, predictions, n_classes, weights=None, type='macro'):
        labels_and_predictions_by_class = [(tf.equal(labels, c), tf.equal(predictions, c)) for c in range(0, n_classes)]
        tp_by_class_val, tp_by_class_update_op = zip(*[tf.metrics.true_positives(label, prediction, weights=weights)
                                                       for label, prediction in labels_and_predictions_by_class])
        fn_by_class_val, fn_by_class_update_op = zip(*[tf.metrics.false_negatives(label, prediction, weights=weights)
                                                       for label, prediction in labels_and_predictions_by_class])
        fp_by_class_val, fp_by_class_update_op = zip(*[tf.metrics.false_positives(label, prediction, weights=weights)
                                                       for label, prediction in labels_and_predictions_by_class])

        f1_update_op = tf.group(*chain(tp_by_class_update_op, fn_by_class_update_op, fp_by_class_update_op))

        if type == 'macro':
            epsilon = [10e-6 for _ in range(n_classes)]
            f1_val = tf.multiply(2., tp_by_class_val) / (tf.reduce_sum([tf.multiply(2., tp_by_class_val),
                                                                        fp_by_class_val, fn_by_class_val, epsilon],
                                                                       axis=0))
            f1_val = tf.reduce_mean(f1_val)
        else:
            epsilon = 10e-6
            total_tp = tf.reduce_sum(tp_by_class_val)
            total_fn = tf.reduce_sum(fn_by_class_val)
            total_fp = tf.reduce_sum(fp_by_class_val)

            f1_val = tf.squeeze(tf.multiply(2., total_tp) / (tf.multiply(2., total_tp) +
                                                             total_fp + total_fn + epsilon,
                                                             ))
        return f1_val, f1_update_op

    def fast_text_model_fn(self, features, labels, mode, params):
        vocab_table = lookup.index_table_from_file(vocabulary_file=self.VOCAB_FILE, num_oov_buckets=1,
                                                   default_value=-1)
        text = features[self.FEATURE_COL]
        words = tf.string_split(text)
        dense_words = tf.sparse_tensor_to_dense(words, default_value=self.PAD_WORD)
        word_ids = vocab_table.lookup(dense_words)
        padding = tf.constant([[0, 0], [0, self.MAX_LEN]])
        # Pad all the word_ids entries to the maximum document length
        word_ids_padded = tf.pad(word_ids, padding)
        word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, self.MAX_LEN])

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.keras.backend.set_learning_phase(True)
        else:
            tf.keras.backend.set_learning_phase(False)

        with tf.name_scope('embedding'):
            embedding_vectors = layers.embed_sequence(word_id_vector, vocab_size=self.VOCAB_LEN,
                                                      embed_dim=self.EMBED_DIM,
                                                      initializer=layers.xavier_initializer(seed=42))
            tf.logging.info('Word Vectors = {}'.format(embedding_vectors))

        with tf.name_scope('fast_text'):
            average_vectors = tf.reduce_sum(embedding_vectors, axis=1)
            tf.logging.info('Average Word Vectors = {}'.format(average_vectors))

        with tf.name_scope('hidden_layer'):
            fc1 = tf.keras.layers.Dense(1024, activation='relu')(average_vectors)
            d1 = tf.keras.layers.Dropout(0.5)(fc1)
            fc2 = tf.keras.layers.Dense(self.EMBED_DIM / 2, activation='relu')(d1)
            d2 = tf.keras.layers.Dropout(0.5)(fc2)
            tf.logging.info('Hidden Layer = {}'.format(d2))

        with tf.name_scope('output'):
            logits = tf.keras.layers.Dense(self.TARGET_SIZE, activation=None)(d2)
            tf.logging.info('Logits Layer = {}'.format(logits))

        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, axis=1)

        tf.summary.histogram('fasttext', average_vectors)
        tf.summary.histogram('softmax', probabilities)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class': predicted_indices,
                'probabilities': probabilities
            }

            exported_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=exported_outputs)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.summary.scalar('loss', loss)
        acc = tf.equal(predicted_indices, labels)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

        tf.summary.scalar('acc', acc)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics_ops = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_indices),
                'precision': tf.metrics.precision(labels=labels, predictions=predicted_indices),
                'recall': tf.metrics.recall(labels=labels, predictions=predicted_indices),
                'f1_score': self.streaming_f1(labels=labels, predictions=predicted_indices, n_classes=self.TARGET_SIZE)
            }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)
