import tensorflow as tf
from tensorflow.contrib import lookup

from model import commons


def custom_fast_text(features, labels, mode, params):
    vocab_table = lookup.index_table_from_file(vocabulary_file='data/vocab.csv', num_oov_buckets=1, default_value=-1)
    text = features[commons.FEATURE_COL]
    words = tf.string_split(text)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=commons.PAD_WORD)
    word_ids = vocab_table.lookup(dense_words)

    padding = tf.constant([[0, 0], [0, commons.CNN_MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, commons.CNN_MAX_DOCUMENT_LENGTH])

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.keras.backend.set_learning_phase(True)
    else:
        tf.keras.backend.set_learning_phase(False)

    embedded_sequences = tf.keras.layers.Embedding(params.N_WORDS, 20, input_length=commons.CNN_MAX_DOCUMENT_LENGTH)(
        word_id_vector)
    f1 = tf.keras.layers.GlobalMaxPooling1D()(embedded_sequences)
    logits = tf.keras.layers.Dense(commons.TARGET_SIZE, activation=None)(f1)

    predictions = tf.nn.sigmoid(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction_dict = {
            'class': tf.cast(tf.map_fn(lambda x: tf.cond(x > 0.30, lambda: 1.0, lambda: 0.0),
                                       tf.squeeze(predictions)), dtype=tf.int32),


        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(prediction_dict)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    tf.summary.scalar('loss', loss)

    acc = tf.equal(tf.cast(predictions, dtype=tf.int32), labels)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    tf.summary.scalar('acc', acc)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)
