import multiprocessing
from itertools import chain

import tensorflow as tf
from tensorflow.contrib import lookup

from model import commons, l4

__author__ = 'KKishore'

filter_sizes = [3, 4, 5]


def parse_record(record):
    columns = tf.decode_csv(record, record_defaults=commons.HEADER_DEFAULTS)
    features = dict(zip(commons.HEADERS, columns))
    label = features.pop(commons.LABEL_COL)

    features[commons.WEIGHT_COLUNM_NAME] = tf.case(
        {
            tf.equal(label, '7'): lambda: 7.5,
            tf.equal(label, '5'): lambda: 5.5,
            tf.equal(label, '8'): lambda: 5.0,
            tf.equal(label, '10'): lambda: 4.2,
            tf.equal(label, '1'): lambda: 3.5
        }, default=lambda: 1.0, exclusive=True
    )

    return features, label


def input_fn(file_name, shuffle=False, repeat_count=1, batch_size=128):
    n_cpu = multiprocessing.cpu_count()

    data_set = tf.data.TextLineDataset(filenames=file_name).skip(1)

    if shuffle:
        data_set.shuffle(shuffle)

    data_set = data_set.map(lambda record: parse_record(record), num_parallel_calls=n_cpu).repeat(repeat_count).batch(
        batch_size=batch_size)
    iterator = data_set.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, tf.string_to_number(target, out_type=tf.int64)


def streaming_f1(labels, predictions, n_classes, weights=None, type='macro'):
    '''
    Refer http://rushdishams.blogspot.in/2011/08/micro-and-macro-average-of-precision.html
    for micro and macro f1
    :param labels:
    :param predictions:
    :param n_classes:
    :param type:
    :return:
    '''
    labels_and_predictions_by_class = [(tf.equal(labels, c), tf.equal(predictions, c)) for c in range(0, n_classes)]
    tp_by_class_val, tp_by_class_update_op = zip(
        *[tf.metrics.true_positives(label, prediction, weights=weights) for label, prediction in
          labels_and_predictions_by_class])
    fn_by_class_val, fn_by_class_update_op = zip(
        *[tf.metrics.false_negatives(label, prediction, weights=weights) for label, prediction
          in labels_and_predictions_by_class])
    fp_by_class_val, fp_by_class_update_op = zip(
        *[tf.metrics.false_positives(label, prediction, weights=weights) for label, prediction
          in labels_and_predictions_by_class])

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


def model_fn(features, labels, mode, params):
    '''
    CNN model based on Yoon Kim

    https://arxiv.org/pdf/1408.5882.pdf
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    '''
    vocab_table = lookup.index_table_from_file(vocabulary_file='dataset/vocab.csv', num_oov_buckets=1, default_value=-1)
    text = features[commons.FEATURE_COL]
    words = tf.string_split(text)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=commons.PAD_WORD)
    word_ids = vocab_table.lookup(dense_words)

    padding = tf.constant([[0, 0], [0, commons.MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, commons.MAX_DOCUMENT_LENGTH])

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.keras.backend.set_learning_phase(True)
    else:
        tf.keras.backend.set_learning_phase(False)

    embedded_sequences = tf.keras.layers.Embedding(params.N_WORDS, 128, input_length=commons.MAX_DOCUMENT_LENGTH)(
        word_id_vector)
    conv_layer = []
    for filter_size in filter_sizes:
        l_conv = tf.keras.layers.Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = tf.keras.layers.MaxPooling1D(pool_size=3)(l_conv)
        conv_layer.append(l_pool)

    l_merge = tf.keras.layers.concatenate(conv_layer, axis=1)
    conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(l_merge)
    pool = tf.keras.layers.MaxPooling1D(pool_size=3)(conv)
    f1 = tf.keras.layers.Dropout(0.5)(pool)
    f1 = tf.keras.layers.Flatten()(f1)
    f1 = tf.keras.layers.Dense(128, activation='relu')(f1)
    logits = tf.keras.layers.Dense(commons.TARGET_SIZE, activation=None)(f1)

    predictions = tf.nn.softmax(logits)
    prediction_indices = tf.argmax(predictions, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction_dict = {
            'class': prediction_indices,
            'probabilities': predictions
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(prediction_dict)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    weights = features[commons.WEIGHT_COLUNM_NAME]
    tf.logging.info('Logits Layer = {}'.format(logits))
    tf.logging.info('Logits Layer = {}'.format(labels))
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)

    tf.summary.scalar('loss', loss)

    acc = tf.equal(prediction_indices, labels)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    tf.summary.scalar('acc', acc)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=prediction_indices, weights=weights),
            'precision': tf.metrics.precision(labels=labels, predictions=prediction_indices, weights=weights),
            'recall': tf.metrics.recall(labels=labels, predictions=prediction_indices, weights=weights),
            'f1_score': streaming_f1(labels=labels, predictions=prediction_indices, n_classes=commons.TARGET_SIZE,
                                     weights=None)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


def serving_fn():
    receiver_tensor = {
        commons.FEATURE_COL: tf.placeholder(dtype=tf.string, shape=None)
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


'''
data_set = tf.data.TextLineDataset(
    filenames='D:/Git/Tensorflow/Tensorflow-Solutions/Problems-Solutions/text/finance-complaints/dataset/trainpreprocess.csv').skip(
    1)
data_set = data_set.shuffle(True)
data_set = data_set.map(lambda record: parse_record(record), num_parallel_calls=4).repeat(1).batch(batch_size=128)

iterator = data_set.make_initializable_iterator()

sess = tf.Session()

sess.run(iterator.initializer)

X, Y = iterator.get_next()

print(sess.run(X['weight']))
print('\n\n')
print(sess.run(Y))
sess.close()
'''
