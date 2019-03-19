import multiprocessing

import tensorflow as tf
from tensorflow.contrib import estimator
from tensorflow.contrib import lookup
from model import commons

__author__ = 'KKishore'

head = estimator.binary_classification_head()


def parse_csv_row(row):
    columns = tf.decode_csv(row, record_defaults=commons.HEADER_DEFAULTS, field_delim='\t')
    features = dict(zip(commons.HEADERS, columns))
    target = features.pop(commons.LABEL_COL)
    return features, tf.string_to_number(target, out_type=tf.int32)


def input_fn(file_name, batch_size=32, shuffle=False, repeat_count=1):
    num_threads = multiprocessing.cpu_count()

    data_set = tf.data.TextLineDataset(filenames=file_name).skip(1)

    if shuffle:
        data_set = data_set.shuffle(buffer_size=1000)

    data_set = data_set.map(lambda row: parse_csv_row(row), num_parallel_calls=num_threads).batch(batch_size) \
        .repeat(repeat_count).prefetch(1000)

    iterator = data_set.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, target


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.keras.backend.set_learning_phase(True)
    else:
        tf.keras.backend.set_learning_phase(False)

    vocab_table = lookup.index_table_from_file(vocabulary_file='data/vocab.csv', num_oov_buckets=1, default_value=-1)
    text = features[commons.FEATURE_COL]
    words = tf.string_split(text)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=commons.PAD_WORD)
    word_ids = vocab_table.lookup(dense_words)

    padding = tf.constant([[0, 0], [0, commons.MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, commons.MAX_DOCUMENT_LENGTH])

    f1 = tf.keras.layers.Embedding(params.N_WORDS, 100, input_length=commons.MAX_DOCUMENT_LENGTH)(word_id_vector)
    f2 = tf.keras.layers.Embedding(params.N_WORDS, 200, input_length=commons.MAX_DOCUMENT_LENGTH)(word_id_vector)
    f3 = tf.keras.layers.Embedding(params.N_WORDS, 300, input_length=commons.MAX_DOCUMENT_LENGTH)(word_id_vector)

    filter_sizes = [3, 5]

    conv_pools = []
    for text_embedding in [f1, f2, f3]:
        for filter_size in filter_sizes:
            l_zero = tf.keras.layers.ZeroPadding1D((filter_size - 1, filter_size - 1))(text_embedding)
            l_conv = tf.keras.layers.Conv1D(filters=32, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = tf.keras.layers.GlobalMaxPool1D()(l_conv)
            conv_pools.append(l_pool)
    merged = tf.keras.layers.Concatenate(axis=1)(conv_pools)
    dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(merged)
    dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dense1)

    logits = tf.keras.layers.Dense(1, activation=None)(dense2)

    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    optimizer = tf.train.AdamOptimizer()

    def _train_op_fn(loss):
        return optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return head.create_estimator_spec(features=features, labels=labels, mode=mode, logits=logits,
                                      train_op_fn=_train_op_fn)


def serving_fn():
    receiver_tensor = {
        commons.FEATURE_COL: tf.placeholder(dtype=tf.string, shape=None)
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
