import multiprocessing

import tensorflow as tf

from utils import commons

__author__ = 'KKishore'


def parse_csv_row(row):
    columns = tf.decode_csv(row, record_defaults=commons.HEADER_DEFAULTS)
    features = dict(zip(commons.HEADERS, columns))
    target = features.pop(commons.LABEL_COL)
    return features, target


def input_fn(file_name, batch_size=16, shuffle=False, repeat_count=1):
    num_threads = multiprocessing.cpu_count()

    data_set = tf.data.TextLineDataset(filenames=file_name).skip(1)

    if shuffle:
        data_set = data_set.shuffle(buffer_size=1000)

    data_set = data_set.map(lambda row: parse_csv_row(row), num_parallel_calls=num_threads).batch(batch_size) \
        .repeat(repeat_count).prefetch(1000)

    iterator = data_set.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, tf.string_to_number(target, out_type=tf.int64)


def serving_fn():
    receiver_tensor = {
        commons.FEATURE_COL: tf.placeholder(dtype=tf.string, shape=None)
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
