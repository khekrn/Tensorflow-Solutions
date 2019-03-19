import multiprocessing

import tensorflow as tf

from model import commons


def parse_record(record):
    columns = tf.decode_csv(record, record_defaults=commons.HEADER_DEFAULTS, field_delim='\t')
    features = columns[0]
    target = columns[1:]
    target = tf.cast(tf.string_to_number(target), dtype=tf.int32)
    target = tf.stack(target, axis=0)
    return {commons.FEATURE_COL: features}, target


def input_fn(file_name, shuffle=False, batch_size=16, repeat_count=1):
    n_cpu = multiprocessing.cpu_count()
    data_set = tf.data.TextLineDataset(filenames=file_name).skip(1)

    if shuffle:
        data_set = data_set.shuffle(shuffle)

    data_set = data_set.map(lambda record: parse_record(record), num_parallel_calls=n_cpu) \
        .batch(batch_size=batch_size) \
        .repeat(repeat_count)

    iterator = data_set.make_one_shot_iterator()
    features, label = iterator.get_next()
    return features, label


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
    filenames='D:/Git/Tensorflow/Tensorflow-Solutions/Problems-Solutions/text/ToxicComments_MultiLabel/data/train_preprocess.csv')\
    .skip(1)
data_set = data_set.shuffle(True)
data_set = data_set.map(lambda record: parse_record(record), num_parallel_calls=4).repeat(1).batch(batch_size=1)

iterator = data_set.make_initializable_iterator()

sess = tf.Session()

sess.run(iterator.initializer)

X, Y = iterator.get_next()

print(sess.run(X))
print('\n\n')
print(sess.run(Y))
sess.close()
'''
