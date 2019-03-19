import time

import tensorflow as tf

print('Tensorflow Version - ', tf.__version__)  # Tensorflow 1.3
tf.logging.set_verbosity(tf.logging.INFO)

train_file = 'data/iris_training.csv'
test_file = 'data/iris_test.csv'

feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth'
]


def input_fn(file, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:]
        del parsed_line[-1]
        features = parsed_line
        parsed_data = dict(zip(feature_names, features)), label
        return parsed_data

    data_set = (tf.contrib.data.TextLineDataset(file).skip(1).map(decode_csv))

    if perform_shuffle:
        data_set = data_set.shuffle(buffer_size=256)

    data_set = data_set.repeat(repeat_count)
    data_set = data_set.batch(32)
    iterator = data_set.make_one_shot_iterator()
    batch_features, batch_label = iterator.get_next()
    return batch_features, batch_label


feature_columns = [tf.feature_column.numeric_column(feature) for feature in feature_names]


def iris_serving_input_fn():
    """Build the serving inputs."""

    inputs = {}
    for feat in feature_columns:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }
    return tf.contrib.learn.InputFnOps(features, None, inputs)


run_config = tf.estimator.RunConfig()
run_config.replace(save_checkpoints_secs=1)

classifier = tf.estimator.DNNClassifier(hidden_units=[100, 70, 20, 12], feature_columns=feature_columns, n_classes=3,
                                        model_dir='build/', config=run_config, dropout=0.4,
                                        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                                                                    l1_regularization_strength=0.01),
                                        activation_fn=tf.nn.elu)

classifier.train(input_fn=lambda: input_fn(train_file, perform_shuffle=True, repeat_count=30))

evaluation_results = classifier.evaluate(input_fn=lambda: input_fn(test_file, perform_shuffle=False, repeat_count=1))

for key in evaluation_results:
    print(" {} was {}".format(key, evaluation_results[key]))

time.sleep(5)
print('\n\n Exporting Iris Model')


def serving_input_receiver_fn():
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


classifier.export_savedmodel(export_dir_base='build/', serving_input_receiver_fn=serving_input_receiver_fn)