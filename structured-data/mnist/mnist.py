import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import keras.backend as K
from keras import layers

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 100
NUM_STEPS = 1000


def cnn_model(features, labels, mode):
    input_layer = tf.reshape(features['x'], shape=[-1, 28, 28, 1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        K.set_learning_phase(1)
    else:
        K.set_learning_phase(0)

    conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1),
                          activation='relu')(input_layer)
    conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPool2D(pool_size=(2, 2))(conv2)
    dropout = layers.Dropout(0.5)(pool1)

    conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(dropout)
    conv4 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    pool2 = layers.MaxPool2D(pool_size=(3, 3))(conv4)
    dropout2 = layers.Dropout(0.5)(pool2)

    flatten = layers.Flatten()(dropout2)

    dense1 = layers.Dense(256)(flatten)
    lrelu = layers.LeakyReLU()(dense1)
    dropout3 = layers.Dropout(0.5)(lrelu)
    dense2 = layers.Dense(256)(dropout3)
    lrelu2 = layers.LeakyReLU()(dense2)
    logits = layers.Dense(10, activation='linear')(lrelu2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits=logits, name='softmax_tensor')
    }

    prediction_output = tf.estimator.export.PredictOutput({"classes": tf.argmax(input=logits, axis=1),
     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")})

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output
        })

    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)

    tf.summary.scalar('loss', loss)
    tf.summary.histogram('conv1', conv1)
    tf.summary.histogram('dense', dense1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


def generate_input_fn(data_set, batch_size=BATCH_SIZE):
    def _input_fn():
        X = tf.constant(data_set.images)
        Y = tf.constant(data_set.labels, dtype=tf.int32)
        image_batch, label_batch = tf.train.shuffle_batch([X, Y],
                                                          batch_size=batch_size,
                                                          capacity=8 * batch_size,
                                                          min_after_dequeue=4 * batch_size,
                                                          enqueue_many=True)
        return {'x': image_batch}, label_batch

    return _input_fn


mnist = input_data.read_data_sets(train_dir='data/')
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

predict_data_batch = mnist.test.next_batch(10)

mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir='build/')

probabilty_log = {"probabilities": "softmax_tensor"}
train_logging_hook = tf.train.LoggingTensorHook(tensors=probabilty_log, every_n_iter=1000)

mnist_classifier.train(input_fn=generate_input_fn(mnist.train, batch_size=BATCH_SIZE), steps=NUM_STEPS,
                       hooks=[train_logging_hook])

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": predict_data_batch[0]}, y=None, num_epochs=1,
                                                      shuffle=False)
predict_results = mnist_classifier.predict(input_fn=predict_input_fn)

for i, p in enumerate(predict_results):
    print("Correct label: %s" % predict_data_batch[1][i])
    print("Prediction: %s" % p)


def serving_input_receiver_fn():
    feature_tensor = tf.placeholder(tf.float32, [None, 784])
    return tf.estimator.export.ServingInputReceiver({'x': feature_tensor}, {'x': feature_tensor})


exported_model_dir = mnist_classifier.export_savedmodel('build/', serving_input_receiver_fn)
decoded_model_dir = exported_model_dir.decode("utf-8")
