import numpy as np
import tensorflow as tf

from model.folding import Folding
from model.pooling import KMaxPooling

tf.logging.set_verbosity(tf.logging.INFO)

print(tf.__version__)

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen, name='input'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.ZeroPadding1D((49, 49)))
model.add(tf.keras.layers.Conv1D(64, 50, padding="same"))
model.add(KMaxPooling(k=10, axis=1))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.ZeroPadding1D((25, 25)))
model.add(tf.keras.layers.Conv1D(64, 25, padding="same"))
model.add(Folding())
model.add(KMaxPooling(k=10, axis=1))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print("\n", model.summary())
print("\n")

print("Compiling Model \n")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='build/')

y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_train, (len(y_test), 1))

print(np.shape(x_train), 'train Label')
print(np.shape(x_test), 'test Label')

estimator.train(input_fn=tf.estimator.inputs.numpy_input_fn(x={"input_input": x_train},
                                                            y=y_train, num_epochs=10, batch_size=batch_size,
                                                             shuffle=True))

res = estimator.evaluate(input_fn=tf.estimator.inputs.numpy_input_fn(x={"input_input": x_test},
                                                            y=y_test, num_epochs=1, batch_size=batch_size,
                                                             shuffle=False))
print('{}'.format(res))
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
