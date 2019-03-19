from keras.layers import Dense, ZeroPadding1D
from keras.layers import Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

import numpy as np

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
epochs = 2


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train =pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')


text_seq_input = Input(shape=(maxlen,), dtype='int32')
text_embedding1 = Embedding(max_features, 100, input_length=maxlen, trainable=True)(text_seq_input)
text_embedding2 = Embedding(max_features, 200, input_length=maxlen, trainable=True)(text_seq_input)
text_embedding3 = Embedding(max_features, 300, input_length=maxlen, trainable=True)(text_seq_input)

#k_top = 4
filter_sizes = [3,5]

conv_pools = []
for text_embedding in [text_embedding1, text_embedding2, text_embedding3]:
    for filter_size in filter_sizes:
        l_zero = ZeroPadding1D((filter_size-1,filter_size-1))(text_embedding)
        l_conv = Conv1D(filters=16, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
        l_pool = GlobalMaxPool1D()(l_conv)
        conv_pools.append(l_pool)

l_merge = Concatenate(axis=1)(conv_pools)
l_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(l_merge)
l_dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(l_dense)
l_out = Dense(1, activation='sigmoid')(l_dense)
model_1 = Model(inputs=[text_seq_input], outputs=l_out)

print(model_1.summary())

print('Compiling Model')
model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Training started .........')
model_1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))