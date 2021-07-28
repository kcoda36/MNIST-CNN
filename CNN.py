from keras.datasets import mnist
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import keras.layers as layers
from keras import models
from keras import Input
import keras
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs/", histogram_freq=1)

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Shape the tensor
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 255
X_test /= 255

# Y data catergorize encoding 0-9
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


def resLayer(in_m):
    batch = layers.normalization.BatchNormalization()(in_m)
    atv = layers.Activation(activation="swish")(batch)
    cnv = layers.Conv2D(32, (3, 3), activation="linear", padding="same", use_bias=False)(atv)
    batch = layers.normalization.BatchNormalization()(cnv)
    atv = layers.Activation(activation="swish")(batch)
    cnv = layers.Conv2D(32, (3, 3), activation="linear", padding="same", use_bias=False)(atv)
    batch = layers.normalization.BatchNormalization()(cnv)
    atv = layers.Activation(activation="swish")(batch)
    cnv = layers.Conv2D(32, (5, 5), activation="linear", padding="same", use_bias=False)(atv)
    add = layers.add([cnv, in_m])
    return add

#Model Creation
inp = Input(shape=[28,28,1])
cnv = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(inp)
atv = layers.Activation("swish", name='swish1')(cnv)
res = atv

for i in range(10):
    res = resLayer(res)
drop = layers.Dropout(0.3)(res)

flat = layers.Flatten()(drop)
d = layers.Dense(128, activation='swish')(flat)
batch = layers.BatchNormalization()(d)
drop = layers.Dropout(0.3)(batch)
out = layers.Dense(10, activation='softmax')(drop)
model = models.Model(inp, out)

#Complile Model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#Training and Validation
model.fit(X_train, Y_train, batch_size=128, epochs=30, validation_data=(X_test, Y_test), callbacks=[tensorboard_callback])

#Evaluate model
print('Evaluation: I have large fatass nuts')
model.evaluate(X_test, Y_test)

#Save Model
model.save('saved_model/my_model')

