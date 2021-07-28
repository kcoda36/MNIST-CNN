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


def resLayer(in_m, size):
    cnv = layers.Conv2D(size, (3, 3), activation="linear", padding="same")(in_m)
    batch = layers.normalization.BatchNormalization()(cnv)
    atv = layers.Activation(activation="swish")(batch)
    cnv = layers.Conv2D(size, (3, 3), activation="linear", padding="same")(atv)
    batch = layers.normalization.BatchNormalization()(cnv)
    atv = layers.Activation(activation="swish")(batch)
    cnv = layers.Conv2D(size, (5, 5), activation="linear", padding="same")(atv)
    batch = layers.normalization.BatchNormalization()(cnv)
    atv = layers.Activation(activation="swish")(batch)
    add = layers.add([atv, in_m])
    return add

#Model Creation
inp = Input(shape=[28,28,1])

res = resLayer(inp, 32)
for i in range(1):
    res = resLayer(res, 32)
drop = layers.Dropout(0.3)(res)

res = resLayer(drop, 32)
for i in range(1):
    res = resLayer(res, 32)
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
model.fit(X_train, Y_train, batch_size=128, epochs=50, validation_data=(X_test, Y_test), callbacks=[tensorboard_callback])

#Evaluate model
print('Evaluation: I have large fatass nuts')
model.evaluate(X_test, Y_test)

#Save Model
model.save('saved_model/my_model')

