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


def resLayer(inp, size):
    cnv = layers.Conv2D(size, (3, 3), activation="linear", padding="same")(inp)
    batch = layers.normalization.BatchNormalization()(cnv)
    atv = layers.Activation(activation="swish")(batch)
    add = layers.add([atv, inp])
    return add

#Model Creation
inp = Input(shape=[28,28,1])

res = resLayer(inp, 64)
for i in range(2):
    res = resLayer(res, 64)

cnv = layers.Conv2D(112, (3, 3), activation="linear", padding="same")(res)
batch = layers.normalization.BatchNormalization()(cnv)
res = layers.Activation(activation="swish")(batch)
for i in range(5):
    res = resLayer(res, 112)

cnv = layers.Conv2D(160, (3, 3), activation="linear", padding="same")(res)
batch = layers.normalization.BatchNormalization()(cnv)
res = layers.Activation(activation="swish")(batch)
for i in range(8):
    res = resLayer(res, 160)

drop = layers.Dropout(0.2)(res)
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

