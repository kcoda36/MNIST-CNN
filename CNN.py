from keras.datasets import mnist
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import keras.layers as layers
from keras import models
from keras import Input

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

#Model Creation
inp = Input(shape=[28,28,1])
cnv = Conv2D(25, (3, 3), strides=1, padding='valid', name='conv1')(inp)
atv = layers.Activation("relu", name='relu1')(cnv)
pool = MaxPool2D(pool_size=1)(atv)
flat = layers.Flatten()(pool)
d = layers.Dense(100, activation='relu')(flat)
out = layers.Dense(10, activation='softmax')(d)
model = models.Model(inp, out)

#Complile Model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#Training and Validation
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

#Save Model
model.save('saved_model/my_model')

