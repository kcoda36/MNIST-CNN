from keras import models
import cv2
import os
import numpy as np

# Load The Model
model = models.load_model('saved_model/my_model')

# Loop through images in the numbers folder
for image_path in os.listdir('./numbers'):
    img = cv2.imread(str("numbers/" + image_path),0)

    X_test = np.array([img])
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32')
    X_test /= 255

    # Predict the output for this image
    y = model.predict(X_test)

    # Print the most likely output
    print('Number detected: ', np.argmax(y))