import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os

import MNIST_format_extractor as gze

class LREngine:
    model = None

    def train_model():
        # Will check to see if trained model already exists
        if os.path.exists('LR_Model.h5'):
            LREngine.model = keras.models.load_model('LR_Model.h5')
            print("Loaded Model from saved file.")
            return
        else:
            print("There is no saved Model! Training new model...")

        # Extract and load the train data from the EMNIST dataset
        with tf.io.gfile.GFile(os.path.expanduser(
            "~/.keras/datasets/EMNIST/emnist-byclass-train-images-idx3-ubyte.gz"), "rb") as f:
            train_images = gze.extract_images(f)

        with tf.io.gfile.GFile(os.path.expanduser(
            "~/.keras/datasets/EMNIST/emnist-byclass-train-labels-idx1-ubyte.gz"), "rb") as f:
            train_labels = gze.extract_labels(f)

        # Dictionary converts the label id to the ASCII id
        class_names = []

        # Display the details of the train data
        print("Train Images (amount, width, height): ", end = '')
        print(train_images.shape)
        print("Train Labels (amount): ", end = '')
        print(len(train_labels))

        # Sets up the layers of the model
        LREngine.model = keras.Sequential([
            keras.layers.Flatten(input_shape = (28, 28)),
            keras.layers.Dense(128, activation = 'relu'),
            keras.layers.Dense(62, activation = 'softmax')
        ])

        # Compiles the model with a few settings before training
        LREngine.model.compile(
                optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy']
            )

        # Trains the model
        LREngine.model.fit(train_images, train_labels, epochs = 10)

        # Saves the model to the dir
        LREngine.model.save('LR_Model.h5')

    def test_model():
        # Check if the model has been trained
        if (LREngine.model == None):
            print('The LR Model has not been trained! Train it first before testing or making predictions')
            return
        print("Testing Model...")

        # Extract and load the test data from the EMNIST dataset
        with tf.io.gfile.GFile(os.path.expanduser(
            "~/.keras/datasets/EMNIST/emnist-byclass-test-images-idx3-ubyte.gz"), "rb") as f:
            test_images = gze.extract_images(f)

        with tf.io.gfile.GFile(os.path.expanduser(
            "~/.keras/datasets/EMNIST/emnist-byclass-test-labels-idx1-ubyte.gz"), "rb") as f:
            test_labels = gze.extract_labels(f)
        
        # Gets and prints the predictions from the model
        prediction_indexes = LREngine.getPredictions(test_images)
        for i in range(len(prediction_indexes)):
            print("Prediction: " + str(prediction_indexes[i]) + "\t\tActual: " + str(test_labels[i]))

    def showImages(images, labels, n=1):
        # Shows the first n images along with their labels
        plt.figure(figsize=(10,10))
        for i in range(n):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap = plt.cm.binary)
            plt.xlabel(class_names[labels[i]])
        plt.show()

    def getPredictions(images):
        # Asks the model to make predictions on an list of images
        predictions = LREngine.model.predict(images)

        # Loops through all returned predictions
        # Gets the max value (most confident) of each image prediction 
        # and puts that index into the indexes list
        prediction_class_indexes = []
        for i in range(len(predictions)):
            prediction_class_indexes.append(np.argmax(predictions[i]))
        return prediction_class_indexes;


LREngine.train_model()
LREngine.test_model()

