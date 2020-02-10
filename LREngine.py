import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os

import MNIST_format_extractor as gze

class LREngine:
    # Model holds the trained CNN
    model = None
    
    # Dictionary converts the label id to the ASCII id
    class_names = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 
            19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 
            28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 
            37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 
            46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 
            55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z' }


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
            print("Prediction: " + str(LREngine.class_names[prediction_indexes[i]]) + "\t\tActual: " + str(LREngine.class_names[test_labels[i]]))

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

