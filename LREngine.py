import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

class LREngine:
    def __init__(self):
        # Load the train data from Keras
        self.fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()

        # Stores the cloth names in a list to convert their index to a string
        self.class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

        # Shows the data of the train and test images
        print("Size of train images (length, px, px): ", end = '')
        print(self.train_images.shape)
        print("Size of test images (length, px, px): ", end = '')
        print(self.test_images.shape)
        
        # Initiates other variables
        self.model = None

    def prepareImages(self):
        # Converts the images to grayscale
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def showImages(self, n):
        # Shows the first n train images along with their labels
        plt.figure(figsize=(10,10))
        for i in range(n):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap = plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i]])
        plt.show()

    def buildModel(self):
        # Sets up the layers of the model
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape = (28, 28)),
            keras.layers.Dense(128, activation = 'relu'),
            keras.layers.Dense(10, activation = 'softmax')
        ])

        # Compiles the model with a few settings before training
        self.model.compile(
                optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy']
            )

        # Trains the model
        self.model.fit(self.train_images, self.train_labels, epochs = 10)

    def getPredictions(self, images):
        # Asks the model to make predictions on an list of images
        predictions = self.model.predict(images)

        # Loops through all returned predictions
        # Gets the max value of each image prediction and puts that index into the ids list
        prediction_ids = []
        for i in range(len(predictions)):
            prediction_ids.append(np.argmax(predictions[i]))
        return prediction_ids;


engine = LREngine()
engine.prepareImages()
# engine.showImages(20)
engine.buildModel()

ids = engine.getPredictions(engine.test_images)
for i in range(len(ids)):
    print("Prediction: " + str(engine.class_names[ids[i]]) + " --  Actual: " + str(engine.class_names[engine.test_labels[i]]))
