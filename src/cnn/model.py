import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from tensorflow import argmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from database import Database


class CNNClassifier(Sequential):
    """
    Simple convolutional network classifier with methods to interract
    with a local image database.
    """

    def __init__(self, n_output, target_size=(150, 150)):
        super(Sequential, self).__init__()
        self.target_size = target_size

        self.add(Conv2D(32, (3, 3), input_shape=(*target_size, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(32, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(64, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)
        self.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.add(Dense(64))
        self.add(Activation('relu'))
        self.add(Dropout(0.5))
        self.add(Dense(n_output))
        self.add(Activation('softmax'))

        self.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def plot_history(self, history):
        for key in history.history.keys():
            plt.plot(history.history[key])
        
        plt.ylabel('accuracy or loss')
        plt.xlabel('epoch')
        plt.legend(history.history.keys(), loc='upper left')
        plt.show()

    def train(self, database, batch_size=16, epochs=15, history=False, overwrite=False):
        """
        Parameters:
            - ...
            - overwrite: allow to overwrite existing training data.
            - history: if set to true, plot the evolution of metrics over epochs.
        """

        # if weights exists and user don't want to overwrite
        # just load previously generated model weights
        if database.weights_exists and not overwrite:
            self.load_weights(database.weights_filename)
            print('Weights already existing, skipping training step.')
            return
        
        train_images = database.get_images_generator('train')
        validation_images = database.get_images_generator('validation')

        train_history = self.fit( # perform model training
            train_images,
            epochs=epochs,
            validation_data=validation_images
        )

        if save:
            self.save_weights(database.weights_filename)

        if history:
            self.plot_history(train_history)
        

    def classify_test_images(self, database, confusion_matrix=False, separate_miss=False):
        """
        Use trained model to predict classes of images in test/ folder.
        Output theses images in a folder called predictions/
        Parameter:
            - database: neural_network.Database object.
            - separate_miss: if true, create a miss folder to store failed predictions.
            - confusion_matrix: if true, plot a confusion matrix showing classfication accuracy.
        """

        # load test images and don't shuffle it to be able associate results
        # to original filenames.
        test_images = database.get_images_generator('test', shuffle=False)

        # predict classes of images in test generator and 
        # output results as a numpy array.
        predictions = argmax(self.predict(test_images), axis=1).numpy()

        if confusion_matrix:
            self.confusion_matrix(test_images.classes, predictions, labels=database.classes)
        
        
    def confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Plot confusion matrix using matplotlib
        """
        cmatrix = confusion_matrix(y_true, y_pred, normalize='true')
        display = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=labels)
        display.plot()
        plt.show()



if __name__ == '__main__':
    database = Database('database')
    print(database.classes)

    model = CNNClassifier(len(database))
    model.train(database, epochs=8, history=True, overwrite=False)
    model.classify_test_images(database, confusion_matrix=True)
