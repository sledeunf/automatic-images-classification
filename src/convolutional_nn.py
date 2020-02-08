import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from tensorflow import argmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from .database import Database


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
        """
        Plot metrics history in function of epochs after training.
        """
        for key in history.history.keys():
            plt.plot(history.history[key])
        
        plt.ylabel('accuracy or loss')
        plt.xlabel('epoch')
        plt.legend(history.history.keys(), loc='upper left')
        plt.show()

    def train(self, database, batch_size=16, epochs=15, history=False, overwrite=False):
        """
        Train model on test subfolder of database

        Parameters:
            - database: database object defined in this library.
            - overwrite: allow to overwrite existing training data.
            - history: if set to true, plot the evolution of metrics over epochs.
        """

        # si le modele a deja ete entraine, charge les poids existants
        # sauf si l'utilisateur demande explicitement de l'ecraser
        if database.weights_exists and not overwrite:
            self.load_weights(database.weights_filename)
            print('Weights already existing, skipping training step.')
            return
        
        train_images = database.get_images_generator('train')
        validation_images = database.get_images_generator('validation')

        # entraine le modele en utilisant les images de train et validation
        train_history = self.fit(
            train_images,
            epochs=epochs,
            validation_data=validation_images
        )

        # sauvegarde les poids du modele pour pouvoir les reutiliser plus tard
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

        # charge la liste des images dans le jeu de test et leurs labels
        test_images = database.get_images_generator('test', shuffle=False)

        # predit les labels des images du jeu de test
        predictions = argmax(self.predict(test_images), axis=1).numpy()

        # si demande, affiche la matrice de confusion des predictions
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
