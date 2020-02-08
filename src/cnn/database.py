
import os
import shutil
from random import randint, shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Database:
    """
    Helper to create and interract with database.
    """
    def __init__(self, database_path):
        self.path = database_path
    
    @classmethod
    def random_classes(cls, from_folder):
        """
        Return a set of 2-8 classes randomly picked in <from_folder>.
        """
        n = randint(2, 8)
        all_classes = os.listdir(from_folder)
        shuffle(all_classes)
        return all_classes[:n]
        

    @classmethod
    def create(cls, database_name, from_folder, classes=None, ratios=(0.7, 0.15, 0.15)):
        """
        Create a new database with test, validation and train subfolder.

        Parameters:
            - database_name: name of the folder to be created.
            - from_folder: another folder containing images labelized in subfolders.
            - ratios: proportion of respectively train, validation and test subsets.
            - classes: list of classes that will be extracted from original folder
                if not specified, a set of 2-8 classes will be randomly picked.
        """
        assert sum(ratios) == 1, "Sum of ratios must be equal to 1!"

        # drop possibly existing database with the same name
        shutil.rmtree(database_name, ignore_errors=True)

        # if not classes are passed, select some randomly
        if not classes:
            classes = cls.random_classes(from_folder)

        # create database arborescence
        for folder in ('train', 'validation', 'test'):
            for classe in classes:
                os.makedirs( os.path.join(database_name, folder, classe) )

               


    @property
    def classes(self):
        """
        This method act like an attribute and return a list of classes in database.
        """
        return sorted(os.listdir(os.path.join(self.path, 'train')))


    @property
    def weights_filename(self):
        """
        return weights filename depending on classes in database.
        """
        return '__'.join(self.classes) + '.h5'

    @property
    def weights_exists(self):
        """
        Return true if a weights file exists (model has already been trained)
        """
        return os.path.isfile(self.weights_filename)

    def get_images_generator(self, image_subfolder, shuffle=True, batch_size=16, target_size=(150, 150)):
        """
        Return a generator that will yield an infinite number of images.

        Parameters:
            - image_subfolder: Subfolder of the database where images will be picked.
            - shuffle: If true, randomize images order.
            - target_size: A tuple (width, length) used to resize yielded images.
            - batch_size: Number of images per chuncks.
        """
        return ImageDataGenerator(
            rescale=1./255
        ).flow_from_directory(
            os.path.join(self.path, image_subfolder),
            target_size=target_size,
            shuffle=shuffle,
            batch_size=batch_size,
            class_mode='sparse'
        )

    def __len__(self):
        return len(self.classes)


if __name__ == '__main__':
    db = Database.create('database2', 'coreldb')
    #generator = db.get_images_generator('train')
