
import os
import shutil
from random import randint, random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Database:
    """
    Helper to create and interract with database.
    """
    def __init__(self, database_path):
        self.path = database_path

    @classmethod
    def _generate_labels_file(cls, database_path, subfolder):
        """
        Create a CSV file wich associate image path and its class.

        Parameters:
            - subfolder: string in (train, test, validation)
        """
        database_subfolder = os.path.join(database_path, subfolder)
        labels_file = os.path.join(database_path, subfolder + '_labels.csv')

        if os.path.exists(labels_file):
            return
        
        with open(labels_file, 'w', encoding='UTF-8') as f:
            f.write("img,cls")
            for root, _, files in os.walk(database_subfolder, topdown=False):
                classe = os.path.split(root)[-1]

                for name in files:
                    if name[:-4] in ('.jpg', '.png'):
                        continue
                
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, classe))
    
    @classmethod
    def random_classes(cls, from_folder):
        """
        Return a set of 2-8 classes randomly picked in <from_folder>.
        """
        n = randint(2, 8)
        # recupere la liste de toutes les classes dans un ordre aleatoire
        all_classes = sorted(os.listdir(from_folder), key=lambda x: random())
        # renvoie les n premiers resultats
        return all_classes[:n]
        

    @classmethod
    def create(cls, database_name, from_folder, classes=None, ratios=(0.7, 0.15, 0.15), csv_labels=True):
        """
        Create a new database with train, validation and test subfolders.

        Parameters:
            - database_name: name of the folder to be created.
            - from_folder: another folder containing images labelized in subfolders.
            - ratios: proportion of respectively train, validation and test subsets.
            - classes: list of classes that will be extracted from original folder
                if not specified, a set of 2-8 classes will be randomly picked.
        """
        assert sum(ratios) == 1, "Sum of ratios must be equal to 1!"

        # supprime une base de données de même nom qui pourrait exister
        shutil.rmtree(database_name, ignore_errors=True)

        # si aucune classe n'est passée en paramètre, en choisi aléatoirement
        if not classes:
            classes = cls.random_classes(from_folder)
        
        for classe in classes:
            origin_class_path = os.path.join(from_folder, classe)
            # recupere la liste des images de cette classe dans un ordre aleatoire
            class_images = sorted(os.listdir(origin_class_path), key=lambda x: random())
            # 
            n = len(class_images)
            slices = (0, int(n*ratios[0]), int(n*(ratios[0]+ratios[1])), n)

            subfolders = ('train', 'validation', 'test')
            for k in range(3):
                # crée au fur et a mesure l'arborescence de la nouvelle base de données
                dest_folder = os.path.join(database_name, subfolders[k], classe)
                os.makedirs(dest_folder)
                # copie un sous ensemble des images dans le repertoire
                for image in class_images[slices[k]: slices[k+1]]:
                    shutil.copy(
                        os.path.join(origin_class_path, image),
                        os.path.join(dest_folder, image)
                    )

        if csv_labels:
            cls._generate_labels_file(database_name, 'train')
            cls._generate_labels_file(database_name, 'validation')
            cls._generate_labels_file(database_name, 'test')


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
        return os.path.join(self.path, 'model_weights.h5')

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
    db = Database.create('database', from_folder='coreldb', classes=['pet_dog'])
    #generator = db.get_images_generator('train')
