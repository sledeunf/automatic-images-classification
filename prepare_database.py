import os, shutil
from random import shuffle
from constants import COREL_DB_PATH, WORKSPACE_DB

def random_classes(n):
    """
    Tire au maximum n classes aléatoires parmi celles contenues dans la base.
    """
    # liste les fichiers disponibles dans le répertoire de la base COREL
    files = os.listdir(COREL_DB_PATH)

    classes = list(filter(lambda file: (
        # conserve uniquement les classes (répertoires)
        os.path.isdir(os.path.join(COREL_DB_PATH, file))
        # ignore les repertoire caches (systeme) sous linux
        and not file.startswith('.')
    ), files))

    # melange (en place) la liste de classes
    shuffle(classes)

    # retourne les n (au plus) premieres classes
    return classes[:n]
    
  
def split_classes_data(classes):
    """
    Reparti chaque classe d'images en 3 sous-repertoires.
    Proportion: train=70% validation=15% test=15%
    """
    try:
      # supprime la base de tests precendente si elle existe
      shutil.rmtree(WORKSPACE_DB)
    except FileNotFoundError:
      pass

    # recrée un répertoire pour la base de tests
    os.mkdir(WORKSPACE_DB)

    # crée 3 répertoires pour répartir nos classes
    folders = ('train', 'validation', 'test')
    for folder in folders:
      os.mkdir(os.path.join(WORKSPACE_DB, folder))
    
    # pour chacune des classes choisies
    for classe in classes:
      # recupere les images disponibles dans la base COREL
      images = os.listdir(os.path.join(COREL_DB_PATH, classe))
      shuffle(images)
      # calcule en fonction du nombre d'image les indices des slices
      n = len(images)
      indexes = [0, int(0.7*n), int(0.85*n), n]
      
      for k in range(3):
        # cree le repertoire de la classe dans chaque sous-repertoire
        os.mkdir(os.path.join(WORKSPACE_DB, folders[k], classe))
        # déplace les images de la base COREL vers la base de travail
        for image in images[indexes[k]: indexes[k+1]]:
          shutil.copy(
              os.path.join(COREL_DB_PATH, classe, image),
              os.path.join(WORKSPACE_DB, folders[k], classe, image)
          )

if __name__ == '__main__':
    # genere une base de donnee de travail contenant 2 classes aleatoirement
    #classes = random_classes(5)
    split_classes_data(['woman', 'pet_dog', 'pl_flower', 'obj_car'])
