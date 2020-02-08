# -*- coding: utf-8 -*-

from __future__ import print_function

import pandas as pd
import os

DB_TRAIN_PATH = 'database/train'
DB_TEST_PATH = 'database/test'


class Database(object):

  def __init__(self, database_path, info=False):
    self.path = os.path.abspath(database_path)
    self.name = os.path.split(self.path)[-1]
    self.labels_file = os.path.join(self.path, 'labels.csv')

    self._generate_labels_file()
    self.data = pd.read_csv(self.labels_file)
    self.classes = set(self.data["cls"])

    print('Information sur la base "%s": lignes: %s, classes (%s): %s'%(
      self.name,
      len(self),
      len(self.classes),
      self.classes
    ))

  def _generate_labels_file(self):
    if os.path.exists(self.labels_file):
      return
    
    with open(self.labels_file, 'w', encoding='UTF-8') as f:
      f.write("img,cls")
      for root, _, files in os.walk(self.path, topdown=False):
        cls = root.split('/')[-1]
        for name in files:
          if not name.endswith('.jpg'):
            continue
          img = os.path.join(root, name)
          f.write("\n{},{}".format(img, cls))

  def __len__(self):
    return len(self.data)

  def get_class(self):
    return self.classes

  def get_data(self):
    return self.data


if __name__ == "__main__":
  db_train = Database(DB_TRAIN_PATH, info=True)
  db_test = Database(DB_TEST_PATH, info=True)
  
