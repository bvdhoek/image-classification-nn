

# Our data files are 5 training batches and a test batch
# These files are "pickled" Python objects produced with cPickle
# This method takes a file and returns a dictionary
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

dict1 = unpickle("./data/data_batch_1")
dict_test = unpickle("./data/test_batch")

data_dict1 = dict1[b'data'] / 255
labels_dict1 = dict1[b'labels']

test_data = dict_test[b'data'] / 255
test_labels = dict_test[b'labels']

# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

#print(data_dict1)
#print(labels_dict1)

def data(list):
  data = []
  for j in range(len(list)):
    row = []
    img = list[j]
    for i in range(1024):
      row.append([img[i], img[i + 1023], img[i + 2047]])
    data.append(np.reshape(row, (32, 32, -1)))
  return data

imgs = data(data_dict1)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imgs[i])
    plt.xlabel(class_names[labels_dict1[i]])
plt.show()


