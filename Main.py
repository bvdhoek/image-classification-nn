

# Our data files are 5 training batches and a test batch
# These files are "pickled" Python objects produced with cPickle
# This method takes a file and returns a dictionary
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


dict1 = unpickle("./data/data_batch_1")
dict_test = unpickle("./data/test_batch")

data_dict1 = dict1[b'data'] / 255
labels_dict1 = dict1[b'labels']

test_data = dict_test[b'data'] / 255
test_labels = dict_test[b'labels']

# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

print(data_dict1)
print(test_data)


