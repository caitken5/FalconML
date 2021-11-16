# Written here is the machine learning code to train and save the network.


# 0: Boilerplate
# import os
# import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# Retrieve versions of packages being used.
print(np.__version__)
print(tf.__version__)

# 1: Load Data
numSamples = 10  # The number of time samples included in each back of the array.
numParam = 9  # The number of included columns in the input file.

# Generate labels for each sample value based on title of file.


# Create class or function here for loading in each text file.
class FalconRetriever(Sequence):
    """
    FalconRetriever: Easy generation of single batch from data including reshaped training data and labels.
    Adapted from here: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly, and here:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, num_files_in_folder, files_in_folder, c_force, path):
        """
        Initialize sequence generator of class FalconRetriever with provided values.
        :param num_files_in_folder: Number of files in the path folder.
        :param files_in_folder: Names of files in the path folder.
        :param c_force: Label data from file name.
        :param path: Path to folder where desired files are contained.
        """
        self.numFiles = num_files_in_folder
        self.fileNames = files_in_folder
        self.y = c_force
        self.path = path

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        """
        Pull file from path of the generator that's next in line in the file_names array.
        :param idx: index of next file to be pulled by sequence generator.
        :return: Numpy array of training data, and separate numpy array of label data.
        """

# 2: Pre-Processing
# IMPORTANT NOTE: Most of this should have been completed in FalconMLPrep file.

# 3: Define Hyper-parameters


# 4: Define the Network


# Define the input shape of the array.
input_shape = (9, numSamples, 1)

# Define the layers of the model.
model_1 = Sequential()
model_1.add(layers.Dense(2, activation="relu", name="layer1"))

# 5: Train the Network

# 6: Perform Inference and Output

# 7: Output
# Below is example of how to save tensorflow model.
# tf.saved_model.save(model, path)
# Below is to load saved model.
# loaded = tf.saved_model.load(mobilenet_save_path)

