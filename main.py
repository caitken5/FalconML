# Written here is the machine learning code to train and save the network.


# 0: Boilerplate
import os
# import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# Self declared classes.
import MLFunctions

# Retrieve versions of packages being used.
print(np.__version__)
print(tf.__version__)

# 1: Load Data
numSamples = 10  # The number of time samples included in each back of the array.
numParam = 9  # The number of included columns in the input file.

# Define the path to be passed to FalconFolder object.
d_directory = os.path.dirname(__file__)
d_folder = 'CleanedForces/v1/'
d_path = os.path.join(d_directory, d_folder)

# Create a FalconFolder object. This object will contain all file names, sizes, and associated forces.
d_folder = MLFunctions.FalconFolder(d_path)
d_folder.populate_lists()

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

