# Written here is the machine learning code to train and save the network.

# 0: Boilerplate
import os

import keras
import numpy as np
import tensorflow as tf
import sklearn
# Import specific modules form packages.
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# My classes and functions.
import MLFunctions

# Retrieve versions of packages being used.
print("Numpy version: ", np.__version__)
print("Tensorflow version: ", tf.__version__)
print("sklearn version: ", sklearn.__version__)

# 1: Load Data
numParam = 9  # The number of included columns in the input file.
# Declare model design number.
model_name = 'Simple_Feed_Forward'
column_names = ['Time', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vel_X', 'Vel_Y', 'Vel_Z', 'C_Force', 'F_Force', 'A_Force_X',
                'A_Force_Y', 'A_Force_Z']
y_raw_index = column_names.index('C_Force')
# Define the path to be passed to FalconFolder object.
d_directory = os.path.dirname(__file__)
d_models = 'Models/'
d_folder = 'CleanedForces/v1/'
d_path = os.path.join(d_directory, d_folder)
d_path_models = os.path.join(d_directory, d_models)

# Create a FalconFolder object. This object will contain all file names, sizes, and associated forces.
d_folder = MLFunctions.FalconFolder(d_path, column_names)
d_folder.populate_lists()
# Declare filename to save the new array to.
file_structure = MLFunctions.ManageFileStructure(d_path_models, model_name)
file_structure.create_new_folder()
# Model_1: Create single array out of all the filenames.
raw_data = d_folder.create_single_array(file_structure.newModelPath, file_structure.newModelVersion)

# 2: Pre-Processing
# IMPORTANT NOTE: Most of this should have been completed in FalconMLPrep file.
# Split raw data into x and y sets, where y is the label data c_force.
y_data = raw_data[:, y_raw_index]
x_data = np.delete(arr=raw_data, obj=y_raw_index, axis=1)
x_data_names = np.delete(arr=column_names, obj=y_raw_index)
# For Simple_Feed_Forward model, stratify according to the number of samples for each force value.
x_tav, x_test, y_tav, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
x_train, x_val, y_train, y_val = train_test_split(x_tav, y_tav, test_size=0.2, random_state=42, stratify=y_tav)
# Convert all required arrays to tensors.
x_train = tf.convert_to_tensor(np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)))
x_val = tf.convert_to_tensor(np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1)))
x_test = tf.convert_to_tensor(np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)))
y_train = tf.convert_to_tensor(np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1)))
y_val = tf.convert_to_tensor(np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1)))
y_test = tf.convert_to_tensor(np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1)))

# 3: Define Hyper-parameters
num_epochs = 50
numSamples = 10  # The number of time samples included in each back of the array.
batch_size = 50
epochs = 20
learning_rate = 0.001  # Default ADAM learning rate.
beta1 = 0.9
beta2 = 0.999
optimizer_ = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
loss_ = tf.keras.losses.MeanSquaredError()  # Loss should probably be mean squared error.

# 4: Define the Network
# Define the input shape of the array.
input_shape = (len(x_data_names), 1)

# Define the layers of the model.
model = Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Dense(11, activation="relu", name="layer1"))
model.add(layers.Dense(5, activation="relu", name="layer2"))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer=optimizer_, loss=loss_)
# TODO: Finish building the model, ensure all values are chosen using hyperparameters so that they can be saved in the \
#  text file or different model files.
# Display the model and save the graph.

# 5: Train the Network
# model.fit(my parameters here)
# 6: Perform Inference and Output
# TODO: Do this using tensorboard if possible.
# 7: Output
# Below is example of how to save tensorflow model.
# tf.saved_model.save(model, path)
# Below is to load saved model.
# loaded = tf.saved_model.load(mobilenet_save_path)
