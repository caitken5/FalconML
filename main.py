# Written here is the machine learning code to train and save the network.

# region 0: Boilerplate
import os
import numpy as np
import sklearn
import tensorflow as tf
import keras

# Import specific modules form packages.
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorboard import program

# My classes and functions.
import MLFunctions

# Retrieve versions of packages being used.
print("Numpy version: ", np.__version__)
print("Tensorflow version: ", tf.__version__)
print("sklearn version: ", sklearn.__version__)
# endregion

# region 1: Load Data and 2: Pre-Processing
column_names = ['Time', 'Pos_X', 'Pos_Y', 'Pos_Z', 'Vel_X', 'Vel_Y', 'Vel_Z', 'C_Force', 'F_Force', 'A_Force_X',
                'A_Force_Y', 'A_Force_Z']
# Define the required path names for setting up project directory management.
directory = os.path.dirname(__file__)
model_name = 'Simple_Feed_Forward'  # Declare model design number.
d_version = 'v1'
o_folder = 'CleanedForces'
d_folder = 'Data'
log_folder = 'Logs/Fit'
m_type = 'Models'

# Create ManageData object.
data_manager = MLFunctions.ManageData(directory, o_folder, d_folder, d_version, column_names)
# Check to see if the desired version of data has already been created and separated.
check = "/".join(data_manager.fPath2DDir)
if not os.path.isdir(check):
    print(d_version, " does not exist. Making required files in directory...")
    # Create numpy array and save in new directory folder.
    data_manager.populate_lists()
    x_train, x_test, x_val, y_train, y_test, y_val = data_manager.create_single_array()
# If the sub-directory already exists, then don't need to do a bunch.
else:
    data_manager.get_populated_lists()
    x_train, x_test, x_val, y_train, y_test, y_val = data_manager.load_arr()  # 6 new variables will take the data \
    # passed by the function.
    print(d_version, " exists. Loading required data...")
# Declare filename to save the new array to.
model_manager = MLFunctions.ManageFileStructure(directory, m_type, model_name)
model_manager.create_new_folder()
# Set path to where logs of events will be stored.
log_manager = MLFunctions.ManageFileStructure(directory, log_folder, model_name)
log_manager.create_new_folder()
# endregion

# region 3: Define Hyper-Parameters
num_epochs = 50
numSamples = 10  # The number of time samples included in each back of the array.
batch_size = 50
epochs = 20
learning_rate = 0.001  # Default ADAM learning rate.
beta1 = 0.9
beta2 = 0.999
optimizer_ = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
loss_ = tf.keras.losses.MeanSquaredError()  # Loss should probably be mean squared error.
# endregion

# region 4: Define the Network
# Define the layers of the model.
input_shape = 9
model = Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Dense(11, activation="relu", name="layer1"))
model.add(layers.Dense(5, activation="relu", name="layer2"))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer=optimizer_, loss=loss_)
# TODO: Finish building the model, ensure all values are chosen using hyperparameters so that they can be saved in the \
#  text file or different model files.
# Display the model and save the graph.
# endregion

# region 5: Train the Network
# model.fit(my parameters here)
# endregion

# region 6: Perform Inference
# TODO: Do this using tensorboard if possible.
tracking_address = model_manager.fullPath  # The path of the log file.
if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on ")
# endregion

# region 7: Output
# Below is example of how to save tensorflow model.
# tf.saved_model.save(model, path)
# Below is to load saved model.
# loaded = tf.saved_model.load(mobilenet_save_path)
# endregion

# TODO: In a single text file, record result from each run for easy comparison on best performance of the model. \
#  Do this separately for each model type.
# Will record prediction time as well for one sample, try this for 10 different samples and record the time for each.
# Then calculate average and record to file.
