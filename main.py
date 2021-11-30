# Written here is the machine learning code to train and save the network.

# region 0: Boilerplate
import os

# May need to: import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import keras

# Import specific modules form packages.
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorboard import program
from contextlib import redirect_stdout

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
hype_name = "hyper_parameters.txt"

# Create ManageData object.
data_manager = MLFunctions.ManageData(directory, o_folder, d_folder, d_version, column_names)
# Check to see if the desired version of data has already been created and separated.
check = "/".join(data_manager.fPath2DDir)
if not os.path.isdir(check):
    print(d_version, " does not exist. Making required files in directory...")
    # Create numpy array and save in new directory folder.
    data_manager.populate_lists()
    # Check returned dtype of y_train, y_test, and y_val.
    x_train, x_test, x_val, y_train, y_test, y_val = data_manager.create_single_array()
# If the sub-directory already exists, then don't need to do a bunch.
else:
    data_manager.get_populated_lists()
    x_train, x_test, x_val, y_train, y_test, y_val = data_manager.load_arr()  # 6 new variables will take the data \
    # passed by the function.
    print(d_version, " exists. Loading required data...")
# Convert the variables to tensors. Make sure they are also float64 rather than double (equivalents).
x_train = tf.convert_to_tensor(np.expand_dims(x_train, axis=2), dtype=tf.float64)
x_test = tf.convert_to_tensor(np.expand_dims(x_test, axis=2), dtype=tf.float64)
x_val = tf.convert_to_tensor(np.expand_dims(x_val, axis=2), dtype=tf.float64)
# Declare filename to save the new array to.
model_manager = MLFunctions.ManageFileStructure(directory, m_type, model_name)
model_manager.create_new_folder()
# Set path to where logs of events will be stored.
log_manager = MLFunctions.ManageFileStructure(directory, log_folder, model_name)
log_manager.create_new_folder()
# endregion

# region 3: Define Hyper-Parameters
numSamples = 10  # The number of time samples included in each back of the array.
batch_size = 64
epochs_ = 5
act_1 = 'relu'
act_2 = 'linear'
learning_rate = 0.001  # Default ADAM learning rate.
beta1 = 0.9
beta2 = 0.999
optimizer_ = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
loss_ = tf.keras.losses.MeanSquaredError()  # Loss can be mean squared error.
metrics_ = [tf.keras.metrics.RootMeanSquaredError()]  # A valid metric that takes the root of the mean squared error \
# so that outlier data is penalized more, as a lot of outlier data is probably caused by user interaction.
# Save hyper-parameters to model folder.
hype_columns = ['Batch Size', 'Number of Epochs', 'Activation Function 1', 'Activation Function 2',
                'Optimizer Function', 'Optimizer Learning Rate', 'Optimizer Beta 1', 'Optimizer Beta 2',
                'Loss Function', 'Metrics']
hyper_parameters = [batch_size, epochs_, act_1, act_2, optimizer_, learning_rate, beta1, beta2, loss_, metrics_]
# Define pandas array to save the hyper-parameters to.
hyper_parameters = pd.DataFrame(hyper_parameters, index=hype_columns)
hype_folder = "/".join(model_manager.fPath2CurrDir)
hype_file = "_".join((model_manager.fPath2CurrDir[-1], hype_name))
hype_folder = "/".join((hype_folder, hype_file))
hyper_parameters.to_csv(hype_folder)  # Default write mode is string.
# endregion

# region 4: Define the Network
# Define the layers of the model.
input_shape = x_train.get_shape().as_list()[1:]
# Make model using Functional API to allow naming of model.
# TODO: Figure out how I can use Normalization in the model.
inputs_ = tf.keras.Input(shape=input_shape)
layer = tf.keras.layers.Dense(11, activation=act_1)(inputs_)
layer = tf.keras.layers.Dense(5, activation=act_2)(layer)
outputs_ = tf.keras.layers.Dense(1)(layer)
tracking_address_file = "_".join((model_manager.fPath2CurrDir[-1], "model.txt"))
model = keras.Model(inputs=inputs_, outputs=outputs_, name=tracking_address_file)
model.compile(optimizer=optimizer_, loss=loss_, metrics=metrics_)
tracking_address_mod = "/".join(model_manager.fPath2CurrDir)  # The path to the model file.
tracking_address_mod = "/".join((tracking_address_mod, tracking_address_file))
# Try saving model summary.
with open(tracking_address_mod, 'w') as f:
    with redirect_stdout(f):
        model.summary()
# tf.keras.utils.plot_model(model, to_file="/".join(model_manager.fPath2CurrDir), show_shapes=True)
# endregion

# region 5: Train the Network
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_, validation_data=(x_val, y_val))
# Let the fit function equal 'history' to allow for saving and collection of data.
# endregion

# region 6: Perform Inference
# TODO: Visualize the training results. Save the images to appropriate log file.
tracking_address = "/".join(log_manager.fPath2CurrDir)  # The path of the log file.
tracking_history = "_".join((log_manager.fPath2CurrDir[-1], "history.csv"))
tracking_address = "/".join((tracking_address, tracking_history))
hist_df = pd.DataFrame(history.history)
# Save history to CSV.
with open(tracking_address, mode='w') as f:
    hist_df.to_csv(f)
# Evaluate model on the testing data.
results = model.evaluate(x_test, y_test, batch_size=128)
# endregion

# TODO: Test calculation time if single dataset through model.evaluate.

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
