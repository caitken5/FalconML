import os
import numpy as np
import re
from tensorflow.keras.utils import Sequence


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
        file_name = self.fileNames[idx]

        # Open the file.
        data = np.asarray(np.load())
        x_train = data
        y_train = self.y

        return x_train, y_train


# Create class that contains attributes of files that exist in desired folder.
class FalconFolder:
    def __init__(self, path):
        self.path = path
        self.fileNames = list()
        self.dimensions = list()
        self.cForce = list()

    def add_file_names(self, filename):
        self.fileNames.append(filename)

    def add_dimensions(self, dimensions):
        self.dimensions.append(dimensions)

    def add_c_force(self, commanded_force):
        self.cForce.append(commanded_force)

    def add_file_names_and_c_force(self, filename):
        self.add_file_names(filename)
        # Parse the filename to get the commanded force.
        temp_name = re.split("[._]", filename)
        c_force = '.'.join(temp_name[-3:-1])
        self.add_c_force(c_force)

    def populate_lists(self):
        for file_name in os.listdir(self.path):
            if file_name.endswith(".npy"):
                self.add_file_names_and_c_force(file_name)
                f_open = os.path.join(self.path, file_name)
                temp_data = np.load(f_open)
                self.add_dimensions(temp_data.shape)

# TODO: Create class or function for printing some graphs (try using tensorboard).
