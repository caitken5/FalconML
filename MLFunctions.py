import itertools
import os
import pathlib

import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split


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
        data = np.asarray(np.load(file_name))
        x_train = data
        y_train = self.y

        return x_train, y_train


# Create class that contains attributes of files that exist in desired folder.
class ManageData:
    def __init__(self, p_dir, o_dir, d_dir, v_dir, c_names=[]):
        self.fPath2ODir = list()
        self.fPath2DDir = list()
        self.fPath2DDirSub = ['Train', 'Test', 'Val']
        self.fileNames = list()
        self.dimensions = list()
        self.cForce = list()
        self.num_files = 0
        # Populate current attributes given input data.
        self.fPath2ODir.extend((p_dir, o_dir, v_dir))
        self.fPath2DDir.extend((p_dir, d_dir, v_dir))
        self.cNames = c_names
        self.cNamesLen = len(self.cNames)
        self.yIndex = self.cNames.index('C_Force')

    def add_file_names(self, filename):
        self.fileNames.append(filename)

    def add_dimensions(self, dimensions):
        self.dimensions.append(dimensions)

    def add_c_force(self, commanded_force):
        self.cForce.append(commanded_force)

    def add_dict(self, dictionary):
        self.cNames = dictionary

    def add_file_names_and_c_force(self, filename):
        self.add_file_names(filename)
        # Parse the filename to get the commanded force.
        temp_name = re.split("[._]", filename)
        c_force = '.'.join(temp_name[-3:-1])
        self.add_c_force(c_force)

    def populate_lists(self):
        sf_dir_str = "/".join(self.fPath2ODir)
        # Create new sub_folder.
        ddir_str = "/".join(self.fPath2DDir)
        os.mkdir(ddir_str)
        for file_name in os.listdir(sf_dir_str):
            if file_name.endswith(".npy"):
                self.add_file_names_and_c_force(file_name)
                f_open = os.path.join(sf_dir_str, file_name)
                temp_data = np.load(f_open)
                self.add_dimensions(temp_data.shape[0])
                self.num_files += 1
        self.save_populated_lists()

    def create_single_array(self):
        odir_str = "/".join(self.fPath2ODir)
        ddir_str = "/".join(self.fPath2DDir)
        # Loop through .npy file names.
        temp_data = list()
        for file_name in self.fileNames:
            # Because of populate_lists function, filenames are automatically .npy files.
            f_open = os.path.join(odir_str, file_name)
            temp_data_piece = np.load(f_open)
            temp_data = np.append(temp_data, temp_data_piece)
        # Reshape the appended data.
        temp_array = np.asarray(np.reshape(temp_data, (-1, len(self.cNames))))
        # Save the reshaped array. Save to appropriate model folder.
        name_arr = "/".join((ddir_str, self.fPath2DDir[-1]))
        self.generate_graph_and_statistics(temp_array, name_arr)
        np.save(name_arr, temp_array, allow_pickle=False, fix_imports=False)
        x_train, x_test, x_val, y_train, y_test, y_val = self.make_ttv_split(temp_array)  # Return 6 arrays.
        return x_train, x_test, x_val, y_train, y_test, y_val

    def save_populated_lists(self):
        # Save populated lists.
        # Save one file with data containing num_files, cNames, and yIndex.
        l_file = 'l_file.txt'
        n_folder = "/".join(self.fPath2DDir)
        l_file = "/".join((n_folder, l_file))
        temp_list = np.array([str(self.num_files), str(self.yIndex), str(self.cNamesLen)])
        np.savetxt(l_file, temp_list, fmt="%s")
        # Save second file with data containing fileNames, dimensions, and cForce
        att_file = 'att_file.txt'
        att_file = "/".join((n_folder, att_file))
        temp_arr = np.transpose(np.vstack([np.asarray(self.fileNames), np.asarray(self.dimensions),
                                           np.asarray(self.cForce)]))
        np.savetxt(att_file, temp_arr, fmt="%s")

    def get_populated_lists(self):
        # Retrieve populated lists.
        # Load associated l_file.
        n_file = "/".join(self.fPath2DDir)
        l_file = "/".join((n_file, "l_file.txt"))
        att_file = "/".join((n_file, "att_file.txt"))
        temp_list = np.loadtxt(l_file)
        self.num_files = int(temp_list[0])
        self.yIndex = int(temp_list[1])
        self.cNamesLen = int(temp_list[2])
        temp_arr = np.loadtxt(att_file, dtype='str')
        self.fileNames = list(temp_arr[:, 0])
        self.dimensions = list(temp_arr[:, 1])
        self.cForce = list(temp_arr[:, 2])
        print("Called ManageData.get_populated_lists.")

    def generate_graph_and_statistics(self, data, name):
        # TODO: Generate graph of data using seaborn, and table of statistics using pandas. Display and save both.
        # Convert data to pandas dataframe.
        data = pd.DataFrame(data, columns=self.cNames)
        # Note to self... probably shouldn't do the entire pandas array but select only a few attributes.
        my_plot = sns.pairplot(data, corner=True)
        # fig = my_plot.get_figure()
        # fig.savefig()
        my_plot.show()
        print("Called ManageData.generate_graph.")

    def make_ttv_split(self, data):
        print("Called ManageData.make_ttv_split.")
        y_data = data[:, self.yIndex]
        x_data = np.delete(arr=data, obj=self.yIndex, axis=1)
        x_tav, x_test, y_tav, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)
        x_train, x_val, y_train, y_val = train_test_split(x_tav, y_tav, test_size=0.2, stratify=y_tav)
        self.add_arr2sub(self.fPath2DDirSub[0], x_train, y_train)
        self.add_arr2sub(self.fPath2DDirSub[1], x_test, y_test)
        self.add_arr2sub(self.fPath2DDirSub[2], x_val, y_val)
        return x_train, x_test, x_val, y_train, y_test, y_val

    def add_arr2sub(self, f_name, x, y):
        # Create function that will add the numpy array to the corresponding sub folder.
        ddir_str = "/".join(self.fPath2DDir)
        path = "/".join((ddir_str, f_name))
        x_name = "/".join((path, "x"))
        y_name = "/".join((path, "y"))
        # Create folder to hold variables.
        os.mkdir(path)
        # Save x.
        np.save(x_name, x, allow_pickle=False, fix_imports=False)
        # Save y.
        np.save(y_name, y, allow_pickle=False, fix_imports=False)

    def load_sub2arr(self, f_name):
        ddir_str = "/".join(self.fPath2DDir)
        path = "/".join((ddir_str, f_name))
        x_name = "/".join((path, "x.npy"))
        y_name = "/".join((path, "y.npy"))
        x = np.load(x_name)
        y = np.load(y_name)
        return x, y

    def load_arr(self):
        # Load desired array into corresponding workspace.
        x_train, y_train = self.load_sub2arr(self.fPath2DDirSub[0])
        x_test, y_test = self.load_sub2arr(self.fPath2DDirSub[1])
        x_val, y_val = self.load_sub2arr(self.fPath2DDirSub[2])
        return x_train, x_test, x_val, y_train, y_test, y_val


# Create class that creates new model file folders for maintaining all associated data so the result could \
#  be recreated.
class ManageFileStructure:
    def __init__(self, project_directory, path_models, path_model):
        # Initializer needs two standard path names: path to where all models are kept, and name of current model type.
        self.fPath2CurrDir = list()
        self.fileNames = list()
        self.folderNums = list()
        self.prev_num = int()
        self.fPath2CurrDir.extend((project_directory, path_models, path_model))
        # Call def get_latest_file_path and generate new folder for latest model run.
        self.populate_folder_names()

    def add_file_names(self, filename):
        self.fileNames.append(filename)

    def populate_folder_names(self):
        # Using paths provided, search the given folder for current path names.
        # Join both parts of path together.
        dir_str = "/".join(self.fPath2CurrDir)
        if not os.path.isdir(dir_str):
            os.mkdir(dir_str)
        directory_contents = os.listdir(dir_str)
        for item in directory_contents:
            if os.path.isdir(os.path.join(dir_str, item)):
                self.fileNames.append(item)
                # Parse out number associated with fileName.
                folder_num = item.split("_")[-1]
                self.populate_folder_numbers(folder_num)

    def populate_folder_numbers(self, folder_num):
        self.folderNums.append(int(folder_num))

    def get_largest_folder_number(self):
        if not self.folderNums:
            self.prev_num = -1
        else:
            self.prev_num = max(self.folderNums)
        return self.prev_num

    def create_new_folder(self):
        # Assume populate_folder_names has already filled the associated lists. Get largest value of current folder \
        # set, add one, then create new folder. Assign name to saving directory.
        new_val = str(self.get_largest_folder_number() + 1)
        v_str = self.fPath2CurrDir[-1] + "_" + str(new_val)
        self.fPath2CurrDir.append(v_str)
        os.mkdir("/".join(self.fPath2CurrDir))

# Create class or function for printing some graphs (try using tensorboard).
