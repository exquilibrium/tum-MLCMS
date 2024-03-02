import os
import pandas as pd
from utils.data_preprocessing import *
def get_all_file_paths(folder_path):
    # Initialize an empty list to store file paths
    file_paths = []

    # Iterate through the directory tree using os.walk
    for root, dirs, files in os.walk(folder_path):
        # Iterate through the files in the current directory
        for file in files:
            # Append the full path of each file to the list
            file_paths.append(os.path.join(root, file))

    # Return the list of file paths
    return file_paths

def getdatapoints(all_file_paths,arg,k):
    """
    Does preprocessing on experimental data files.
    """
    # Initialize an empty list to store all file paths
    data_point = []

    for i in range(len(all_file_paths)):
        file_path = all_file_paths[i]
        print(str(i + 1) + "/" + str(len(all_file_paths)) + " procesing: " + str(file_path))
        # read file
        column_names = ['ID', 'FRAME', 'X', 'Y', 'Z']
        data = pd.read_csv(file_path, delimiter=' ', names=column_names)

        ID = data['ID'].unique()
        assert len(ID) == data['ID'].unique()[-1]

        t = Trajectories()
        for i in range(len(data)):
            t.add_frame(data['FRAME'][i], data['ID'][i], data['X'][i], data['Y'][i])
        t.calculate_speed(8)

        f = Frames(t)
        for i in range(len(data)):
            f.add_frame(data['FRAME'][i], data['ID'][i], data['X'][i], data['Y'][i])
        f.calculate_neighbors(k)

        for id_frame, value in f.frames.items():
            if len(value) != 0:
                for i in value:
                    if len(i['relative_position']) != 0:
                        x = i['relative_position'][:, 0]
                        y = i['relative_position'][:, 1]
                        v = i['relative_speed'][:, 0]
                        u = i['relative_speed'][:, 1]
                        s = i['Sk']
                        label_speed_v = i['v']
                        label_speed_u = i['u']
                        result_array = np.concatenate((x, y, v, u))
                        if arg.use_rel_vel:
                            result_array = np.concatenate((result_array, v, u))
                        if arg.use_mean_dist_spacing:
                            result_array = np.append(result_array, s)
                        result_array = np.append(result_array, label_speed_v)
                        result_array = np.append(result_array, label_speed_u)
                        data_point.append(result_array)
    return np.stack(data_point)
