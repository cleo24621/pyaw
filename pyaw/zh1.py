# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/22
@DESCRIPTION: 
"""
import os

import h5py
import pandas as pd
import matplotlib.pyplot as plt

class ZH1:
    def __init__(self):
        self.fv_gpm_fgm = (-999999,-9999)

    @staticmethod
    def r_hpm_fgm(fp):
        # Open the HDF5 file in read mode
        with h5py.File(fp, 'r') as hdf:
            # Create an empty dictionary to hold dataset names and data
            data_dict = {}


            # Recursively iterate through groups and datasets
            def recursively_extract_data(group, data_dict):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        data = item[:]
                        # Check the shape of the dataset
                        if data.shape[1] == 1:
                            # Flatten the (n,1) dataset to a single column
                            data_dict[key] = data.flatten()
                        elif data.shape[1] == 3:
                            # Split the (n,3) dataset into 3 separate columns
                            data_dict[f'{key}1'] = data[:, 0]
                            data_dict[f'{key}2'] = data[:, 1]
                            data_dict[f'{key}3'] = data[:, 2]
                    elif isinstance(item, h5py.Group):
                        # If the item is a group, recursively extract its datasets
                        recursively_extract_data(item, data_dict)


            recursively_extract_data(hdf, data_dict)

            # Convert the dictionary to a pandas DataFrame
            df = pd.DataFrame(data_dict)
            df['DATETIME'] = pd.to_datetime(df['UTC_TIME'].astype(str), format='%Y%m%d%H%M%S%f')
        return df

    @staticmethod
    def r_hpm_cdsm(fp):
        # Open the HDF5 file in read mode
        with h5py.File(fp, 'r') as hdf:
            # Create an empty dictionary to hold dataset names and data
            data_dict = {}

            # Recursively iterate through groups and datasets
            def recursively_extract_data(group, data_dict):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        data = item[:]
                        # Flatten the (n,1) dataset to a single column
                        data_dict[key] = data.flatten()
                    elif isinstance(item, h5py.Group):
                        # If the item is a group, recursively extract its datasets
                        recursively_extract_data(item, data_dict)

            recursively_extract_data(hdf, data_dict)

            # Convert the dictionary to a pandas DataFrame
            df = pd.DataFrame(data_dict)
            df['DATETIME'] = pd.to_datetime(df['UTC_TIME'].astype(str), format='%Y%m%d%H%M%S%f')
        return df


# # read all data of 1 d
# # according to time, slice to 24 hour
# dirp = r"D:\cleo\PycharmProjects\auroramaps\data\ZH-1\HPM\1111"
# fns = os.listdir(dirp)
# fps = [os.path.join(dirp, fn) for fn in fns]
# dfs = [ZH1.r_hpm_fgm(fp) for fp in fps]
# df_20191111 = pd.concat(dfs,ignore_index=True)
# df_20191111['HOUR'] = df_20191111['DATETIME'].dt.hour
# grouped = df_20191111.groupby('HOUR')
# # Step 3: Split into separate DataFrames by hour
# dfs_by_hour = {hour: group for hour, group in grouped}
# for hour, df in dfs_by_hour.items():
#     # Customize file name with hour and additional information
#     file_name = f'fgm_20191111_{hour}.pkl'
#
#     # Save the DataFrame as a .pkl file
#     df.to_pickle(file_name)
#
#     # You can print or log the saved file name for confirmation
#     print(f"Saved: {file_name}")
