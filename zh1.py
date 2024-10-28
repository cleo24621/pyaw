# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/21
@DESCRIPTION: 
"""
import h5py
import pandas as pd

fp = r"D:\cleo\master\pyaw\data\ZH-1\HPM\FGM\CSES_01_HPM_5_L02_A2_106060_20191231_231400_20191231_235019_000.h5"

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

