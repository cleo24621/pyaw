import h5py
import pandas as pd
from pandas import DataFrame


def _recursively_extract_data(group, data_dict):
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
                data_dict[f"{key}1"] = data[:, 0]
                data_dict[f"{key}2"] = data[:, 1]
                data_dict[f"{key}3"] = data[:, 2]
        elif isinstance(item, h5py.Group):
            # If the item is a group, recursively extract its datasets
            _recursively_extract_data(item, data_dict)


# main()
def get_df(file_path: str) -> DataFrame:
    """
    Args:
        file_path:

    Returns:
        单维变量数据为1列，多维变量数据拆分成多列
    """
    # Open the HDF5 file in read mode
    with h5py.File(file_path, "r") as hdf:
        # Create an empty dictionary to hold dataset names and data
        data_dict = {}
        # Recursively iterate through groups and datasets
        _recursively_extract_data(hdf, data_dict)
    return pd.DataFrame(data_dict)
