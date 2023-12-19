"""Module that handles the creation of the pandas.DataFrame used in the analysis

Returns:
    df (pandas.Dataframe): pandas.DataFrame with the features, the targets or the detector information 
    depending on the flag apassed
"""

from pathlib import Path
import inspect
import os
import sys
import pandas as pd

ROOT_PATH = Path(__file__).parent

print(ROOT_PATH)

def sample_loader(flag):
    """Function that creates the pandas.Dataframe

    Args:
        flag (str): flag that defines which pandas.Dataframe has to be created.
            Possible values are: dataset, targets and geometry. Any other values will raise an error

    Returns:
        df (pandas.DataFrame): pandas.Dataframe with either the features, the targets or the detector geometry
    """
    try:
        if str(flag) == 'dataset':
            file_path = os.path.join(ROOT_PATH, 'batch_1.parquet')
            df = pd.read_parquet(file_path).reset_index()
        elif str(flag) == 'targets':
            file_path = os.path.join(ROOT_PATH, 'train_meta.parquet')
            df = pd.read_parquet(file_path).reset_index()
        elif str(flag) == 'geometry':
            file_path = os.path.join(ROOT_PATH, 'sensor_geometry.csv')
            df = pd.read_csv(file_path)
        else:
            print("Flag passed is wrong")
            sys.exit(1)
    except OSError as e:
        calling_frame = inspect.stack()[1]
        calling_module = inspect.getmodule(calling_frame[0]).__name__
        calling_function = calling_frame.function
        print(f"file not found while trying to create dataframe {flag} in function {calling_function} in module {calling_module}: {e}")
        sys.exit(1)

    return df

def sample_loader_non_local_testing(flag):
    """Function that creates the pandas.Dataframe to be run in the unit test

    Args:
        flag (str): flag that defines which pandas.Dataframe has to be created.
            Possible values are: dataset, targets and geometry. Any other values will raise an error

    Returns:
        df (pandas.DataFrame): pandas.Dataframe with either the feature or the targets
    """
    try:
        if str(flag) == 'dataset':
            file_path = os.path.join(ROOT_PATH, 'subset.parquet')
            df = pd.read_parquet(file_path).reset_index()
        elif str(flag) == 'targets':
            file_path = os.path.join(ROOT_PATH, 'res_subset.parquet')
            df = pd.read_parquet(file_path).reset_index()
        else:
            print("Flag passed is wrong")
            sys.exit(1)
    except OSError as e:
        calling_frame = inspect.stack()[1]
        calling_module = inspect.getmodule(calling_frame[0]).__name__
        calling_function = calling_frame.function
        print(f"file not found while trying to create dataframe {flag} in function {calling_function} in module {calling_module}: {e}")
        sys.exit(1)

    return df