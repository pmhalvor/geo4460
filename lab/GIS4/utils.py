from simpledbf import Dbf5

import os  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_snow_melt_data_sampled(dir_path: str = ".", file_name: str = "sample_raster_snow_melt.csv"):
    """
    Method for loading randomly sampled snow and thaw data.
    Load snow and thaw data from the specified directory.

    Args:
        dir_path (str): The path to the directory containing the data file.

    Returns:
        dict: A dictionary containing the loaded data.
    """
    df = pd.read_csv(os.path.join(dir_path, file_name), sep=";", header=0)
    
    df = preprocess(df)
    return df


def load_snow_melt_data_raw(dir_path: str = ".", file_name: str = "month_thaw_snow.dbf"):
    """
    Method for loading raw snow and thaw data.
    Load snow and thaw data from the specified directory.

    Args:
        dir_path (str): The path to the directory containing the data files.

    Returns:
        dict: A dictionary containing the loaded data.
    """
    snow_melt_path = os.path.join(dir_path, file_name)

    snow_melt_dbf = Dbf5(snow_melt_path)

    return snow_melt_dbf.to_dataframe()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by renaming columns and dropping unnecessary ones.

    Args:
        data (pd.DataFrame): The data to preprocess.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    df.rename(columns={"X": "XM", "Y": "YM"}, inplace=True)
    df.drop(columns=["sample"], inplace=True)
    df.rename(columns=lambda col: col.replace("Raster_Krig_", "").replace("_Band_1", ""), inplace=True)
    df = df.replace(',', '.', regex=True).astype(float)
    return df[["XM", "YM", "SNOW200705", "THAW200705", "THAW200706", "THAW200707", "THAW200708", "THAW200709"]].copy()


def process_regression_data(data: dict):
    regression_df = pd.DataFrame(data, columns=['THAW Column', '1st Order', '2nd Order', '3rd Order'])

    regression_df['1st Order'] = regression_df['1st Order'].apply(lambda x: np.round(x, 5))
    regression_df['2nd Order'] = regression_df['2nd Order'].apply(lambda x: np.round(x, 5))
    regression_df['3rd Order'] = regression_df['3rd Order'].apply(lambda x: np.round(x, 5))

    return regression_df



if __name__ == "__main__":
    load_snow_melt_data_sampled("exports")
    # load_snow_thaw_data_raw("GIS4_datafiles")