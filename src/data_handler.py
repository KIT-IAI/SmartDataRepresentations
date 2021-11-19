import os

import numpy as np
import pandas as pd


def get_data():
    """ Load DataFrame for the energy forecasting use case. """
    with open(os.path.join('data', 'opsd_and_weather.csv'), 'r') as file:
        data = pd.read_csv(file)

    # create pandas DataFrame and store energy and time data in it
    # NOTE: Using energy and time starting at 1: because of first nan values
    data_df = pd.DataFrame()
    data_df['time'] = pd.to_datetime(data['utc_timestamp'][1:])
    data_df['energy'] = data['DE_load_actual_entsoe_transparency'][1:]
    data_df.set_index('time', inplace=True)

    return data_df


def train_test_split(data):
    """ Create train and test split. Train will also split into train and validation later. """
    test_start = "2019-01-01 00:00"
    test_end = "2020-01-01 00:00"
    start_idx = np.where(data.index == test_start)[0][0]
    end_idx = np.where(data.index == test_end)[0][0]

    train = data[:start_idx]
    test = data[start_idx:end_idx]

    return train, test
