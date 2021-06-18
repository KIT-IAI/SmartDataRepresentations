import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isna


OPSD_URL = 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'
OPSD_KEEP = [
    'utc_timestamp',
    'DE_load_actual_entsoe_transparency',
    'DE_solar_capacity',
    'DE_solar_generation_actual',
    'DE_wind_capacity',
    'DE_wind_generation_actual',
]
WEATHER_URL = 'https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv'
WEATHER_KEEP = [
    'utc_timestamp',
    'DE_temperature'
]


def download(url, output):
    os.system(f'wget {url} -O {output}')


def load_csv(path):
    with open(path, 'rb') as file:
        pkl = pd.read_csv(file)
        file.close()
    return pkl


def main():
    # download(OPSD_URL, 'opsd.csv')
    opsd = load_csv('opsd.csv')
    opsd = opsd[OPSD_KEEP]
    opsd['utc_timestamp'] = pd.to_datetime(opsd['utc_timestamp'])
    opsd.set_index('utc_timestamp', inplace=True)
    # download(WEATHER_URL, 'weather.csv')
    weather = load_csv('weather.csv')
    weather = weather[WEATHER_KEEP]
    weather['utc_timestamp'] = pd.to_datetime(weather['utc_timestamp'])
    weather.set_index('utc_timestamp', inplace=True)

    opsd_frame = opsd.join(weather)
    opsd_frame.to_csv('opsd_and_weather.csv')
    print(opsd_frame.describe())
    print(opsd_frame)

    for column in opsd_frame.columns:
        plt.plot(opsd_frame.index, opsd_frame[column])
        plt.savefig(f'{column}.png')
        plt.close()


if __name__ == '__main__':
    main()
