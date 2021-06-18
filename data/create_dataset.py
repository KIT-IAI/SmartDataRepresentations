import os
import pickle

import numpy as np
import pandas as pd

from workalendar.europe import Germany


OPSD_DATA_CSV = os.path.join('opsd', 'opsd_and_weather.csv')
OPSD_MAPPING = {
    'DE_load_actual_entsoe_transparency': 'load_germany',
    'DE_solar_generation_actual': 'solar_germany',
    'DE_wind_generation_actual': 'wind_germany',
    'DE_temperature': 'temp_germany',
}


def get_opsd_data(path):
    df = pd.read_csv(path)
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['utc_timestamp'] = df['utc_timestamp'].dt.tz_localize(None)
    df.set_index('utc_timestamp', inplace=True)
    return df


def load_calendar_features(time_index):
    germany = Germany()
    calendar_features = [
        # choose one of cyclic or linear features
        # linear calendar features
        (lambda x: x.hour, 'hour'),
        (lambda x: x.weekday(), 'weekday'),
        (lambda x: x.month, 'month'),
        (lambda x: x.timetuple().tm_yday, 'day_of_year'),
        # cyclic calendar features
        (lambda x: np.sin(2 * np.pi * x.hour / 23), 'sin_hour'),
        (lambda x: np.cos(2 * np.pi * x.hour / 23), 'cos_hour'),
        (lambda x: np.sin(2 * np.pi * x.weekday() / 6), 'sin_weekday'),
        (lambda x: np.sin(2 * np.pi * x.weekday() / 6), 'cos_weekday'),
        (lambda x: np.sin(2 * np.pi * x.timetuple().tm_yday / 364), 'sin_year'),
        (lambda x: np.cos(2 * np.pi * x.timetuple().tm_yday / 364), 'cos_year'),
        # boolean calendar features
        (lambda x: x.weekday() > 4, 'weekend'),
        (lambda x: germany.is_holiday(x), 'holiday')
    ]

    dataset = {}
    for feature_func, name in calendar_features:
        func = np.vectorize(feature_func)
        dataset[name] = func(time_index.astype(object)).astype(np.float32)
    
    return dataset


def main():
    opsd_df = get_opsd_data(OPSD_DATA_CSV)
    dataset = {'time': opsd_df.index}
    dataset.update(load_calendar_features(dataset['time']))
    for key, value in OPSD_MAPPING.items():
        dataset[value] = opsd_df[key].values.astype(np.float32)

    with open('dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)


if __name__ == '__main__':
    main()
