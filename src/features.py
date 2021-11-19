import numpy as np
import pandas as pd
import xarray as xr

from workalendar.europe import Germany
from xarray.core.dataarray import DataArray


def calendar_features(x):
    """ Generate calendar features for each given time step. """
    time = pd.to_datetime(x.time)
    features = list()
    for attr, tmax in [('hour', 23), ('dayofweek', 6), ('dayofyear', 365)]:
        data = getattr(time, attr)
        features.append(np.sin(2 * np.pi * data / tmax))
        features.append(np.cos(2 * np.pi * data / tmax))

    germany = Germany()
    features.append(np.array(list(map(lambda x: germany.is_holiday(x), time))))
    features.append(time.dayofweek / 5 >= 1)
    return xr.DataArray(
        data=np.array(features, dtype=float).T,
        dims=['time', 'cal'],
        coords={
            'time': time
        }
    )


def differencing(x, forecast_horizon):
    """ Calculate differences depending on the forecast horizon. """
    data = x.values

    kernel = np.zeros(forecast_horizon + 1)
    kernel[0] = 1
    kernel[-1] = -1

    data = np.convolve(data, kernel, mode='valid')

    return xr.DataArray(
        data=data,
        dims=['time'],
        coords={
            'time': pd.DatetimeIndex(x.time[forecast_horizon:])
        }
    )


def reshaping(x):
    """ Reshape hourly data of a week such that it is a 7x24 dimensional image. """
    return DataArray(
        data=x.values.reshape(-1, 1, 7, 24),
        dims=['time', 'channel', 'days', 'hours'],
        coords={
            'time': x.time
        }
    )
