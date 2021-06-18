import numpy as np

from datasets.base import BaseDataset


class RealisticPointForecast(BaseDataset):
    """ Real point forecast dataset class. """

    def __init__(self, hparams, raw):
        """ Initialize dataset given hparams namespace and raw data. """
        super().__init__(hparams, raw)

    def __getitem__(self, index):
        """ Return data for realistic point forecast. """
        if index < 0:
            index = self.__len__() + index

        index = self.indices[index]

        dummy_data = self.dummy[index + self.hparams.forecast_horizon]

        if self.hparams.energy_lag == 0:
            energy_data = []
        else:
            energy_data = self.energy[index - self.hparams.energy_lag + 1:index + 1]
            if self.hparams.energy_diff:
                energy_data = energy_data[self.hparams.forecast_horizon:] - energy_data[:(-1) * self.hparams.forecast_horizon]

            if self.hparams.energy_transform is None:
                pass
            elif self.hparams.energy_transform == '24x7':
                energy_data = energy_data.reshape(1, 7, 24)
            # elif self.hparams.energy_transform == '24x7_same':
            #     current_timestamp = self.time[index].astype(object)
            #     last_monday_index = index - (current_timestamp.weekday() * 24 + current_timestamp.hour)
            #     energy_data = np.concatenate([energy_data[last_monday_index:], energy_data[:last_monday_index]])
            #     energy_data = energy_data.reshape(1, 7, 24)
            else:
                raise NotImplementedError('Unknown energy_transform setting!')

        if self.hparams.weather_lag == 0:
            weather_data = []
        else:
            if len(self.weather.shape) == 1:
                weather_data = self.weather[index - self.hparams.weather_lag + 1:index + 1].flatten()
            else:
                weather_data = self.weather[index - self.hparams.weather_lag + 1:index + 1]

        if self.hparams.energy_diff:
            y = self.energy[index + self.hparams.forecast_horizon] - self.energy[index]
        else:
            y = self.energy[index + self.hparams.forecast_horizon]

        return ((index, dummy_data, energy_data, weather_data), y)


class TrueWeatherPointForecast(BaseDataset):
    """ Oracle point forecast dataset class. """

    def __init__(self, hparams, raw):
        """ Initialize dataset given hparams namespace and raw data. """
        super().__init__(hparams, raw)

    def __getitem__(self, index):
        """ Return data for oracle point forecast. """
        if index < 0:
            index = self.__len__() + index

        index = self.indices[index]

        dummy_data = self.dummy[index + self.hparams.forecast_horizon]

        if self.hparams.energy_lag == 0:
            energy_data = []
        else:
            energy_data = self.energy[index - self.hparams.energy_lag + 1:index + 1]
            if self.hparams.energy_transform is None:
                pass
            elif self.hparams.energy_transform == '24x7':
                energy_data = energy_data.reshape(1, 7, 24)
            else:
                raise NotImplementedError('Unknown energy_transform setting!')

        if self.hparams.weather_lag == 0:
            raise Exception('Leaving weather data does not make sense for True Weather Datasets.')
        else:
            if len(self.weather.shape) == 1:
                if self.hparams.weather_lag == 1:
                    weather_data = self.weather[[index + self.hparams.forecast_horizon]]
                elif self.hparams.weather_lag > 1:
                    raise Exception('Weather lag greater than 1 does not make sense for True Weather Datasets.')
            else:
                if self.hparams.weather_lag == 1:
                    weather_data = self.weather[[index + self.hparams.forecast_horizon]]
                elif self.hparams.weather_lag > 1:
                    raise Exception('Weather lag greater than 1 does not make sense for True Weather Datasets.')

        y = self.energy[index + self.hparams.forecast_horizon]

        return ((index, dummy_data, energy_data, weather_data), y)
