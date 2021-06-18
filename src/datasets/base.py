from logging import raiseExceptions
import os
import pickle

import numpy as np

import torch


class BaseDataset(torch.utils.data.Dataset):
    """ Base dataset class all datasets have in common. """

    def __init__(self, hparams, raw):
        """ Initialize dataset given hparams namespace and raw data. """
        super().__init__()
        self.hparams = hparams

        # load dataset and init class attributes
        # NOTE self.hparams must be set before calling _init_*
        self._init_time(raw)
        self._init_dummy(raw)

        if self.hparams.energy_lag != 0:
            self._init_energy(raw)

        if self.hparams.weather_lag != 0:
            self._init_weather(raw)

        self._filter_nans()
        self._scale()

        self._set_indices()

    def _init_time(self, raw):
        """ Initialise time attribute given the raw data. """
        if 'time' not in raw:
            raise AssertionError('Assume dataset dict contains time key.')
        else:
            self.time = raw['time']

    def _init_dummy(self, raw):
        """ Initialise dummy features attribute given the raw data. """
        if self.hparams.energy_class == 'load':
            dummy_features_keys = [
                'sin_hour', 'cos_hour',
                'sin_weekday', 'cos_weekday',
                'sin_year', 'cos_year',
                'weekend', 'holiday'
            ]
        else:
            dummy_features_keys = [
                'sin_hour', 'cos_hour',
                'sin_year', 'cos_year'
            ]
        dummy = list()
        for key in dummy_features_keys:
            if key not in raw:
                raise AssertionError(f'Assume dataset contains {key} key.')
            else:
                dummy.append(raw[key])

        # Try to take care of increasing share of wind or solar power
        # by using a linear time feature that increases with time.
        # NOTE: Choose M for months, H for Hours, Y for years, ...
        self.dummy = np.array(dummy).T

    def _init_energy(self, raw):
        """ Initialise energy data attribute given the raw data. """
        # collect energy time-series
        if self.hparams.energy_target == 'germany':
            self.energy = raw[f'{self.hparams.energy_class}_{self.hparams.energy_target}']
        elif self.hparams.energy_target in ['50hertz', 'amprion', 'tennet', 'transnetbw']:
            self.energy = raw[f'{self.hparams.energy_class}_{self.hparams.energy_target}']
        else:
            raise AssertionError(
                'Assume energy target parameter in '
                '["germany", "germany-tso", "tso", '
                '"50hertz","amprion","tennet","trannsnetbet"].'
            )

    def _init_weather(self, raw):
        """ Initialise weather data attribute given the raw data. """
        # collect weather features
        if self.hparams.weather_source == 'dwd':
            data = raw[f'{self.hparams.weather_source}_{self.hparams.interpolation_type}_{self.hparams.weather_class}']
            if self.hparams.weather_replace_nan is not None:
                if self.hparams.weather_replace_nan.lower() == 'mean':
                    data[np.isnan(data)] = np.nanmean(data)
                elif self.hparams.weather_replace_nan.lower() == 'nearest':
                    data[np.isnan(data)] = raw[f'{self.hparams.weather_source}_nearest_{self.hparams.weather_class}'][np.isnan(data)]
                else:
                    raise Exception('Unkown \'weather_replace_nan\' method.')
            weather = data
        elif self.hparams.weather_source == 'era5':
            weather = raw[f'{self.hparams.weather_source}_{self.hparams.weather_class}']

        # transform regarding hparams
        if self.hparams.weather_transform is not None:
            if self.hparams.weather_transform.lower() == 'mean':
                weather = weather.reshape(weather.shape[0], -1).mean(axis=1)
            elif self.hparams.weather_transform.lower().startswith('clusters'):
                pass
            else:
                raise Exception('Unkown \'weather_transform\' method.')
        
        self.weather = weather

    def _filter_nans(self):
        """ Filter nan values for all given features. """
        valid = np.isnan(self.time)
        valid |= np.isnan(self.dummy).reshape(
            self.dummy.shape[0], -1).any(axis=1)
        if self.hparams.energy_lag != 0:
            valid |= np.isnan(self.energy).reshape(
                self.energy.shape[0], -1).any(axis=1)
        if self.hparams.weather_lag != 0:
            valid |= np.isnan(self.weather).reshape(
                self.weather.shape[0], -1).any(axis=1)
        valid = ~valid

        self.time = self.time[valid]
        self.dummy = self.dummy[valid]
        if self.hparams.energy_lag != 0:
            self.energy = self.energy[valid]
        if self.hparams.weather_lag != 0:
            self.weather = self.weather[valid]

        delta = self.time[1] - self.time[0]
        self.intervals = list()
        start = 0
        for i in range(self.time.shape[0] - 1):
            if self.time[i] + delta != self.time[i+1]:
                self.intervals.append((start, i+1))
                start = i+1
        self.intervals.append((start, self.time.shape[0]))

    def _set_indices(self):
        """ Generate list of indices that are possible to use. """
        indices = list()
        i = 0
        past = max(self.hparams.energy_lag, self.hparams.weather_lag)
        future = self.hparams.forecast_horizon
        for start, end in self.intervals:
            possible_indices = end - start - (past - 1) - future
            if possible_indices >= 1:
                start_idx = start + (past - 1)
                end_idx = start_idx + possible_indices
                indices.append(np.arange(start_idx, end_idx))
        self.indices = np.concatenate(indices)

    def _scale(self):
        """ Scale all energy and weather data. """
        # prepare class attribute and calculate min, max, mean, std
        self.scaling_stats = {}
        for name in ['energy', 'weather']:
            if getattr(self.hparams, f'{name}_lag') != 0:
                self.scaling_stats[name] = {}
                data = getattr(self, name)
                for stat in ['min', 'max', 'mean', 'std']:
                    self.scaling_stats[name][stat] = getattr(np, stat)(data)

        # scale energy
        if self.hparams.energy_lag != 0:
            self.energy = self.scale_energy(self.energy)

        # scale weather data
        if self.hparams.weather_lag != 0:
            self.weather = self.scale_weather(self.weather)

    def scale_energy(self, y):
        """ Scale energy data. """
        stats = self.scaling_stats['energy']
        if self.hparams.scaling.lower() == 'minmax':
            return (y - stats['min']) / (stats['max'] - stats['min'])
        elif self.hparams.scaling.lower() == 'meanstd':
            return (y - stats['mean']) / stats['std']
        else:
            raise Exception('Unknown \'scaling\' method.')

    def inverse_scale_energy(self, y):
        """ Inverse scale energy data. """
        stats = self.scaling_stats['energy']
        if self.hparams.scaling.lower() == 'minmax':
            return (y * (stats['max'] - stats['min'])) + stats['min']
        elif self.hparams.scaling.lower() == 'meanstd':
            return (y * stats['std']) + stats['mean']
        else:
            raise Exception('Unknown \'scaling\' method.')

    def scale_weather(self, y):
        """ Scale weather data. """
        stats = self.scaling_stats['weather']
        if self.hparams.scaling.lower() == 'minmax':
            return (self.weather - stats['min']) / (stats['max'] - stats['min'])
        elif self.hparams.scaling.lower() == 'meanstd':
            return (self.weather - stats['mean']) / stats['std']
        else:
            raise Exception('Unknown \'scaling\' method.')

    def inverse_scale_weather(self, y):
        """ Inverse scale weather data. """
        # prepare min, max, mean, std as tensor values
        stats = self.scaling_stats['weather']
        if self.hparams.scaling.lower() == 'minmax':
            return (y * (stats['max'] - stats['min'])) + stats['min']
        elif self.hparams.scaling.lower() == 'meanstd':
            return (y * stats['std']) + stats['mean']
        else:
            raise Exception('Unknown \'scaling\' method.')

    def __len__(self):
        """ Get length of dataset. """
        return len(self.indices)
