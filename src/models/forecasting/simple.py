import torch

from models.forecasting.base import BaseModel

class SimpleCNN(BaseModel):
    """ """

    def __init__(self, hparams, ds_train, ds_val, ds_test):
        super().__init__(hparams, ds_train, ds_val, ds_test)
        
    @torch.no_grad()
    def _build(self):
        example_dummy = self.ds_train[0][0][1]
        example_energy = self.ds_train[0][0][2]
        example_weather = self.ds_train[0][0][3]

        num_dummy_features = example_dummy.shape[0]

        if self.hparams.energy_lag > 0:
            self._build_energy_net(shape=example_energy.shape)
            energy_features = self._energy_net(torch.Tensor([example_energy]))
            num_energy_features = energy_features.shape[-1]
        else:
            num_energy_features = 0

        if self.hparams.weather_lag > 0:
            self._build_weather_net(shape=example_weather.shape)
            weather_features = self._weather_net(torch.Tensor([example_weather]))
            num_weather_features = weather_features.shape[-1]
        else:
            num_weather_features = 0

        self._fc_sequential = torch.nn.Sequential(
            torch.nn.Linear(num_weather_features + num_energy_features + num_dummy_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.hparams.dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def _build_energy_net(self, shape):
        if len(shape) == 1:
            # one dimensional energy data (naive time-series)
            self._energy_net = torch.nn.Sequential(
                torch.nn.Linear(shape[0], 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.hparams.dropout),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU()
            )
        elif len(shape) == 3:
            # reshaped energy data (1, x, y)
            block = lambda input_size, features: [
                torch.nn.Conv2d(input_size, features, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(features),
                torch.nn.MaxPool2d(2),
            ]
            self._energy_net = torch.nn.Sequential(
                *block(1, 16),
                *block(16, 32),
                torch.nn.Flatten()
            )

    def _build_weather_net(self, shape):
        if len(shape) == 1:
            # one dimensional weather data (naive time-series)
            if self.hparams.weather_lag == 1:
                self._weather_net = torch.nn.Identity()
            else:
                self._weather_net = torch.nn.Sequential(
                    torch.nn.Linear(self.ds_train[0][0][3].shape[0], 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU()
                )
        elif len(shape) == 3:
            # three dimensional weather data (filter, x, y)
            if self.hparams.weather_lag == 1:
                block = lambda input_size, features: [
                    torch.nn.Conv2d(input_size, features, 3),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(features),
                    torch.nn.Conv2d(features, features, 3),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(features),
                    torch.nn.MaxPool2d(2),
                ]
                self._weather_net = torch.nn.Sequential(
                    *block(1, 16),
                    *block(16, 32),
                    *block(32, 64),
                    torch.nn.Flatten()
                )
            else:
                # NOTE: Conv3D need a lot of memory and computational power.
                #       Therefore, we decided to use Conv2D.
                #       Also, Conv3D did not outperform Conv3D (as far as we tested).
                # block = lambda input_size, features: [
                #     torch.nn.Conv3d(input_size, features, 3),
                #     torch.nn.ReLU(),
                #     torch.nn.BatchNorm3d(features),
                #     torch.nn.Conv3d(features, features, 3),
                #     torch.nn.ReLU(),
                #     torch.nn.BatchNorm3d(features),
                #     torch.nn.MaxPool3d(2),
                # ]
                # self._weather_net = torch.nn.Sequential(
                #     *block(1, 8),
                #     *block(8, 16),
                #     # *block(16, 32),
                #     torch.nn.Flatten()
                # )

                block = lambda input_size, features: [
                    torch.nn.Conv2d(input_size, features, 3),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(features),
                    torch.nn.Conv2d(features, features, 3),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(features),
                    torch.nn.MaxPool2d(2),
                ]
                self._weather_net = torch.nn.Sequential(
                    *block(self.hparams.weather_lag, 16),
                    *block(16, 32),
                    *block(32, 64),
                    torch.nn.Flatten()
                )
        else:
            raise NotImplemented('Weather data can only be shaped as 1D or 2D+t (3D) data.')

    def forward(self, x: torch.Tensor):
        """Forward data through the network."""
        _, dummy, energy, weather = x
        feature_set = list()

        feature_set.append(dummy.float())

        if self.hparams.energy_lag > 0:
            energy_features = self._energy_net(energy)
            feature_set.append(energy_features)

        if self.hparams.weather_lag > 0:
            weather_features = self._weather_net(weather)
            feature_set.append(weather_features)

        collected_features = torch.cat(feature_set, axis=1)

        return self._fc_sequential(collected_features)
