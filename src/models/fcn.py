import torch

from src.models import BaseModel


class FCN(BaseModel):
    """
    Deep Neural Network for processing energy and calendar data,
    where the energy data is processed by fully-connected layers.
    """

    def __init__(self, hparams, dataset):
        super().__init__(hparams, dataset)

    @torch.no_grad()
    def _build(self):
        example_energy = self.dataset[0][0][0]
        example_calendar = self.dataset[0][0][1]

        # energy processing part
        # one dimensional energy data (naive time-series)
        self._energy_net = torch.nn.Sequential(
            torch.nn.Linear(168, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU()
        )
        energy_features = self._energy_net(torch.Tensor(example_energy.T))
        num_energy_features = energy_features.shape[-1]

        num_calendar_features = example_calendar.shape[0]
        self._fc_sequential = torch.nn.Sequential(
            torch.nn.Linear(num_energy_features + num_calendar_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        """ Inference of the neural network model. """
        energy, calendar = x
        feature_set = list()

        feature_set.append(calendar.float())
        energy_features = self._energy_net(energy)
        feature_set.append(energy_features)

        collected_features = torch.cat(feature_set, axis=1)

        return self._fc_sequential(collected_features)
