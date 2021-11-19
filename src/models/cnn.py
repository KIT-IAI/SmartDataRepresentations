import torch

from src.models import BaseModel


class CNN(BaseModel):
    """
    Deep Neural Network for processing energy and calendar data,
    where the calendar data is processed by convolutional layers.
    """

    def __init__(self, hparams, dataset):
        super().__init__(hparams, dataset)

    @torch.no_grad()
    def _build(self):
        example_energy = self.dataset[0][0][0]
        example_calendar = self.dataset[0][0][1]

        # energy processing part
        # reshaped energy data (1, x, y)
        block = lambda input_size, features: [
            torch.nn.Conv2d(input_size, features, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        ]
        self._energy_net = torch.nn.Sequential(
            *block(1, 16),
            *block(16, 32),
            torch.nn.Flatten()
        )
        energy_features = self._energy_net(torch.Tensor([example_energy]))
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
