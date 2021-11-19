import torch

from src.models import BaseModel


class MLP(BaseModel):
    """
    Simple neural network containing 128 hidden neurons and processing
    energy and calendar time series data to forecast energy.
    """

    def __init__(self, hparams, dataset):
        super().__init__(hparams, dataset)

    @torch.no_grad()
    def _build(self):
        example_energy = self.dataset[0][0][0]
        example_calendar = self.dataset[0][0][1]
        num_energy_features = example_energy.shape[-1]
        num_calendar_features = example_calendar.shape[0]
        self._fc_sequential = torch.nn.Sequential(
            torch.nn.Linear(num_energy_features + num_calendar_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        """ Inference of the neural network model. """
        energy, calendar = x
        collected_features = torch.cat([energy, calendar], axis=1)
        return self._fc_sequential(collected_features)
