import os
from enum import Enum
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from src.pytorch_lightning import TerminalCallback
from src.models import Simple, MLP, FCN, CNN


class Models(Enum):
    LINEAR = 0
    MLP = 1
    SIMPLE = 2
    DNN = 3


class MyDataset(Dataset):
    """
    Dataset class to return tuple data for multiple input networks.
    """
    def __init__(self, energy, calendar, y=None):
        """ Initialize energy, calendar, and target datasets. """
        self.energy = energy.astype(np.float32)
        self.calendar = calendar.astype(np.float32)
        if y is None:
            self.y = None
        else:
            self.y = y.astype(np.float32)

    def __getitem__(self, index):
        """ Get tuple data for multiple input networks. """
        energy = self.energy[index].flatten()
        calendar = self.calendar[index].flatten()
        if self.y is None:
            return (energy, calendar)
        else:
            return (energy, calendar), self.y[index]

    def __len__(self):
        """ Get the length of the dataset. """
        return len(self.energy)


class ModelHandler(BaseEstimator):
    """
    PyWATTS model handler class to initialize, train, and predict neural network models.
    """

    def __init__(self, hparams, name: str = "DNN"):
        super().__init__(name)
        self.hparams = hparams
        self.model = None
        self.trainer = None

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def fit(self, energy, calendar, y):
        """ Train, validate, and test the model based on hyperparameters set. """
        # seed run
        seed_everything(self.hparams.seed, workers=True)

        # WARNING: This is only works with a 2015-2019 training set.
        val_split_idx = np.where(
            pd.to_datetime(energy.time) == np.datetime64('2018-01-01T00:00')
        )[0][0]

        train = MyDataset(
            energy[:val_split_idx].values, calendar[:val_split_idx].values,
            y[:val_split_idx].values)
        validation = MyDataset(
            energy[val_split_idx:].values, calendar[val_split_idx:].values,
            y[val_split_idx:].values)
        train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(validation, batch_size=128, num_workers=0)

        # init model
        if self.hparams.energy_reshape:
            if self.hparams.model == Models.DNN:
                self.model = CNN(self.hparams, train)
            else:
                raise NotImplementedError('Please use a DNN for reshaped energy time series.')
        else:
            if self.hparams.model == Models.SIMPLE:
                self.model = Simple(self.hparams, train)
            elif self.hparams.model == Models.MLP:
                self.model = MLP(self.hparams, train)
            elif self.hparams.model == Models.DNN:
                self.model = FCN(self.hparams, train)

        # init loggers and trainer
        terminal_callback = TerminalCallback()
        timecode = datetime.now().strftime('%H-%M-%S')
        callbacks = [
            terminal_callback,
            ModelCheckpoint(monitor=self.hparams.monitor, dirpath=os.path.join('ckpts', timecode),
                            save_top_k=1),
            EarlyStopping(monitor=self.hparams.monitor, **self.hparams.early_stopping_params)
        ]
        wandb_logger = WandbLogger()
        logger = [
            wandb_logger
        ]

        self.trainer = Trainer(gpus=-1, callbacks=callbacks, logger=logger,
                               progress_bar_refresh_rate=0, num_sanity_val_steps=0)

        # train, evaluate and test model
        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.is_fitted = True

    def transform(self, energy, calendar, y):
        """ Forecast energy based on the trained network. """
        self.model.eval()
        with torch.no_grad():
            # WARNING: This is only works with a 2015-2019 training set.
            train_or_test = len(np.where(pd.to_datetime(energy.time)
                                == np.datetime64('2018-01-01T00:00'))[0]) == 0
            if train_or_test:
                # test
                dataset = MyDataset(energy.values, calendar.values, y.values)
                data_loader = DataLoader(dataset, batch_size=128, num_workers=0)
                self.trainer.test(
                    ckpt_path=self.trainer.checkpoint_callback.best_model_path,
                    test_dataloaders=data_loader
                )
                dataset = MyDataset(energy.values, calendar.values)
                data_loader = DataLoader(dataset, batch_size=128, num_workers=0)
                inference = self.trainer.predict(
                    ckpt_path=self.trainer.checkpoint_callback.best_model_path,
                    dataloaders=data_loader
                )
            else:
                # test
                dataset = MyDataset(energy.values, calendar.values)
                data_loader = DataLoader(dataset, batch_size=128, num_workers=0)
                inference = self.trainer.predict(
                    ckpt_path=self.trainer.checkpoint_callback.best_model_path,
                    dataloaders=data_loader
                )

        return numpy_to_xarray(np.concatenate([x.cpu().numpy() for x in inference]).flatten(),
                               energy, self.name)
