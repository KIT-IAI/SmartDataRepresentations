import os
import sys
import pickle
import argparse
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

sys.path.append(os.path.join('src'))

import datasets
from helper.terminal_callback import TerminalCallback
from models.forecasting.simple import SimpleCNN

import wandb


def run(hparams):
    """ Train, validate, and test the model based on hyperparameters set. """
    # set number of threads fixed due to multiple parallel runs
    torch.set_num_threads(1)

    # seed run
    seed_everything(hparams.seed, workers=True)

    # load raw data and prepare data loaders
    with open(os.path.join('data', 'dataset.pkl'), 'rb') as file:
        raw = pickle.load(file)

    train = getattr(datasets, hparams.dataset)(hparams, raw)
    val = getattr(datasets, hparams.dataset)(hparams, raw)
    test = getattr(datasets, hparams.dataset)(hparams, raw)

    val_split_idx = np.where(train.time == np.datetime64('2018-01-01T00:00'))[0][0]
    test_split_idx = np.where(train.time == np.datetime64('2019-01-01T00:00'))[0][0]
    train_idx, val_idx, test_idx = (
        train.indices[:np.where(train.indices >= val_split_idx)[0][0]],
        train.indices[np.where(train.indices >= val_split_idx)[0][0]:np.where(train.indices >= test_split_idx)[0][0]],
        train.indices[np.where(train.indices >= test_split_idx)[0][0]:]
    )

    train.indices = train_idx
    val.indices = val_idx
    test.indices = test_idx

    train_loader = DataLoader(train, batch_size=hparams.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=hparams.batch_size, num_workers=0)
    test_loader = DataLoader(test, batch_size=hparams.batch_size, num_workers=0)

    # init model
    model = SimpleCNN(hparams, train, val, test)
    print(model)

    # init wandb
    wandb.init(project='ci-paper')
    wandb.watch(model)

    # init loggers and trainer
    terminal_callback = TerminalCallback()
    timecode = datetime.now().strftime('%H-%M-%S')
    callbacks = [
        terminal_callback,
        ModelCheckpoint(monitor=hparams.monitor, dirpath=os.path.join('ckpts', timecode), save_top_k=1),
        EarlyStopping(monitor=hparams.monitor, **hparams.early_stopping_params)
    ]
    wandb_logger = WandbLogger()
    logger = [
        wandb_logger
    ]
    trainer = Trainer(gpus=-1, callbacks=callbacks, logger=logger,
                      progress_bar_refresh_rate=0, num_sanity_val_steps=0)

    # train, evaluate and test model
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path, test_dataloaders=test_loader)

    # log optimal metrics and close wandb
    wandb_logger.log_metrics(terminal_callback.opt_metric)
    wandb.finish()

    # clean up
    os.remove(trainer.checkpoint_callback.best_model_path)


def none_string(val):
    if not val or val.lower() == 'none':
        return None
    else:
        return val


def bool_string(val):
    if val.lower() in ['t', 'true']:
        return True
    elif val.lower() in ['f', 'false']:
        return False
    else:
        raise Exception('Please use t/true or f/false for boolean parameters.')


def get_parser():
    parser = argparse.ArgumentParser()

    # energy data params
    parser.add_argument('--energy_class', type=str, default='load')
    parser.add_argument('--energy_lag', type=int, default=168)  # 0: None, 1: 1D/2D, >1: 1D/2D+t
    parser.add_argument('--energy_target', type=str, default='germany')
    parser.add_argument('--energy_diff', type=bool_string, default=False)
    parser.add_argument('--energy_transform', type=none_string, default=None)
    parser.add_argument('--forecast_horizon', type=int, default=1)
    # weather data params
    parser.add_argument('--interpolation_type', type=str, default='linear')
    parser.add_argument('--weather_class', type=str, default='default')
    parser.add_argument('--weather_lag', type=int, default=0)  # 0: None, 1: 1D/2D, >1: 1D/2D+t
    parser.add_argument('--weather_source', type=str, default='era5')
    parser.add_argument('--weather_transform', type=none_string, default=None)
    parser.add_argument('--weather_replace_nan', type=none_string, default='nearest')
    # data, model, and train params
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='RealisticPointForecast')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--early_stopping_params', type=dict, default={'patience': 35, 'min_delta': 0.001})
    parser.add_argument('--monitor', type=str, default='val_loss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--optimizer_params', type=dict, default={'lr': 0.0001, 'amsgrad': True, 'weight_decay': 0.01})
    parser.add_argument('--scaling', type=str, default='meanstd')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--scheduler_params', type=dict, default={'cooldown': 5, 'patience': 5, 'factor': 0.5, 'verbose': True})
    parser.add_argument('--seed', type=int, default=42)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    hparams = parser.parse_args()

    if hparams.energy_diff:
        hparams.energy_lag = hparams.energy_lag + hparams.forecast_horizon

    if hparams.weather_class.lower() == 'default':
        if hparams.energy_class.lower() == 'load':
            hparams.weather_class = 't2m'
        if hparams.energy_class.lower() == 'solar':
            hparams.weather_class = 'ssr'
        if hparams.energy_class.lower() == 'wind':
            hparams.weather_class = 'u+v'

    run(hparams)
