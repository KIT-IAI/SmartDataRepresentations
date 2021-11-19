import argparse
import pprint

import pandas as pd
import wandb

from pywatts.core.summary_formatter import SummaryJSON

from src.data_handler import get_data, train_test_split
from src.pipeline import create_pipeline
from src.pywatts.model_handler_module import Models


def bool_string(val):
    """ String to bool parser for argparse ArgumentParser.  """
    if val.lower() in ['t', 'true', 'y', 'yes']:
        return True
    elif val.lower() in ['f', 'false', 'n', 'no']:
        return False
    else:
        raise Exception('Please use t/true or f/false for boolean parameters.')


def model_parser(val):
    """ Model parser for argparse ArgumentParser. """
    if val.lower() in ['lm', 'linear', 'linearmodel']:
        return Models.LINEAR
    elif val.lower() in ['simple']:
        return Models.SIMPLE
    elif val.lower() in ['mlp']:
        return Models.MLP
    elif val.lower() in ['dnn', 'deep', 'neuralnetwork', 'deepneuralnetwork']:
        return Models.DNN
    else:
        raise Exception('Please choose one of the implemented models (Linear, DNN)')


def get_parser():
    """ Create argparse ArgumentParser. """
    parser = argparse.ArgumentParser()

    # energy data params
    parser.add_argument('--energy_differences', type=bool_string, default=False)
    parser.add_argument('--energy_reshape', type=bool_string, default=False)
    parser.add_argument('--forecast_horizon', type=int, default=1)

    # data, model, and train params
    parser.add_argument('--model', type=model_parser, default=Models.DNN)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--early_stopping_params', type=dict, default={'patience': 35, 'min_delta': 0.001})
    parser.add_argument('--monitor', type=str, default='val_loss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--optimizer_params', type=dict, default={'lr': 0.0001, 'amsgrad': True, 'weight_decay': 0.01})
    parser.add_argument('--scaling', type=str, default='StandardScaler')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--scheduler_params', type=dict, default={'cooldown': 5, 'patience': 5, 'factor': 0.5, 'verbose': True})
    parser.add_argument('--seed', type=int, default=1)

    # debug parameter
    parser.add_argument('--debugging', type=bool_string, default=False)

    return parser


def log_to_wandb(result, summary, path):
    """ Log all evaluation results including prediction to weights&biases. """
    eval_metrics = {
        f'{path}/scaled/mae': summary["Summary"]["MAE"]["results"]["y_hat"],
        f'{path}/scaled/mape': summary["Summary"]["MAPE"]["results"]["y_hat"]
    }
    print(result.keys())
    data = [[time, energy] for time, energy in
            zip(pd.to_datetime(result["GroundTruth"].time), result["Forecast"].values.flatten())]
    table = wandb.Table(data=data, columns=['time', 'energy'])
    wandb.log({'test/forecast': table})
    wandb.log(eval_metrics)
    pprint.pprint(eval_metrics)
    

def main():
    """ Parse arguments, run pipeline, and log results for evaluation. """
    parser = get_parser()
    hparams = parser.parse_args()

    data = get_data()
    train, test = train_test_split(data)

    wandb.init('ci-paper-2021')

    pipeline = create_pipeline(hparams)
    result, summary = pipeline.train(train, summary_formatter=SummaryJSON(), summary=True)
    log_to_wandb(result, summary, "train_val")

    result, summary = pipeline.test(test, summary_formatter=SummaryJSON(), summary=True)
    log_to_wandb(result, summary, "test")

    wandb.finish()


if __name__ == '__main__':
    main()
