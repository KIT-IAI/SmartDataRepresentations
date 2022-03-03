# Smart Data Representations: Impact on the Accuracy of Deep Neural Networks

This repository contains the Python implementation of the results presented in the following paper:
>[O. Neumann](mailto:oliver.neumann@kit.edu), N. Ludwig, M. Turowski, B. Heidrich, V. Hagenmeyer and R. Mikut, 2021, "Smart Data Representations: Impact on the Accuracy of Deep Neural Networks," in Proceedings. 31. Workshop Computational Intelligence Berlin, 25. – 26. November 2021, H. Schulte, F. Hoffmann, R. Mikut (Eds.), KIT Scientific Publishing Karlsruhe.

Available at [arXiv](https://arxiv.org/abs/2111.09128) or [KIT Scientific Publishing](https://www.ksp.kit.edu/site/books/e/10.5445/KSP/1000138532/).

## Abstract

Deep Neural Networks are able to solve many complex tasks with less engineering effort and better performance. However, these networks often use data for training and evaluation without investigating its representation, i.e. the form of the used data. In the present paper, we analyze the impact of data representations on the performance of Deep Neural Networks using energy time series forecasting. Based on an overview of exemplary data representations, we select four exemplary data representations and evaluate them using two different Deep Neural Network architectures and three forecasting horizons on real-world energy time series. The results show that, depending on the forecast horizon, the same data representations can have a positive or negative impact on the accuracy of Deep Neural Networks.

## Installation

Before the pipeline can be run, you need to prepare a python environment and download the energy time-series data. 

### Setup Python Environment

First, a virtual environment has to be set up. Therefore, you can use, for example, venv (`python -m venv venv`) or anaconda (`conda create -n env_name`) as a virtual environment. Afterwards, you can install the dependencies via `pip install -r requirements.txt`. 

### Download Data

After the environment is prepared, you can download the data by executing the python script in the data folder via `python download.py`. This downloads the OPSD data as CSV files and merges the data into an 'opsd_and_weather.csv' file. Also, some plots are made to give you a quick look into the data.

## Run Pipeline

Finally, you can run the pipeline via `python run.py` with the default parameters. You can see a list of available parameters for that script by calling `python run.py --help`. The most important parameters are:

```
--forecast_horizon (int)
    Defines the forecast time horizon,
    i.e. the next 24th hour.
--model (string)
    Defines the model to use in the pipeline,
    i.e. linear for a 'linear' regression model
    or 'DNN' for a deep neural network
    as presented in the paper.
--energy_differences (bool)
    If 'true' the differences
    data representation for the
    energy time series is used.
--energy_reshape (bool)
    If 'true' the resulting
    energy time series is also
    reshaped into a 24x7 matrix
    as presented in the paper.
--seed (int)
    Seed to set for the random
    functions in the pipeline.
    This ensures reproducibility.
```

The results of a pipeline run are printed in the command line and logged to [W&B](http://wandb.com/). In our pipeline we use mainly [scikit-learn](https://scikit-learn.org/stable/) to train linear models, [PyTorch](https://pytorch.org/) with [Lightning](https://www.pytorchlightning.ai/) to train neural networks, and [pyWATTS](https://github.com/KIT-IAI/pyWATTS) as a workflow engine.
