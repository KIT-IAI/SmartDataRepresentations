import numpy as np
import xarray as xr

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from pywatts.callbacks import PrintCallback
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import FunctionModule, Sampler, Slicer
from pywatts.modules.wrappers import FunctionModule, SKLearnWrapper
from pywatts.summaries.mae_summary import MAE
from pywatts.summaries.mape_summary import MAPE
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from src.features import calendar_features, differencing, reshaping
from src.pywatts.model_handler_module import Models
from src.pywatts import ModelHandler


def validate(x):
    """
    Perform validation checks before starting pipeline to ensure only
    valid data is processed.
    """
    nan_idx = np.where(np.isnan(x.values))[0]
    if len(nan_idx) > 0:
        print('Found nan values!')
        print(x.time[nan_idx])
        raise ValueError('Energy time-series containing nan values.')
    return x


def create_pipeline(hparams):
    """
    Set up pywatts pipeline to preprocess, train, predict and postprocess
    data for the energy forecasting use case and make evaluations.
    """
    pipeline = Pipeline(path='run')
    callbacks = [PrintCallback()] if hparams.debugging else []

    # check and clean nan values
    valid_data = FunctionModule(validate, name='Cleaning')(x=pipeline['energy'])

    # create differences if needed
    if hparams.energy_differences:
        difference_data = FunctionModule(
            lambda x: differencing(x, forecast_horizon=hparams.forecast_horizon),
            name='Differencing'
        )(x=valid_data)
    else:
        difference_data = valid_data

    # normalize
    normalizer = SKLearnWrapper(module=StandardScaler(), name='Normalizer')
    normalize_data = normalizer(x=difference_data)

    # generate output target
    output = Slicer(167 + hparams.forecast_horizon, name='Output')(x=normalize_data)

    # generate input targets (energy time series, calendar data)
    energy_features = Sampler(sample_size=168, name='Features')(x=normalize_data)
    energy_input = Slicer(167, -1 * hparams.forecast_horizon,
                          name='EnergyInput')(x=energy_features)
    calendar_input = FunctionModule(
        lambda x: calendar_features(x), name='CalendarInput'
    )(x=output)

    # reshape data representation
    if hparams.energy_reshape:
        reshaped_input = FunctionModule(
            lambda x: reshaping(x), name='Reshape'
        )(x=energy_input, callbacks=callbacks)
    else:
        reshaped_input = energy_input

    # train model
    if hparams.model == Models.LINEAR:
        forecast_normalized = SKLearnWrapper(module=LinearRegression(), name='LinearModel')(
            energy=reshaped_input, calendar=calendar_input, target_y=output)
    elif hparams.model in [Models.SIMPLE,  Models.MLP, Models.DNN]:
        forecast_normalized = ModelHandler(hparams=hparams, name='ModelHandler')(
            energy=reshaped_input, calendar=calendar_input, y=output)

    # evaluate
    forecast = normalizer(
        x=forecast_normalized,
        computation_mode=ComputationMode.Transform, use_inverse_transform=True
    )

    # revert differences
    if hparams.energy_differences:
        base = Slicer(167 + hparams.forecast_horizon, -1 * (hparams.forecast_horizon),
                      name='Base')(x=pipeline['energy'])
        forecast = FunctionModule(lambda a, b: numpy_to_xarray(
                                    a.values.flatten() + b.values.flatten(), a, 'Forecast'
                                  ), name='Forecast')(a=base, b=forecast, callbacks=callbacks)
        ground_truth = Slicer(167 + 2 * hparams.forecast_horizon,
                              name='GroundTruth')(x=pipeline['energy'], callbacks=callbacks)
    else:
        ground_truth = Slicer(167 + hparams.forecast_horizon, name='GroundTruth')(
            x=pipeline['energy'], callbacks=callbacks)
        forecast = FunctionModule(lambda x: numpy_to_xarray(
                                    x.values.flatten(), x, 'Forecast'
                                  ), name='Forecast')(x=forecast)

    MAE()(y=ground_truth, y_hat=forecast)
    MAPE()(y=ground_truth, y_hat=forecast)

    return pipeline
