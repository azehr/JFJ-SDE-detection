import numpy
import pandas

""" Metrics used to fit the AR-HMM """
def get_fitting_metrics(y_true, y_pred, y_low=None, y_high=None):
    squared_err = (y_true - y_pred) ** 2
    rmse = numpy.sqrt(numpy.mean(squared_err))
    mae = numpy.mean(numpy.abs(y_true - y_pred))
    mean_pred = numpy.mean(y_true) 
    nrmse_m = rmse / numpy.abs(mean_pred)
    nrmse_sd = rmse / numpy.std(y_true) 
    r2 = 1. - numpy.sum(squared_err) / numpy.sum((y_true - mean_pred) ** 2)
    if y_low is not None and y_high is not None:
        picp = get_picp(y_true, y_low, y_high)
        mpiw = get_mpiw(y_low, y_high)
    else:
        picp = numpy.nan
        mpiw = numpy.nan
    metric_df = pandas.DataFrame({'rmse': rmse, 'mae': mae, 'nrmse_m': nrmse_m, 'nrmse_sd': nrmse_sd,
                                  'r2': r2, 'picp': picp, 'mpiw': mpiw}, index=[0])
    return metric_df

def get_picp(y_true, y_low, y_high):
    # callibration, coverage probability of prediction interval, PI
    # if expected PI is 95%, picp should be as close as possible to 95
    return numpy.mean((y_low < y_true) & (y_true < y_high))

def get_mpiw(y_low, y_high):
    # mean width of prediction interval
    # should be as narrow as possible
    return numpy.mean(numpy.abs(y_high - y_low))




