#!/usr/bin/env python
"""
Defines various global variables
"""

# Default verbosity level for all Forecaster objects
VERBOSE = True

# Working directory for datasets
MAIN_PATH = "./results"

# Interval between prints when training
FEEDBACK_INTERVAL = 10

# Default random seed (for numpy, random, and reservoirpy naref.modules)
DEFAULT_SEED = 0

# Other examples : MockDevice, IroiseMVP, AnalogDevice
from pulser.devices import Chadoq2
PULSER_DEVICE = Chadoq2

# Other examples : mean_squared_log_error, ...
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ERROR_METRIC = mean_absolute_error
try:
	ERROR_NAME = ERROR_METRIC.__doc__.split('\n')[0][:-1]
except:
	ERROR_NAME = "Error"
# All errors considered for benchmarking.
ERR_BENCHMARK = {
	'mae': mean_absolute_error,
	'rmse': (lambda x, y: mean_squared_error(x, y) ** .5),
	'r2': r2_score}
