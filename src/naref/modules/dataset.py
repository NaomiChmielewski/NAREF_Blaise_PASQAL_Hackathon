#!/usr/bin/env python
"""
Defines the default data (mackey, sine ...) and initial Forecaster parameters
"""
from reservoirpy.datasets import mackey_glass
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Dataset:
	# Choose input data: mackey glass ('mackey') or sine curve ('sine')
	input_type = "mackey"

	# MACKEY_GLASS
	# create a simple mackey_glass time series, rescaled between 0.2 and 1 for encoding
	mackey = mackey_glass(2000)
	scaler = MinMaxScaler(feature_range=(0.2, 1))
	scaler.fit(mackey)
	# we noticed that the typo on the following line meant that the rescaling was not applied in our calculations.
	# to make our results reproducible, we decided not to change this feature in the package.
	X = scaler.transform(mackey)

	# SINE
	sine = np.sin(np.linspace(0, 6 * np.pi, 600)).reshape(-1, 1)
	scaler = MinMaxScaler(feature_range=(0.2, 1))
	scaler.fit(sine)
	sine = scaler.transform(sine)

	# split input_type into train and test
	base_data = mackey if input_type == "mackey" else sine
	inp_train = base_data[:250]
	inp_test = base_data[250:280]

	# These are the default fixed hyperparameters
	# Use Comparator.hyper if you want to do an exhaustive hyperparameter search

	train_len = len(inp_train)
	test_len = len(inp_test)
	sample_len = 8

	# Not implemented yet : different values for N_samples, sample_len and reset_rate for train and test.

	# Classical Reservoir Parameters
	nb_neurons = 9

	# Quantum Reservoir Parameters
	# To quickly test that the code works, try 4 atoms.
	# To confirm our results, the hyperparameters are listed in the appendix  of our pdf document
	nb_atoms = 9

	inp_duration = 1000
	N_samples = 1024
	reset_rate = 0.
	geometry = "grid_lattice_centred"
	atom_distance = 15

	# For Hyperparameter search
	# Classical
	nb_neurons_list = [
		4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
	# Quantum
	sample_len_list = [2, 3, 6, 8, 9, 10]
	nb_atoms_list = [9]
	inp_duration_list = [16, 100, 1000]
	N_samples_list = [1024]
	reset_rate_list = [0.]
	geometry_list = ["grid_lattice_centred"]
	atom_distance_list = [4, 8, 11, 13, 14, 15, 20, 35]
	# Potentially usable for both the classical & quantum hyperparameter searches
	# Warning ! It is not recommended to change such parameters for the quantum hyp. search by default,
	# in order not to make a default search too long
	input_type_list = ["sine"]
	train_len_list = [train_len]
	test_len_list = [test_len]

	real_train_len = train_len - sample_len
	real_test_len = test_len - sample_len

	# Energy comparisons
	dataset_sizes = [1000 * i for i in range(1, 10 ** 4 + 1)]
	S = 10
	T_pulse = 1e-6
	N_runs = 52


data = Dataset()
