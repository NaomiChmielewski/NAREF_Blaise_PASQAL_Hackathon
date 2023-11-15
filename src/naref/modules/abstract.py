#!/usr/bin/env python
"""
Defines the abstract classes Forecaster and Hyper
"""
from naref.modules import metrics, constants, data
from math import prod
from itertools import product
from collections import defaultdict
import csv


class Forecaster:
	"""
	Abstract class containing functions implemented by QRC, CRC, LinReg classes
	"""
	header = []

	def __init__(
			self,
			input_type: str = 'unknown',
			train_len: int = -1,
			test_len: int = -1,
			verbose: bool = constants.VERBOSE):
		self.name = "AFR"  # String identifier of the Forecaster type
		self.longname = "Abstract Forecaster"  # String explicit name of the Forecaster type
		self.y_data_train: list = []  # Contains each predictable point of the train TS (all but the sample_len first)
		self.y_pred_train: list = []  # Filled with the predicted TS after the training. Ideally equal to y_data_train.
		self.y_data_test: list = []  # Contains each predictable point of the test TS (all but the sample_len first)
		self.y_pred_test: list = []  # Filled with the predicted TS after the training. Ideally equal to y_data_test.
		self.rmtrain: dict = {}  # result of metrics for the training phase
		self.rmtest: dict = {}  # result of metrics for the testing phase
		self.input_type: str = input_type  # Clue on the type of data that will be used to build_model
		self.train_len: int = train_len  # Will contain the real_train_length
		self.test_len: int = test_len  # Will contain the real_test_length
		self.verbose: bool = verbose  # verbosity level

	def get_scores(self) -> None:
		"""
		Calculate train and test score of the quantum reservoir and updates rmtest & rmtrain
		"""
		self.rmtrain = metrics.errors(self.y_pred_train, self.y_data_train)
		self.rmtest = metrics.errors(self.y_pred_test, self.y_data_test)

		print(f"{self.name} Train: {self.rmtrain}")
		print(f"{self.name} Test: {self.rmtest}")

	def announce_build(self) -> None:
		"""
		Displays a message stating that we are building the model.
		"""
		if self.verbose:
			print(f"Building {self.longname}...")

	def get_error_train(self) -> float:
		"""
		Evaluates the training error.
		None of the variables used are modified by the testing phase.
		"""
		try:
			return constants.ERROR_METRIC(self.y_data_train, self.y_pred_train)
		except ValueError:
			raise Exception("Forecaster not built. Use the .build_model method before getting the error.")

	def get_error_test(self) -> float:
		"""
		Evaluates the testing error.
		"""
		try:
			return constants.ERROR_METRIC(self.y_data_test, self.y_pred_test)
		except ValueError:
			raise Exception("Forecaster not built. Use the .build_model method before getting the error.")

	def show_test_prediction(self) -> None:
		"""
		Shows how close are self.y_pred_test and self.y_data_test
		"""
		metrics.show_prediction(self.y_pred_test, self.y_data_test, "Test", self.longname)

	def show_train_prediction(self) -> None:
		"""
		Shows how close are self.y_pred_train and self.y_data_train
		"""
		metrics.show_prediction(self.y_pred_train, self.y_data_train, "Train", self.longname)

	def build_model(self, inp_train: list, inp_test: list) -> list:
		"""
		Updates the inner model and returns the predicted test values after fitting on the train
		Args:
			inp_train: input train datasest (a temporal serie)
			inp_test: input test datasest (a temporal serie)

		Returns:
			the predicted test values after fitting on the train ( y_pred_test )
		"""
		return self.y_pred_test

	def save_data(self, overwrite: bool = False, dic: dict = None, fname: str = 'default', header: list = None):
		"""
			Saves the contents of the qrc to a csv file

			Args:
				overwrite: whether to overwrite the file where we would write data, or just append at the end
				dic: dictionnary containing the contents of a row of data, corresponding to the header
				fname: file name
				header: names of the columns
		"""
		header = header if header is not None else self.header
		dic = dic if dic is not None else {}
		with open(fname, 'w' if overwrite else 'a', encoding='UTF8') as f:
			if overwrite:
				writer = csv.writer(f)
				writer.writerow(header)
			dictwriter_object = csv.DictWriter(f, fieldnames=header)
			dictwriter_object.writerow(dic)
			f.close()


class Hyper:
	"""
	Abstract hyperparameter search class containing functions implemented by HyperC and HyperQ
	"""

	def __init__(
			self,
			sample_len_list=None,
			input_type_list=None,
			train_len_list=None,
			test_len_list=None,
			verbose: bool = constants.VERBOSE):
		self.lists = defaultdict(list, {
			"sample_len": data.sample_len_list if sample_len_list is None else sample_len_list,
			"input_type": data.input_type_list if input_type_list is None else input_type_list,
			"train_len": data.train_len_list if train_len_list is None else train_len_list,
			"test_len": data.test_len_list if test_len_list is None else test_len_list})
		self.k: int = 0
		self.tot_combin: int = 0  # Total number of possible hyperparameters combinations
		self.current: Forecaster = None  # Currently optimised Forecaster
		self.verbose: bool = verbose

	def search(self):
		"""
		Iterates on all possible combinations of parameters in self.lists and adds a line to the relevant csv for each

		Returns:
			all results in a csv file
		"""
		self.k = 0
		self.tot_combin = prod(len(x) for x in self.lists.values())
		print(f"{self.__class__.__name__} search within {self.tot_combin} potential sets of parameters")
		para = product(*self.lists.values())
		for tup in para:
			self.test_parameters(*tup)
		self.print("Hyperparameter search complete")

	def test_parameters(self, *tup):
		"""
		Tests a set of parameters by updating self.current and calling save_parameters
		Args:
			*tup: iterable of parameters

		Returns:
			all results in a csv file
		"""
		raise Exception(f'{self.__class__.__name__} does not implement the test_parameters method.')

	def save_parameters(self):
		"""
		Builds self.current and then appends relevant data in a file
		Returns:
			all results in a csv file
		"""
		self.current.build_model(data.inp_train, data.inp_test)
		self.print(f"{self.k} out of {self.tot_combin}")
		self.current.save_data(self.k < 2)

	def print(self, *a, **k):
		"""
		Wrapper for the print function taking verbosity into account
		"""
		if self.verbose:
			print(*a, **k)
