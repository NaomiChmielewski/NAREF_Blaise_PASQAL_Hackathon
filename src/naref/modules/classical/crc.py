#!/usr/bin/env python
"""
Defines the CRC and LinearRegression classes (classical TSP)
"""
from sklearn.linear_model import LinearRegression
from naref.modules import abstract, constants, data
import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here


def warp_dataset(inp_train: list, inp_test: list) -> (list, list, list, list):
	"""
	Transforms the dataset to fit its own input style in build_baseline_model
	Args:
		inp_train: input train datasest (a temporal serie)
		inp_test: input test datasest (a temporal serie)

	Returns:
		the 4-tuple :
		(x_train = inp_train[:-1]
		y_train = inp_train[1:]
		x_test = inp_test[:-1]
		y_test = inp_test[1:])
	"""
	return inp_train[:-1], inp_train[1:], inp_test[:-1], inp_test[1:]


class LinReg(abstract.Forecaster):

	def __init__(self, verbose: bool = constants.VERBOSE):
		"""
		Class containing Linear Regression model. It is used as the baseline model for comparisons.
		"""
		super().__init__(verbose=verbose)
		self.name: str = "LinReg"
		self.longname: str = "Linear Regression"
		self.model = LinearRegression()  # will contain the last fit to training points

	def build_model(self, inp_train: list, inp_test: list) -> list:
		"""
		Updates the inner model and returns the predicted test values after fitting on the train
		Args:
			inp_train: input train datasest (a temporal serie)
			inp_test: input test datasest (a temporal serie)

		Returns:
			the predicted test values after fitting on the train ( y_pred_test )
		"""
		self.announce_build()
		return self._build_model(*warp_dataset(inp_train, inp_test))

	def _build_model(self, x_train: list, y_train: list, x_test: list, y_test: list) -> list:
		"""
		Updates the inner model and returns the predicted test values after fitting on the train
		Args:
			x_train: training input data
			y_train: training correct output data
			x_test: testing input data
			y_test: testing correct output data

		Returns:
			the predicted test values after fitting on the train ( y_pred_test )
		"""
		self.model.fit(x_train, y_train)
		self.y_data_train: list = y_train
		self.y_pred_train: list = self.model.predict(x_train)
		self.y_data_test: list = y_test
		self.y_pred_test: list = self.model.predict(x_test)
		return self.y_pred_test


class CRC(abstract.Forecaster):
	header = ['sample_len', 'nb_neurons', 'input_type', 'train_len', 'test_len', 'train_mae', 'test_mae']

	def __init__(
			self,
			sample_len: int = data.sample_len,
			nb_neurons: int = data.nb_neurons,
			input_type: str = data.input_type,
			train_len: int = data.train_len,
			test_len: int = data.test_len,
			verbose: bool = constants.VERBOSE):
		super().__init__(input_type, train_len, test_len, verbose)
		self.name: str = "CRC"
		self.longname: str = "Classical Reservoir"
		self.sample_len: int = sample_len
		self.nb_neurons: int = nb_neurons
		self.input_type: str = input_type  # Clue on the type of data that will be used to build_model
		self.train_len: int = train_len  # Will contain the real_train_length
		self.test_len: int = test_len  # Will contain the real_test_length

		# reservoir generation
		rpy.set_seed(constants.DEFAULT_SEED)
		reservoir = rpy.nodes.Reservoir(self.nb_neurons, lr=1, sr=0.9)  # Main part of the CRC
		ridge = rpy.nodes.Ridge(ridge=1e-7)  # A single layer of neurons learning with Tikhonov linear regression.
		self.model = reservoir >> ridge

	def build_model(self, inp_train: list, inp_test: list) -> list:
		"""
		Updates the inner model and returns the predicted test values after fitting on the train
		Args:
			inp_train: input train datasest (a temporal serie)
			inp_test: input test datasest (a temporal serie)

		Returns:
			the predicted test values after fitting on the train ( y_pred_test )
		"""
		self.announce_build()
		self.train_len: int = int(len(inp_train) - self.sample_len)
		self.test_len: int = int(len(inp_test) - self.sample_len)
		return self._build_model(*warp_dataset(inp_train, inp_test))

	def _build_model(self, x_train: list, y_train: list, x_test: list, y_test: list) -> list:
		"""
		Updates the inner model and returns the predicted test values after fitting on the train
		Args:
			x_train: training input data (non-truncated)
			y_train: training correct output data
			x_test: testing input data (non-truncated)
			y_test: testing correct output data

		Returns:
			the predicted test values after fitting on the train ( y_pred_test )
		"""
		x_train = x_train[self.sample_len:self.sample_len + self.train_len]
		x_test = x_test[self.sample_len:self.sample_len + self.test_len]

		self.y_data_train: list = y_train[self.sample_len:self.sample_len + self.train_len]
		self.y_data_test: list = y_test[self.sample_len:self.sample_len + self.test_len]

		self.model = self.model.fit(x_train, self.y_data_train, warmup=0)
		self.y_pred_train: list = self.model.run(x_train)
		self.y_pred_test: list = self.model.run(x_test)
		return self.y_pred_test

	def save_data(self, overwrite: bool = False, *a) -> None:
		"""
		Saves the contents of the qrc to a csv file

		Args:
			overwrite: whether to overwrite the file where we would write data, or just append at the end
			other arguments are ignored
		"""
		dic = {
			'sample_len': self.sample_len,
			'nb_neurons': self.nb_neurons,
			'input_type': self.input_type,
			'train_len': self.train_len,
			'test_len': self.test_len,
			'train_mae': self.get_error_train(),
			'test_mae': self.get_error_test()
		}
		super().save_data(overwrite, dic, f'{constants.MAIN_PATH}/classical_grid.csv', self.header)
