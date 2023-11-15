#!/usr/bin/env python
"""
Defines the QRC class and its abstract classes
"""

from naref.modules import constants, topology, abstract, data
import numpy as np
from numpy import random as rd
import pandas as pd
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
import time
import warnings

rd.seed(constants.DEFAULT_SEED)
warnings.filterwarnings('ignore')


class QRCTrainer(abstract.Forecaster):

	def __init__(
			self,
			sample_len: int = data.sample_len,
			nb_atoms: int = data.nb_atoms,
			N_samples: int = data.N_samples,
			inp_duration: int = data.inp_duration,
			reset_rate: float = data.reset_rate,
			geometry: str = data.geometry,
			atom_distance: float = data.atom_distance,
			input_type: str = data.input_type,
			train_len: int = data.train_len,
			test_len: int = data.test_len,
			verbose: bool = constants.VERBOSE):
		super().__init__(input_type, train_len, test_len, verbose)
		self.name = "QRC"
		self.longname = "Quantum Reservoir"
		self.inp_duration: int = inp_duration  # the reservoir will see each input for this pulse duration (in ns)
		self.N_samples: int = N_samples  # number of shots
		self.sample_len: int = sample_len  # length of input time series
		self.reset_rate: float = reset_rate  # probability of resetting the whole reservoir (for ESP)
		self.train_data: list = []  # contains the latest provided training data
		self.rese_data: list = []  # contains the training data of the reservoir
		self.wlist: list = []  # list of input training data (truncated to fit a window)
		self.clist: list = []  # counts (of sample of window) list
		self.nclist: list = []  # normalised counts (of sample of window) list
		self.ncdf: pd.DataFrame = None  # normalised counts Panda DataFrame, filled with 0 where was NaN
		self.X_all_train: np.ndarray = None  # array version of the above. We compute w by solving w = X^-1 * data.
		self.w: np.ndarray = None  # Reservoir output weights.

		self.n: int = nb_atoms
		self.geom: str = geometry
		self.atom_distance: float = atom_distance
		self.seq: Sequence = None

		self.__k_time: float = 0  # Time since the last print of training time
		self.scope: str = "training"  # Current action

	def __reset_seq(self) -> None:
		""" Sets self.set to an initial, blank sequence, with the right layout, device and channels """
		self.seq = Sequence(self.layout(), constants.PULSER_DEVICE)
		self.seq.declare_channel("global", "rydberg_global")

	def layout(self) -> Register:
		""" Returns the right Register object associated with the lattice of geometry 'self.geom' """

		if self.geom == 'grid_lattice':
			q_dict = topology.grid_lattice(int(np.sqrt(self.n)), int(np.sqrt(self.n)), self.atom_distance)
		elif self.geom == 'grid_lattice_centred':
			q_dict = topology.grid_lattice_centred(int(np.sqrt(self.n)), self.atom_distance)
		elif self.geom == 'circle_lattice':
			q_dict = topology.circle_lattice(self.n, self.atom_distance)
		elif self.geom == 'triangle_lattice':
			q_dict = topology.triangle_lattice(int(-1 / 2 + np.sqrt(1 / 4 + 2 * self.n)), self.atom_distance)
		else:
			raise Exception("Unknown Lattice Type")

		reg = Register(q_dict)
		return reg

	def build_reservoir(self, inp_trunc: list, scope: str = "training") -> list:
		"""
		Encodes input_trunc as input parameters in local pulses and constructs the reservoir from the encoding.
		Returns all measurement outcomes associated with the truncated input

		The input is here called truncated, as typically, the total dataset is divided into training and test set.
		Each can therefore be considered as 'truncated'.

		Args:
			inp_trunc: truncated input (cf above)
			scope: whether to print that we are in the training or testing phase

		Returns:
			the list [lists of the samples of Z_MEASUREMENT( R(window) ) (for each window)]
		"""

		self.rese_data = inp_trunc
		self.scope = scope

		# total time since the last constants.FEEDBACK_INTERVAL steps
		self.__k_time = 0

		# number of available windows
		nwin = len(self.rese_data) - self.sample_len

		print(f"Computing {nwin} training steps...")

		# list of outcomes for each window (uK ...uk+tau)
		return [self.window(k) for k in range(nwin)]

	def window(self, k: int) -> dict:
		"""
		For each window (uK ...uK+tau) during the execution,
		we have to do the sequence of pulses corresponding to R(windowK, rhoK)

		Args:
			k: index of the window (uK ...uK+tau)

		Returns:
			the dict of the samples of Z_MEASUREMENT( R(window) )
		"""

		if self.verbose:
			tic = time.time()  # measure the time spent in this window

		# initialize Sequence self.seq
		self.__reset_seq()

		# extract subsection of time series to pass into the reservoir
		self.wlist = self.rese_data[k:k + self.sample_len]

		# for each ui in (uK ...uK+tau)
		for ui in range(self.sample_len):
			self.add_to_sequence(ui)

		sim = QutipEmulator.from_sequence(self.seq, sampling_rate=1)
		results = sim.run(progress_bar=False)  # run the simulation
		outcomes = results.sample_final_state(N_samples=self.N_samples)

		if self.verbose:
			toc = time.time()
			self.__k_time += (toc - tic)
			if k % 10 == 9:  # as k starts at 0, this ensures 10 steps have passed before the first feedback
				print(f"{k + 1} {self.scope} steps finished, {self.scope} time : {self.__k_time:.3f}s")
				self.__k_time = 0

		return outcomes

	def add_to_sequence(self, i: int) -> None:
		"""
		For each ui in a given window (uK ...uK+tau), we have to add either a corresponding pulse or a reset.
		Args:
			i: index of the input under scrutiny
		"""
		duration = self.inp_duration  # duration for which the reservoir will see the input = pulse duration (in ns)

		reset = rd.choice(
			np.arange(0, 2),
			p=[1 - self.reset_rate, self.reset_rate],
			size=1)  # in case we want to add random noise

		if reset:
			print("reset")
			self.__reset_seq()

		else:
			amplitude = self.wlist[i]  # input is encoded in the pulse amplitude
			detuning = 0

			# It seems that Pulser sometimes messes up the random seed at this point.
			if np.isclose(self.reset_rate, 0):
				# If we do not need randomness at this point, we reinitialise the seed.
				rd.seed(constants.DEFAULT_SEED)

			pulse = Pulse.ConstantPulse(duration, amplitude, detuning, phase=0.00, post_phase_shift=0.0)
			self.seq.add(pulse, "global")

	def train(self, data: list, train_len: int) -> float:
		"""
		Executes the training for all inputs in data.
		Fills in self.train_data.
		Constructs the optimised weights from the training.
		Returns and displays the final total error (train error).

		Args:
			data: input data
			train_len: the number of training steps
		"""
		self.train_data = data[:self.sample_len + train_len]

		total_time = 0
		tic = time.time()
		self.clist = self.build_reservoir(self.train_data)

		# normalise the counts list so that it is coherent between train and test
		self._fill_nclist(self.clist)

		# The following link justifies using from_dict for a list of dict objects
		# www.geeksforgeeks.org/create-a-pandas-dataframe-from-list-of-dicts/
		self.ncdf = pd.DataFrame.from_dict(self.nclist).fillna(0)
		self.X_all_train = self.ncdf.to_numpy()

		# calculate pseudo inverse
		xp = np.linalg.pinv(self.X_all_train)
		# calculate the weight vector (linear regression)
		self.w = np.dot(xp, self.train_data[self.sample_len:self.sample_len + len(self.clist)])

		toc = time.time()
		total_time += (toc - tic)
		print(f"Training finished. Total training time: {total_time:.3f}s")

		self.y_data_train = self.train_data[self.sample_len:self.sample_len + train_len]
		self.y_pred_train = self._get_prediction(self.X_all_train, len(self.clist))
		error_train: float = self.get_error_train()
		print(f"Train error ({constants.ERROR_NAME}): ", error_train)
		return error_train

	def _get_prediction(self, X_all: np.ndarray, c_len: int) -> list:
		"""
		Args:
			X_all: array version of normalized counts (after pd filling the NaN)
			c_len: length of the list of counts

		Returns:
			List of the predictions associated with the reservoir weights w
		"""
		return [np.dot(X_all[c, :], self.w) for c in range(c_len)]

	def _fill_nclist(self, counts: list) -> None:
		"""
		Normalizes the counts list so that it is coherent between the testing and training phases.
		Args:
			counts: list of counts (of sample of window), either from training or testing phase.
		"""
		self.nclist = [{k: v / self.N_samples for k, v in c.items()} for c in counts]


class QRCTester(QRCTrainer):

	def __init__(
			self,
			sample_len: int = data.sample_len,
			nb_atoms: int = data.nb_atoms,
			N_samples: int = data.N_samples,
			inp_duration: int = data.inp_duration,
			reset_rate: float = data.reset_rate,
			geometry: str = data.geometry,
			atom_distance: float = data.atom_distance,
			input_type: str = data.input_type,
			train_len: int = data.train_len,
			test_len: int = data.test_len,
			verbose: bool = constants.VERBOSE):
		super().__init__(
			sample_len, nb_atoms, N_samples, inp_duration, reset_rate, geometry, atom_distance,
			input_type, train_len, test_len, verbose)
		self.test_data: list = []  # contains the latest provided test data
		self.clist_test: list = []  # list of counts (of sample of window) from the testing phase
		self.X_all_test: np.ndarray = None  # array computing ...

	def evaluate(self, test_data: list, test_len: int) -> float:
		"""
		Builds the reservoirs associated with the test data. Returns the observed hidden states and the predictions
		Args:
			test_data: input data
			test_len: the number of testing steps
		"""

		self.test_data = test_data[:self.sample_len + test_len]
		self.clist_test = self.build_reservoir(self.test_data, scope="testing")

		# normalise the counts list so that it is coherent between train and test
		self._fill_nclist(self.clist_test)

		df_copy = self.ncdf.iloc[:0, :].copy().fillna(0)
		df_copy = df_copy.append(self.nclist, ignore_index=True, sort=False).fillna(0)
		self.X_all_test = df_copy.to_numpy()

		# evaluate testing error
		self.y_pred_test = self._get_prediction(self.X_all_test, len(self.clist_test))
		self.y_data_test = self.test_data[self.sample_len:self.sample_len + test_len]

		error_test: float = self.get_error_test()
		print(f"Test error ({constants.ERROR_NAME}): ", error_test)
		return error_test


class QRC(QRCTester):
	header = [
		'input_type', 'train_len', 'test_len', 'inp_duration', 'N_samples', 'sample_len',
		'reset_rate', 'nb_atoms', 'geometry', 'atom_distance', 'train_mae', 'test_mae']

	def __init__(
			self,
			sample_len: int = data.sample_len,
			nb_atoms: int = data.nb_atoms,
			N_samples: int = data.N_samples,
			inp_duration: int = data.inp_duration,
			reset_rate: float = data.reset_rate,
			geometry: str = data.geometry,
			atom_distance: float = data.atom_distance,
			input_type: str = data.input_type,
			train_len: int = data.train_len,
			test_len: int = data.test_len,
			verbose: bool = constants.VERBOSE):
		super().__init__(
			sample_len, nb_atoms, N_samples, inp_duration, reset_rate, geometry, atom_distance,
			input_type, train_len, test_len, verbose)

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
		self.train(inp_train, train_len=self.train_len)
		self.evaluate(inp_test, test_len=self.test_len)
		return self.y_pred_test

	def save_data(self, overwrite: bool = False, *a) -> None:
		"""
		Saves the contents of the qrc to a csv file

		Args:
			overwrite: whether to overwrite the file where we would write data, or just append at the end
			other arguments are ignored
		"""
		dic = {
			'inp_duration': self.inp_duration,
			'N_samples': self.N_samples,
			'sample_len': self.sample_len,
			'reset_rate': self.reset_rate,
			'nb_atoms': self.n,
			'geometry': self.geom,
			'atom_distance': self.atom_distance,
			'input_type': self.input_type,
			'train_len': self.train_len,
			'test_len': self.test_len,
			'train_mae': self.get_error_train(),
			'test_mae': self.get_error_test()}
		fname = f'{constants.MAIN_PATH}/{self.n}_{self.input_type}_{self.geom}.csv'
		super().save_data(overwrite, dic, fname, self.header)
