#!/usr/bin/env python
"""
Defines the Comparator class
"""

from naref.modules import data, abstract, metrics, constants
from naref.modules.quantum import QRC
from naref.modules.classical import LinReg, CRC
from collections import defaultdict
from matplotlib import pyplot as plt


class ErrorComparator:

	def __init__(self, active: list = None, auto_train: bool = True):
		"""
		Provides a framework comparing all available Forecasters for all available metrics.
		Args:
			active: list of Forecasters that we want to study
			auto_train: whether to automatically train active Forecasters. Set to false if only interested in energy.
		"""

		self.active = ['CRC', 'LinReg', 'QRC'] if active is None else active  # Forecasters that we want to study
		self.frc = defaultdict(
			abstract.Forecaster,
			{
				'QRC': QRC(
					sample_len=data.sample_len,
					nb_atoms=data.nb_atoms,
					N_samples=data.N_samples,
					inp_duration=data.inp_duration,
					reset_rate=data.reset_rate,
					geometry=data.geometry,
					atom_distance=data.atom_distance,
					input_type=data.input_type,
					test_len=data.real_test_len,
					train_len=data.real_train_len,
					verbose=True),
				'LinReg': LinReg(),
				'CRC': CRC(
					sample_len=data.sample_len,
					nb_neurons=data.nb_neurons)})
		if auto_train:
			self.train()

		# Outer Accessibility
		# Quantum Reservoir
		self.qrc: QRC = self.frc['QRC']
		# Linear Regression
		self.lrg: LinReg = self.frc['LinReg']
		# Classical Reservoir
		self.crc: CRC = self.frc['CRC']

	def train(self) -> None:
		"""
		Builds the Forecaster's attributes according to contents of the data module
		"""
		for forecaster in self.active:
			self.frc[forecaster].build_model(data.inp_train, data.inp_test)

	def compare(self) -> None:
		"""
		Prints errors for all active forecasters and displays their forecast as pyplot curves
		"""
		for forecaster in self.active:
			print("Train error:", self.frc[forecaster].get_error_train())
			self.frc[forecaster].show_train_prediction()
		for forecaster in self.active:
			print("Test error:", self.frc[forecaster].get_error_test())
			self.frc[forecaster].show_test_prediction()


class Comparator(ErrorComparator):

	def __init__(self, dataset_sizes: list = None, *a, **k):
		super().__init__(*a, **k)
		# self.dataset_sizes corresponds to Tilde_L
		self.dataset_sizes = data.dataset_sizes if dataset_sizes is None else dataset_sizes
		self.S = k["S"] if "S" in k.keys() else data.S
		self.T_pulse = k["T_pulse"] if "T_pulse" in k.keys() else data.T_pulse
		self.N_runs = k["N_runs"] if "N_runs" in k.keys() else data.N_runs

		# Default energy lists
		self.E_Fresnel = []
		self.E_Rubi = []
		self.E_GPU = []
		self.E_Joliot = []

	def qtime(self, L: int) -> float:
		"""
		We precisely calculated this value with information given by PASQAL (refresh rate, etc.)
		Args:
			L: dataset size
		Returns:
			Multiplicative coefficient to get the consumption of a quantum device for this dataset size
		"""
		return self.N_runs * 2 ** 5 * (metrics.fct_L(L) - metrics.fct_S(L)) * (1 + metrics.fct_S(L) * self.T_pulse) / 3600

	def rtime(self, L: int) -> float:
		"""
		We could divide by 10 the value of qtime, as it is what is done by default in the Excel file and as this
		computer has a cryostat, but we cannot know for sure. As in the .pdf file, we set it equal to the qtime.
		Args:
			L: dataset size
		Returns:
			Multiplicative coefficient to get the consumption of a quantum device for this dataset size
		"""
		return self.qtime(L)

	def ctime(self, L: int) -> float:
		"""
		Args:
			L: dataset size
		Returns:
			Multiplicative coefficient to get the consumption of a classical GPU device for this dataset size
		"""
		return self.N_runs * (2.496 * 1e5 * L / 10) / metrics.f_GPU * L * (L / 100) / 3600

	def jtime(self, L: int) -> float:
		"""
		Multiplicative coefficient to get the consumption of a typical classical HPC device (Joliot)
		for this dataset size
		Args:
			L: dataset size
		Returns:
			the multiplicative coefficient
		"""
		return self.N_runs * (2.496 * 1e5 * L / 10) / metrics.f_Joliot * L * (L / 100) / 3600

	def energy(self):
		"""
		Computes the energy consumption for each device and each potential dataset size
		And then plots it
		"""
		self.compute_energy()
		self.plot_energy()

	def compute_energy(self):
		"""
		Computes the energy consumption for each device and each potential dataset size
		"""
		print("Computing Fresnel Energy Consumption")
		self.E_Fresnel = [metrics.C_Fresnel * self.qtime(L) for L in self.dataset_sizes]
		print("Computing Rubi Energy Consumption")
		self.E_Rubi = [metrics.C_Rubi * self.rtime(L) for L in self.dataset_sizes]
		print("Computing GPU Energy Consumption")
		self.E_GPU = [metrics.C_GPU * self.ctime(L) for L in self.dataset_sizes]
		print("Computing Joliot Energy Consumption")
		self.E_Joliot = [metrics.C_Joliot * self.jtime(L) for L in self.dataset_sizes]

	def plot_energy(self):
		"""
		Plots the energy that has been computed in compute_energy
		"""
		plt.rcParams.update({'font.size': 22})
		plt.figure(figsize=(15, 6.5))
		plt.title("Evolution of Carbon Emission for Time Series Forecasting \n as Data Size increases")
		plt.xlabel("Size of the dataset")
		plt.ylabel("tCO2 eq")
		plt.xscale("log")
		plt.yscale("log")
		plt.plot(self.dataset_sizes, self.E_Fresnel, label="QRC on Fresnel", color='blue')
		plt.plot(self.dataset_sizes, self.E_Rubi, label="QRC on Rubi", color='yellow')
		plt.plot(self.dataset_sizes, self.E_GPU, label="RNN on GPU", color='green')
		plt.plot(self.dataset_sizes, self.E_Joliot, label="RNN on Joliot-Curie (HPC)", color='red')
		# plt.xticks(self.dataset_sizes)
		plt.legend()
		plt.savefig(f"{constants.MAIN_PATH}/Emission_Comparison_RNN.pdf")
		plt.show()

	def emissions_duration(self):
		"""
		Considering the same values as before, we can derive the different method times and their carbon emissions
		for a typical dataset size
		"""
		L = 350*10**3

		E_Fresnel = metrics.C_Fresnel * self.qtime(L)
		E_Rubi = metrics.C_Rubi * self.rtime(L)
		E_GPU = metrics.C_GPU * self.ctime(L)
		E_Joliot = metrics.C_Joliot * self.jtime(L)

		T_Fresnel = self.qtime(L)
		T_Rubi = self.rtime(L)
		T_GPU = self.ctime(L)
		T_Joliot = self.jtime(L)

		print("Emissions from Fresnel = {} eq tCO2             Duration = {} h".format(E_Fresnel, T_Fresnel))
		print("Emissions from Rubi = {} eq tCO2                Duration = {} h".format(E_Rubi, T_Rubi))
		print("Emissions from GPU = {} eq tCO2 /               Duration = {} h".format(E_GPU, T_GPU))
		print("Emissions from Joliot-Curie HPC = {} eq tCO2     Duration = {} h".format(E_Joliot, T_Joliot))

