#!/usr/bin/env python
"""
Defines the auxiliary functions handling comparison methods
"""
from naref.modules import constants
from matplotlib import pyplot as plt


def errors(vals: iter, preds: iter):
	"""
	Typically, ERR_BENCHMARK contains 'mae' 'rmse' and 'r2' errors.
	Args:
		vals: an iterable of floats
		preds: an iterable of floats

	Returns:
		a dictionary containing errors obtained when comparing vals & preds (for all errors in ERR_BENCHMARK)
	"""

	return {k: v(vals, preds) for k,v in constants.ERR_BENCHMARK.items()}


# Emission costs (tCO2 eq/hour)
C_Fresnel = (2.1*3.6)/(1000*2.1)
C_Rubi = (4.2*3)/(1000*0.4)
C_GPU = 0.09/1000
C_Joliot = 215/1000
# Required computation time as specified in the given Excel document
T_QPU, T_GPU, T_Joliot = 458, 884000, 67
# Classical processing unit frequencies
f_Joliot, f_GPU = 7 * 1e15, 5 * 1e11


def print_former_benchmarks():
	"""
	From PASQAL data, we noted above the emissions of each type of device in tCO2 eq/hour.
	We can print that it is coherent with the given Excel document
	"""
	print("Emissions from Fresnel = {} eq tCO2".format(T_QPU*C_Fresnel))
	print("Emissions from Rubi = {} eq tCO2".format(T_QPU*C_Rubi))
	print("Emissions from GPU cluster = {} eq tCO2".format(T_GPU*C_GPU))
	print("Emissions from Joliot Curie = {} eq tCO2".format(T_Joliot*C_Joliot))


# useful functions for energy metrics
def fct_L(L_tilde):
	return L_tilde ** (1 / 3)


def fct_S(L):
	return fct_L(L) ** 0.5


def show_prediction(pred: list, real: list, phase: str, longname: str) -> None:
	"""
	Shows how close are prea and real, plottable data
	"""
	plt.figure(figsize=(10, 3))
	plt.title(f"{longname} {phase} Prediction")
	plt.xlabel("$t$")
	plt.plot(pred, label="Predicted", color="blue")
	plt.plot(real, label="Real", color="red")
	plt.legend()
	plt.show()
