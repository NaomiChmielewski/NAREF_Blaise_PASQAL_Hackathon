#!/usr/bin/env python
"""
Defines the HyperC class, finding hyperparameters for the CRC class
"""
from naref.modules import abstract, data
from naref.modules.classical import CRC


class HyperC(abstract.Hyper):
	def __init__(
			self,
			sample_len_list=None,
			nb_neurons_list=None,
			input_type_list=None,
			train_len_list=None,
			test_len_list=None,
			verbose: bool = True):
		super().__init__(sample_len_list, input_type_list, train_len_list, test_len_list, verbose)
		self.lists.update({
			"nb_neurons": data.nb_neurons_list if nb_neurons_list is None else nb_neurons_list})

	def test_parameters(self, *tup) -> None:
		self.k += 1
		sample_len, input_type, train_len, test_len,  nb_neurons = tup
		real_train_len = train_len - sample_len
		real_test_len = test_len - sample_len

		self.current = CRC(
			nb_neurons=nb_neurons,
			sample_len=sample_len,
			input_type=input_type,
			test_len=real_test_len,
			train_len=real_train_len,
			verbose=self.verbose)
		self.save_parameters()
