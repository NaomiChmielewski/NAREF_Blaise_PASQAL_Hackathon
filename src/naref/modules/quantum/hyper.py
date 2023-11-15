#!/usr/bin/env python
"""
Defines the HyperQ class, finding hyperparameters for the QRC class
"""
from naref.modules import abstract, data
from naref.modules.quantum import QRC


class HyperQ(abstract.Hyper):
	def __init__(
			self,
			sample_len_list=None,
			nb_atoms_list=None,
			N_samples_list=None,
			inp_duration_list=None,
			reset_rate_list=None,
			geometry_list=None,
			atom_distance_list=None,
			input_type_list=None,
			train_len_list=None,
			test_len_list=None,
			verbose: bool = True):
		super().__init__(sample_len_list, input_type_list, train_len_list, test_len_list, verbose)
		self.lists.update({
			"nb_atoms": data.nb_atoms_list if nb_atoms_list is None else nb_atoms_list,
			"N_samples": data.N_samples_list if N_samples_list is None else N_samples_list,
			"inp_duration": data.inp_duration_list if inp_duration_list is None else inp_duration_list,
			"reset_rate": data.reset_rate_list if reset_rate_list is None else reset_rate_list,
			"geometry": data.geometry_list if geometry_list is None else geometry_list,
			"atom_distance": data.atom_distance_list if atom_distance_list is None else atom_distance_list})

	def test_parameters(self, *tup) -> None:
		self.k += 1
		(
			sample_len, input_type, train_len, test_len, nb_atoms,
			inp_duration, N_samples, reset_rate, geometry, atom_distance
		) = tup
		real_train_len = train_len - sample_len
		real_test_len = test_len - sample_len

		self.current = QRC(
			nb_atoms=nb_atoms,
			geometry=geometry,
			atom_distance=atom_distance,
			N_samples=N_samples,
			sample_len=sample_len,
			reset_rate=reset_rate,
			inp_duration=inp_duration,
			input_type=input_type,
			test_len=real_test_len,
			train_len=real_train_len,
			verbose=self.verbose)
		self.save_parameters()
