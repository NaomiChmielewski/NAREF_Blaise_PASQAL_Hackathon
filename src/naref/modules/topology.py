#!/usr/bin/env python
"""
Defines the auxiliary functions handling lattices
"""

import numpy as np


def grid_lattice(rows: int, columns: int, distance: float) -> dict:
	"""
	Returns a dictionary corresponding to a grid lattice.
	As the Pulser Devices limit the max distance from the origin, it is preferred to use grid_lattice_centred
	Args:
		- rows: nbr of rows
		- columns: nbr of columns
		- distance: distance between two atoms (nm)
	Output:
		- a dictionary that links the name of the atom with its position
	"""
	q_dict = {}
	for i in range(rows):
		for j in range(columns):
			q_dict["{}".format(i*columns + j)] = np.array([distance*i, distance*j])
	return q_dict


def grid_lattice_centred(dim: int, distance: float) -> dict:
	"""
	Returns a dictionary corresponding to a grid lattice whose center is in 0.
	Args:
			- rows: nbr of rows
			- columns: nbr of columns
			- distance: distance between two atoms (nm)
	Output:
			- a dictionary that links the name of the atom with its position
	"""
	offset = distance * (dim - 1) / 2

	q_dict = {}
	for i in range(dim):
		for j in range(dim):
			q_dict["{}".format(i * dim + j)] = np.array([distance * i, distance * j]) - offset

	return q_dict


def triangle_lattice(edge_size: int, distance: float) -> dict:
	"""
	Returns a dictionary corresponding to a lattice shaped as a triangle
	filled with triangles of same dimension.
	Args:
		- edge_size: nbr of atoms in the largest row
		- distance: distance between two atoms (nm)
	Output:
		- a dictionary that links the name of the atom with its position
	""" 
	q_dict, index = {}, 0
	for i in range(edge_size+1):
		for j in range(i):
			q_dict["{}".format(index)] = np.array([i*distance, j*distance])
			index += 1
	return q_dict


def circle_lattice(nbr_atoms: int, distance: float) -> dict:
	"""
	Returns a dictionary corresponding to a lattice shaped as a circle
	where each point is made of atoms.
	Args:
		- nbr_atoms: distance between two atoms
		- distance: distance between an atom and its nearest neighbours
	Output:
		- a dictionary that links the name of the atom with its position
	"""
	q_dict = {}
	angle_step = 2*np.pi/nbr_atoms
	radius = distance/(np.cos(angle_step)*angle_step)
	for i in range(nbr_atoms):
		q_dict["{}".format(i)] = np.array([radius*np.cos(i*angle_step), radius*np.sin(i*angle_step)])
	return q_dict
