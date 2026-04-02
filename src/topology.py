"""Topological data analysis functions."""

import numpy as np
import matplotlib.pyplot as plt

# import kmapper as km
from ripser import ripser
from sklearn.cluster import DBSCAN


def compute_persistence(distance_matrix, maxdim=2, thresh=np.inf, coeff=2):
	result = ripser(
		distance_matrix, maxdim=maxdim, thresh=thresh, distance_matrix=True, coeff=coeff
	)
	return result["dgms"]


def visualize_mapper_graph(
	mapper,
	graph,
	lens,
	path_html="mapper_graph.html",
	title="Mapper de S² usando coordenada z",
):
	"""Persist the mapper visualization that colors nodes by the filter values."""
	mapper.visualize(
		graph,
		path_html=path_html,
		title=title,
		color_function=lens.flatten(),
		color_function_name="Coordenada z",
	)
