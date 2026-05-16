"""Topological data analysis functions."""

import numpy as np

# import kmapper as km
from ripser import ripser


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


class DGMS_loader:
	"""Class to load persistence diagrams from CSV files."""

	def __init__(self, path_csv):
		self.path_csv = path_csv

	def export(self, dgms, path_csv=None):
		import pandas as pd

		rows = []
		for dim, dgm in enumerate(dgms):
			for birth, death in dgm:
				rows.append({"dimension": dim, "birth": birth, "death": death})
		df = pd.DataFrame(rows)
		if path_csv is None:
			path_csv = self.path_csv
		df.to_csv(path_csv, index=False)

	def import_(self):
		import pandas as pd

		df = pd.read_csv(self.path_csv)
		dgms = []
		for dim in sorted(df["dimension"].unique()):
			dgm_dim = df[df["dimension"] == dim][["birth", "death"]].values
			dgms.append(dgm_dim)
		return dgms
