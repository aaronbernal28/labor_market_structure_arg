"""Topological data analysis functions."""

import numpy as np
# import kmapper as km
from ripser import ripser


def compute_persistence(distance_matrix, maxdim=2, thresh=np.inf, coeff=2, n_perm=100):
	result = ripser(
		distance_matrix, maxdim=maxdim, thresh=thresh, distance_matrix=True, coeff=coeff, n_perm=n_perm
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
		# Conbine duplicate rows with the same dimension, birth, and death by counting occurrences
		df = df.groupby(["dimension", "birth", "death"]).size().reset_index(name="count")
		df.to_csv(path_csv, index=False)

	def import_(self):
		import pandas as pd

		df = pd.read_csv(self.path_csv)
		# Reconstruct the list of diagrams, repeating rows according to the 'count' column
		df = df.loc[df.index.repeat(df["count"])].reset_index(drop=True)
		dgms = []
		for dim in sorted(df["dimension"].unique()):
			dgm_dim = df[df["dimension"] == dim][["birth", "death"]].values
			dgms.append(dgm_dim)
		return dgms


def bottleneck_distance(dgm1, dgm2):
	"""Calcular la distancia bottleneck entre dos diagramas de persistencia.

	Args:
		dgm1 (np.ndarray): Primer diagrama de persistencia de forma (n_points, 2).
		dgm2 (np.ndarray): Segundo diagrama de persistencia de forma (n_points, 2).

	Returns:
		float: Distancia bottleneck entre los dos diagramas.
	"""
	from scipy.optimize import linear_sum_assignment

	# Filtrar puntos con coordenadas infinitas para evitar matrices inviables
	dgm1 = dgm1[np.isfinite(dgm1).all(axis=1)]
	dgm2 = dgm2[np.isfinite(dgm2).all(axis=1)]

	# Handle edge cases
	if len(dgm1) == 0 and len(dgm2) == 0:
		return 0.0

	if len(dgm1) == 0:
		return max(abs(dgm2[i, 1] - dgm2[i, 0]) / 2.0 for i in range(len(dgm2)))

	if len(dgm2) == 0:
		return max(abs(dgm1[i, 1] - dgm1[i, 0]) / 2.0 for i in range(len(dgm1)))

	n1, n2 = len(dgm1), len(dgm2)
	n = n1 + n2

	# Inicializar matriz de costos con infinito
	cost_matrix = np.full((n, n), np.inf)

	# Top-left block (n1 x n2): point-to-point matching costs
	for i in range(n1):
		for j in range(n2):
			cost_matrix[i, j] = max(
				abs(dgm1[i, 0] - dgm2[j, 0]),
				abs(dgm1[i, 1] - dgm2[j, 1])
			)

	# Top-right block (n1 x n1): match dgm1 points to their diagonal slots
	for i in range(n1):
		persistence = abs(dgm1[i, 1] - dgm1[i, 0])
		cost_matrix[i, n2 + i] = persistence / 2.0

	# Bottom-left block (n2 x n2): match dgm2 points to their diagonal slots
	for j in range(n2):
		persistence = abs(dgm2[j, 1] - dgm2[j, 0])
		cost_matrix[n1 + j, j] = persistence / 2.0

	# Bottom-right block (dummy-to-dummy) stays inf; not used

	# Solve assignment problem
	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	return cost_matrix[row_ind, col_ind].max()


def wasserstein_distance(
	dgm_a: np.ndarray, dgm_b: np.ndarray, order: int = 1, internal_p: int = 2
) -> float:
	"""Compute Wasserstein distance; imports POT-backed Gudhi only when needed."""
	dgm_a = dgm_a[np.isfinite(dgm_a).all(axis=1)] if dgm_a.size else dgm_a
	dgm_b = dgm_b[np.isfinite(dgm_b).all(axis=1)] if dgm_b.size else dgm_b

	try:
		from gudhi.wasserstein import wasserstein_distance as pot_wasserstein_distance
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"Wasserstein distance requires the 'pot' package. Install POT (pip install pot)."
		) from exc
	return float(
		pot_wasserstein_distance(dgm_a, dgm_b, order=order, internal_p=internal_p)
	)

def load_diagrams_by_dimension(path: str) -> dict[int, np.ndarray]:
	"""Load persistence diagrams and preserve their explicit dimensions."""
	loader = DGMS_loader(path)
	dgms = loader.import_()
	if not dgms:
		return {}
	by_dim: dict[int, np.ndarray] = {}
	# DGMS_loader orders by dimension; use the index as the dimension label.
	for dim, dgm in enumerate(dgms):
		by_dim[dim] = dgm
	return by_dim


def align_diagrams(
	empirical: dict[int, np.ndarray],
	null: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
	if not empirical and not null:
		return [], []
	max_dim = max([*empirical.keys(), *null.keys()]) if (empirical or null) else -1
	empirical_list: list[np.ndarray] = []
	null_list: list[np.ndarray] = []
	for dim in range(max_dim + 1):
		empirical_list.append(empirical.get(dim, np.zeros((0, 2))))
		null_list.append(null.get(dim, np.zeros((0, 2))))
	return empirical_list, null_list
