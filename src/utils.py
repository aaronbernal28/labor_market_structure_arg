import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from datetime import date
from collections import Counter, defaultdict
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True)
class EphKey:
	year: int
	period: int
	time_numeric: float
	time_date: date
	label: str
	raw: str


def eph_quarter_start_date(year: int, period: int) -> date:
	"""Map EPH quarter to a calendar date at quarter start."""
	if period not in {1, 2, 3, 4}:
		raise ValueError(f"Invalid EPH period {period}. Expected one of 1..4.")
	month = 1 + 3 * (period - 1)
	return date(int(year), int(month), 1)


def parse_eph_file_key(eph_file: str) -> EphKey | None:
	"""Parse EPH filename into a sortable key and date representations."""
	m = re.search(r"[Tt]([1-4])([0-9]{2})", eph_file)
	if not m:
		return None
	period = int(m.group(1))
	yy = int(m.group(2))
	year = 2000 + yy if yy < 100 else yy
	time_numeric = float(year) + float(period - 1) / 4.0
	time_date = eph_quarter_start_date(year, period)
	label = f"{year}-T{period}"
	return EphKey(
		year=year,
		period=period,
		time_numeric=time_numeric,
		time_date=time_date,
		label=label,
		raw=eph_file,
	)


def sort_eph_files(eph_files: list[str]) -> list[str]:
	"""Sort EPH filenames by inferred (year, quarter), then append unparsed names."""
	parsed: list[tuple[int, int, str]] = []
	fallback: list[str] = []
	for name in eph_files:
		k = parse_eph_file_key(name)
		if k is None:
			fallback.append(name)
		else:
			parsed.append((k.year, k.period, k.raw))

	parsed_sorted = [raw for _, _, raw in sorted(parsed)]
	fallback_sorted = sorted(fallback)
	return parsed_sorted + fallback_sorted


def extract_eph_file_from_path(path: Path) -> str:
	"""Extract EPH series folder from expected projection path layout."""
	parts = list(path.parts)
	try:
		i = parts.index("eph")
	except ValueError:
		return path.parent.parent.name
	if i + 1 < len(parts):
		return parts[i + 1]
	return path.parent.parent.name


def ensure_parent_dir(path: Path) -> None:
	"""Create parent directory for an output file path if missing."""
	path.parent.mkdir(parents=True, exist_ok=True)


def label_fn(c, pad=2):
	return f"C{str(int(c)).zfill(pad)}" if isinstance(c, int) else str(c)


@lru_cache(maxsize=1000)
def parse_color_from_string(color_str: str):
	try:
		# Handle numpy string representations like "(np.float64(0.5), np.float64(1.0))"
		if "np.float64" in color_str or "float" in color_str:
			# Remove the type names to just get the brackets and numbers
			clean_str = re.sub(r"np\.float\d*\(", "", color_str).replace(")", "")
			# Extract all floating point numbers
			numbers = [
				float(x)
				for x in re.findall(
					r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", clean_str
				)
			]
			if numbers:
				return tuple(numbers)

		parsed = ast.literal_eval(color_str)
		if hasattr(parsed, "__iter__") and not isinstance(parsed, str):
			return tuple(float(x) for x in parsed)
		return parsed
	except (ValueError, SyntaxError, TypeError):
		print(
			f"Warning: Unable to parse color string '{color_str}'. Defaulting to gray."
		)
		return "gray"


def parse_color(color_value):
	"""Parse color value from CSV (can be string tuple or already a tuple)."""
	if isinstance(color_value, str):
		return parse_color_from_string(color_value)
	# If it's already a tuple/list/array, ensure it's a regular tuple
	if hasattr(color_value, "__iter__") and not isinstance(color_value, str):
		return tuple(float(x) for x in color_value)
	return color_value


def get_class_index(col_name: str) -> int:
	"""Determine if the column corresponds to CAES or CIUO IDs for bipartite graph construction."""
	return int("caes" in col_name.lower())


def get_column_name(
	col_index: int, caes_name: str = "caes", ciuo_name: str = "ciuo"
) -> str:
	"""Get a canonical column label from the bipartite class index."""
	if col_index == 1:
		return caes_name
	if col_index == 0:
		return ciuo_name
	raise ValueError(f"Invalid column index {col_index}. Must be 1 (CAES) or 0 (CIUO).")


def original_caes_id(id: int) -> int:
	"""Recover original CAES ID from disambiguated ID."""
	return id


def _resolve_max_caes_id(max_caes_id: int | None) -> int:
	if max_caes_id is not None:
		return max_caes_id
	try:
		from config import MAX_CAES_ID  # type: ignore

		return int(MAX_CAES_ID)
	except Exception as exc:
		raise ValueError(
			"max_caes_id must be provided when config.MAX_CAES_ID is unavailable."
		) from exc


def original_ciuo_id(id: int, max_caes_id: int | None = None) -> int:
	"""
	Recover original CIUO ID from disambiguated ID.
	"""
	return id - _resolve_max_caes_id(max_caes_id)


def desambiated_caes_id(id: int) -> int:
	return id


def desambiated_ciuo_id(id: int, max_caes_id: int | None = None) -> int:
	"""
	Recover original CIUO ID from disambiguated ID.
	"""
	return id + _resolve_max_caes_id(max_caes_id)


def _as_bool(value: object) -> bool:
	"""Convert Snakemake wildcard values into strict booleans."""
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value != "non_logscale"
	raise ValueError(f"Invalid boolean value for logscale: {value!r}")


def get_top_communities(community_map: dict[int, str], top_n: int = 5) -> set[str]:
	"""Identify the top N communities by node count."""
	if top_n <= 0 or top_n >= len(set(community_map.values())):
		return set(community_map.values())

	community_counts = Counter(community_map.values())
	top_communities = {c for c, _ in community_counts.most_common(top_n)}
	return top_communities


def relabel_communities_by_observations(
	communities: dict,
	n_obs: dict,
	order: str = "desc",
	num_communities: int | None = None,
) -> dict:
	"""
	Relabel communities by their total size (weighted by n_obs).

	Args:
	    communities: Dictionary mapping node -> community_id
	    n_obs: Dictionary mapping node -> number of total observations per node (used for size weighting)
	    order: "desc" for descending size (largest first), "asc" for ascending
	    num_communities: Optional number of highest-weight communities to keep.
	                     Nodes belonging to smaller communities are removed from the output.

	Returns:
	    Dictionary with relabeled communities based on size ordering
	"""
	# Sum the observations (weights) for each community
	community_weights = defaultdict(int)
	for node, comm_id in communities.items():
		community_weights[comm_id] += n_obs.get(node, 0)

	# Sort communities by their total aggregated weight
	is_desc = order.lower() == "desc"
	sorted_communities = sorted(
		community_weights.items(), key=lambda item: item[1], reverse=is_desc
	)

	# Create mapping from old community ID to new ID
	old_to_new = {
		old_id: new_id for new_id, (old_id, _weight) in enumerate(sorted_communities)
	}

	print(
		f"Relabeling communities by total observations with order='{order}' and num_communities={num_communities}."
	)

	# Relabel communities, dropping any that fall outside the top `num_communities`
	relabelled_communities = {}
	for node, comm_id in communities.items():
		new_id = old_to_new[comm_id]

		# If no limit is set, OR if the new_id falls within our requested top N
		if num_communities is None or new_id < num_communities:
			relabelled_communities[node] = new_id

	return relabelled_communities


def filter_communities_by_size(communities: dict, min_size: int = 1) -> dict:
	"""
	Filter out communities smaller than the minimum size threshold (raw node count).

	Args:
		communities: Dictionary mapping node -> community_id
		min_size: Minimum number of nodes a community must have to be kept

	Returns:
		Dictionary with only nodes belonging to communities of size >= min_size
	"""
	# Count nodes in each community
	community_counts = Counter(communities.values())

	# Identify communities that meet the raw size threshold
	valid_communities = {
		comm_id for comm_id, count in community_counts.items() if count >= min_size
	}

	# Keep only nodes in valid communities
	return {
		node: comm_id
		for node, comm_id in communities.items()
		if comm_id in valid_communities
	}


def filter_communities_by_observations(
	communities: dict, n_obs: dict, min_observations: int
) -> dict:
	"""
	Filter out communities smaller than the minimum observation threshold.

	Args:
		communities: Dictionary mapping node -> community_id
		n_obs: Dictionary mapping node -> number of total observations per node (used for size weighting)
		min_observations: Minimum total observations a community must have to be kept

	Returns:
		Dictionary with only nodes belonging to communities of total observations >= min_observations
	"""
	# Sum the observations (weights) for each community
	community_weights = defaultdict(int)
	for node, comm_id in communities.items():
		community_weights[comm_id] += n_obs.get(node, 0)

	# Identify communities that meet the total observation threshold
	valid_communities = {
		comm_id
		for comm_id, weight in community_weights.items()
		if weight >= min_observations
	}

	# Keep only nodes in valid communities
	return {
		node: comm_id
		for node, comm_id in communities.items()
		if comm_id in valid_communities
	}


def compute_node_neighbor_mean(G, feature_map: Mapping, *, weight_attr: str = "weight") -> dict:
	"""Compute weighted (or unweighted) average neighbor feature for each node.

	Returns a dict mapping node -> average(neighbor_feature_values). Only nodes
	with at least one neighbor with a finite feature value are returned.
	"""
	valid_nodes = {n for n, val in feature_map.items() if np.isfinite(val)}

	y_vals_map: dict = {u: [] for u in valid_nodes}
	edge_weights: dict = {u: [] for u in valid_nodes}

	for u, v, data in G.edges(data=True):
		if u in valid_nodes and v in valid_nodes:
			w = data.get(weight_attr, 0.0)
			if np.isfinite(feature_map[u]) and np.isfinite(feature_map[v]):
				y_vals_map[u].append(feature_map[v])
				edge_weights[u].append(w)

				y_vals_map[v].append(feature_map[u])
				edge_weights[v].append(w)

	node_y = {}
	for u in y_vals_map:
		neigh_vals = y_vals_map[u]
		if not neigh_vals:
			continue
		weights = np.asarray(edge_weights[u], dtype=float)
		if weights.size == 0 or np.isclose(weights.sum(), 0.0):
			node_y[u] = float(np.mean(neigh_vals))
		else:
			node_y[u] = float(np.average(neigh_vals, weights=weights))

	return node_y


def get_top_mean_assortativity_communities(
	G,
	feature_map: Mapping,
	community_map: Mapping,
	*,
	top_k: int = 4,
	order: str = "desc",
	weight_attr: str = "weight",
) -> list:
	"""Return the community ids with the most extreme mean neighbor-feature values.

	This computes for each node the (weighted) average of neighbor feature values
	(the $Y_i$ axis in the assortativity plots), then averages those $Y_i$ per
	community and returns the top-k communities with largest (order='desc') or
	smallest (order='asc') mean $Y_i$.

	Args:
		G: networkx Graph (undirected)
		feature_map: mapping node -> numeric feature value
		community_map: mapping node -> community id
		top_k: number of communities to return
		order: 'desc' for highest means, 'asc' for lowest means
		weight_attr: edge attribute to use as weight (falls back to unweighted mean
					 if weights sum to zero)

	Returns:
		List of community ids (length <= top_k) ordered by extremity.
	"""
	if community_map is None:
		return []

	# Compute per-node neighbor-average using local helper
	node_y = compute_node_neighbor_mean(G, feature_map, weight_attr=weight_attr)

	# Aggregate Y_i by community and compute community mean
	comm_to_vals = defaultdict(list)
	for node, y in node_y.items():
		comm = community_map.get(node)
		if comm is None:
			continue
		if not np.isfinite(y):
			continue
		comm_to_vals[comm].append(y)

	# Compute mean per community
	comm_mean = {c: float(np.mean(vals)) for c, vals in comm_to_vals.items() if vals}
	if not comm_mean:
		return []

	reverse = order.lower() == "desc"
	sorted_comms = sorted(comm_mean.items(), key=lambda kv: kv[1], reverse=reverse)
	top = [comm for comm, _ in sorted_comms[:top_k]]
	return top
