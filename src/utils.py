import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from datetime import date
from collections import Counter, defaultdict
from collections.abc import Mapping, Iterable
import networkx as nx
import numpy as np


@dataclass(frozen=True)
class EphKey:
	year: int
	period: int
	time_numeric: float
	time_date: date
	label: str
	raw: str


def setup_networkx_backend(algorithm: str | None = None) -> None:
	try:
		from networkx import config

		if algorithm == "leiden":
			config.backend_priority = ["cugraph", "networkx"]
		else:
			config.backend_priority = ["networkx"]
	except Exception as e:
		print(f"Error setting up networkx backend: {e}")


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


def label_to_id(label: str) -> int:
	"""Convert a label like 'C01' back to an integer ID."""
	m = re.match(r"C(\d+)", label)
	if m:
		return int(m.group(1))
	raise ValueError(f"Invalid label format: {label}. Expected format like 'C01'.")


@lru_cache(maxsize=1000)
def parse_color_from_string(color_str: str):
	color_str_clean = color_str.strip()
	if color_str_clean.startswith("#"):
		return color_str_clean
	if not any(c in color_str_clean for c in "()[],"):
		return color_str_clean
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


def _is_missing_label(label: object) -> bool:
	if label is None:
		return True
	if isinstance(label, float) and np.isnan(label):
		return True
	return False


def _community_sort_key(label: object) -> tuple[int, int | str]:
	label_str = str(label)
	match = re.search(r"(\d+)", label_str)
	if match:
		return (0, int(match.group(1)))
	return (1, label_str)


def build_community_color_map(
	labels: Iterable[object],
	*,
	other_label: str = "Otros",
	palette: str = "default",
) -> dict[str, str]:
	"""Return a label->color map using the Gephi palette for communities."""
	from ggsci import pal_gephi

	clean_labels = [
		str(label)
		for label in labels
		if not _is_missing_label(label) and str(label) != other_label
	]
	np.random.seed(
		len(set(clean_labels))
	)  # Seed based on number of unique labels for consistency
	unique_labels = sorted(set(clean_labels), key=_community_sort_key)
	if unique_labels:
		gephi_colors_func = pal_gephi(palette=palette)
		palette_colors = gephi_colors_func(len(unique_labels))
		color_map = {
			label: palette_colors[i % len(palette_colors)]
			for i, label in enumerate(unique_labels)
		}
	else:
		color_map = {}

	if any(
		str(label) == other_label for label in labels if not _is_missing_label(label)
	):
		color_map[other_label] = "gray"
	return color_map


def build_node_color_map_from_communities(
	community_map: Mapping[int, object],
	*,
	other_label: str = "Otros",
	palette: str = "default",
) -> dict[int, str]:
	"""Return a node->color map based on community labels."""
	label_color_map = build_community_color_map(
		community_map.values(),
		other_label=other_label,
		palette=palette,
	)
	return {
		node_id: label_color_map.get(str(community), "gray")
		for node_id, community in community_map.items()
	}


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


def original_caes_id(id: int, max_caes_id: int | None = None) -> int:
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


def original_id(id: int, class_index: int, max_caes_id: int | None = None) -> int:
	"""Recover original CAES or CIUO ID from disambiguated ID based on class index."""
	if class_index == 1:
		return original_caes_id(id, max_caes_id)
	elif class_index == 0:
		return original_ciuo_id(id, max_caes_id)
	else:
		raise ValueError(
			f"Invalid class index {class_index}. Must be 0 (CIUO) or 1 (CAES)."
		)


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


def get_config_section(
	config: Mapping[str, object] | None,
	preferred_key: str,
	legacy_key: str | None = None,
) -> dict:
	"""Return a config section, preferring new keys and falling back to legacy."""
	if not config:
		return {}
	section = config.get(preferred_key)
	if isinstance(section, Mapping):
		return dict(section)
	if legacy_key:
		section = config.get(legacy_key)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def translate_label(label: str, translation: Mapping[str, str] | None) -> str:
	"""Translate a label using the mapping if present; otherwise return original."""
	if not translation:
		return label
	return translation.get(label, label)


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

	# Sort communities by their total aggregated weight, using old_id as tie-break for reproducibility
	is_desc = order.lower() == "desc"
	sorted_communities = sorted(
		community_weights.items(), key=lambda item: (item[1], item[0]), reverse=is_desc
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


def compute_node_neighbor_mean(
	G, feature_map: Mapping, *, weight_attr: str = "weight"
) -> dict:
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


def compute_node_sizes(
	graph: nx.Graph,
	*,
	nodes: Iterable | None = None,
	factor: float = 1.0,
	min_size: float = 5.0,
	weight_attr: str = "weight",
) -> dict:
	"""Compute node sizes based on weighted degree with a minimum size floor."""
	if nodes is None:
		nodes_list = list(graph.nodes())
	else:
		nodes_list = list(nodes)

	weighted_degree = {node: 0.0 for node in nodes_list}
	for u, v, data in graph.edges(data=True):
		w = float(data.get(weight_attr, 0.0))
		if u in weighted_degree:
			weighted_degree[u] += w
		if v in weighted_degree:
			weighted_degree[v] += w

	max_deg = max(weighted_degree.values()) if weighted_degree else 1.0
	if max_deg <= 0.0:
		max_deg = 1.0

	# Normalize node sizes to a fixed scale (100.0) relative to max_deg,
	# so that weights of different magnitudes (e.g. hidalgo vs dot_product)
	# yield visually consistent node sizes.
	base_constant = 100.0

	size_map = {
		node: max(
			min_size,
			(weighted_degree.get(node, 0.0) / max_deg) * base_constant * float(factor)
		)
		for node in nodes_list
	}
	return size_map


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


def get_markov_time(resolution: float) -> float:
	"""
	Converts a standard resolution parameter to Infomap's Markov time.

	In modularity-based algorithms (Louvain/Leiden), higher resolution yields
	smaller communities. In Infomap, shorter Markov time yields smaller communities.
	Using the reciprocal aligns their behavior for parameter sweeps.

	Args:
		resolution (float): The resolution parameter (expected range: 0.1 to 40).

	Returns:
		float: The corresponding Markov time.
	"""
	if resolution <= 0:
		raise ValueError("Resolution must be strictly greater than 0.")

	# Inverse relationship to match Leiden/Louvain behavior
	return 1.0 / resolution


def get_community_color(community_label, communities):
	"""Assign a color based on pal_gephi palette, with 'Other' as gray."""
	color_map = build_community_color_map(communities, other_label="Otros")
	return color_map.get(str(community_label), "gray")


def update_gexf_metadata(
	filepath: str,
	creator: str,
	description: str | None = None,
	keywords: str | None = None,
	lastmodifieddate: str | None = None,
) -> None:
	"""Update a GEXF file's XML metadata attributes to include creator, description, keywords, date, and defaultedgetype."""
	import xml.etree.ElementTree as ET
	import datetime

	# Parse the xml file
	tree = ET.parse(filepath)
	root = tree.getroot()

	# Extract namespace from root tag
	ns = ""
	if root.tag.startswith("{"):
		ns = root.tag.split("}")[0].strip("{")

	if ns:
		ET.register_namespace("", ns)
		meta_tag = f"{{{ns}}}meta"
		creator_tag = f"{{{ns}}}creator"
		description_tag = f"{{{ns}}}description"
		keywords_tag = f"{{{ns}}}keywords"
		graph_tag = f"{{{ns}}}graph"
	else:
		meta_tag = "meta"
		creator_tag = "creator"
		description_tag = "description"
		keywords_tag = "keywords"
		graph_tag = "graph"

	meta = root.find(meta_tag)
	if meta is None:
		meta = ET.Element(meta_tag)
		root.insert(0, meta)

	if lastmodifieddate is None:
		lastmodifieddate = datetime.date.today().strftime("%Y-%m-%d")
	meta.set("lastmodifieddate", lastmodifieddate)

	# Find or create creator
	creator_elem = meta.find(creator_tag)
	if creator_elem is None:
		creator_elem = ET.SubElement(meta, creator_tag)
	creator_elem.text = creator

	# Find or create description
	if description is not None:
		desc_elem = meta.find(description_tag)
		if desc_elem is None:
			desc_elem = ET.SubElement(meta, description_tag)
		desc_elem.text = description

	# Find or create keywords
	if keywords is not None:
		kw_elem = meta.find(keywords_tag)
		if kw_elem is None:
			kw_elem = ET.SubElement(meta, keywords_tag)
		kw_elem.text = keywords

	# Check graph element for defaultedgetype
	graph_elem = root.find(graph_tag)
	if graph_elem is not None:
		if "defaultedgetype" not in graph_elem.attrib:
			graph_elem.set("defaultedgetype", "undirected")

	# Write back
	tree.write(filepath, encoding="utf-8", xml_declaration=True)
