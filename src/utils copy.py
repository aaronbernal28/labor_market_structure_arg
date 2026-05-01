import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from datetime import date


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
	"""
	Parse color string into a tuple of floats or a color name.
	Handles: tuples like "(1, 0, 0)", hex like "#FF0000", named colors, rgb/rgba functions.
	"""
	color_str = color_str.strip()

	# Try literal_eval first for simple tuples/numbers
	try:
		parsed = ast.literal_eval(color_str)
		if hasattr(parsed, "__iter__") and not isinstance(parsed, str):
			return tuple(float(x) for x in parsed)
		return parsed
	except (ValueError, SyntaxError):
		pass

	# Handle rgb/rgba function notation: rgb(255, 0, 0) or rgba(255, 0, 0, 0.5)
	if color_str.lower().startswith(("rgb(", "rgba(")):
		try:
			# Extract numbers from rgb(...) or rgba(...)
			import re
			numbers = re.findall(r"[\d.]+", color_str)
			# Normalize to 0-1 range if values are > 1 (likely 0-255 range)
			normalized = []
			for num in numbers:
				val = float(num)
				if val > 1:
					val = val / 255.0
				normalized.append(val)
			return tuple(normalized)
		except Exception:
			pass

	# Handle hex colors: #FF0000
	if color_str.startswith("#"):
		try:
			return color_str
		except Exception:
			pass

	# Return as-is if it's a named color or unknown format
	return color_str


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

	from collections import Counter

	community_counts = Counter(community_map.values())
	top_communities = {c for c, _ in community_counts.most_common(top_n)}
	return top_communities


def relabel_communities_by_size(communities: dict, order: str = "desc") -> dict:
	"""
	Relabel communities by their size (number of nodes).

	Args:
		communities: Dictionary mapping node -> community_id
		order: "desc" for descending size (largest first), "asc" for ascending

	Returns:
		Dictionary with relabeled communities based on size ordering
	"""
	from collections import Counter

	# Count nodes in each community
	community_counts = Counter(communities.values())

	# Sort communities by size
	if order.lower() == "desc":
		sorted_communities = sorted(
			community_counts.items(), key=lambda x: x[1], reverse=True
		)
	else:  # asc
		sorted_communities = sorted(community_counts.items(), key=lambda x: x[1])

	# Create mapping from old community ID to new ID
	old_to_new = {
		old_id: new_id for new_id, (old_id, _) in enumerate(sorted_communities)
	}

	# Relabel communities
	relabeled = {node: old_to_new[comm_id] for node, comm_id in communities.items()}

	return relabeled


def filter_communities_by_size(communities: dict, min_size: int = 1) -> dict:
	"""
	Filter out communities smaller than the minimum size threshold.

	Args:
		communities: Dictionary mapping node -> community_id
		min_size: Minimum number of nodes a community must have to be kept

	Returns:
		Dictionary with only nodes belonging to communities of size >= min_size
	"""
	from collections import Counter

	# Count nodes in each community
	community_counts = Counter(communities.values())

	# Identify communities that meet the size threshold
	valid_communities = {
		comm_id for comm_id, count in community_counts.items() if count >= min_size
	}

	# Keep only nodes in valid communities
	filtered = {
		node: comm_id
		for node, comm_id in communities.items()
		if comm_id in valid_communities
	}

	return filtered
