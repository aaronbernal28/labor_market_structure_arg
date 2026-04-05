import ast
from functools import lru_cache


def label_fn(c, pad=2):
	return f"C{str(int(c)).zfill(pad)}" if isinstance(c, int) else str(c)


@lru_cache(maxsize=1000)
def parse_color_from_string(color_str: str):
	parsed = ast.literal_eval(color_str)
	if hasattr(parsed, "__iter__") and not isinstance(parsed, str):
		return tuple(float(x) for x in parsed)
	return parsed


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


def original_ciuo_id(id: int, max_caes_id: int = None) -> int:
	"""
	Recover original CIUO ID from disambiguated ID.
	"""
	return id - _resolve_max_caes_id(max_caes_id)


def desambiated_caes_id(id: int) -> int:
	return id


def desambiated_ciuo_id(id: int, max_caes_id: int = None) -> int:
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
