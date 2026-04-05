from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _stringify(value: object) -> str:
	if isinstance(value, (list, tuple, set)):
		return ", ".join(str(v) for v in value)
	if isinstance(value, dict):
		return ", ".join(f"{k}={v}" for k, v in value.items())
	return str(value)


def _add_block(lines: list[str], title: str, entries: Iterable[str]) -> None:
	lines.append("")
	lines.append(title)
	lines.append("-" * len(title))
	for entry in entries:
		lines.append(entry)


def add_snakemake_overview(lines: list[str], snakemake: object) -> None:
	inputs = getattr(snakemake, "input", [])
	outputs = getattr(snakemake, "output", [])
	params = getattr(snakemake, "params", {})
	wildcards = getattr(snakemake, "wildcards", {})

	_add_block(
		lines,
		"SNAKEMAKE",
		[
			f"Inputs: {_stringify(inputs)}",
			f"Outputs: {_stringify(outputs)}",
			f"Params: {_stringify(params)}",
			f"Wildcards: {_stringify(wildcards)}",
		],
	)


def add_dataframe_info(lines: list[str], label: str, row_count: int, column_count: int) -> None:
	_add_block(
		lines,
		label,
		[
			f"Rows: {row_count}",
			f"Columns: {column_count}",
		],
	)


def add_graph_metrics(lines: list[str], label: str, metrics: Mapping[str, object]) -> None:
	entries = [
		f"Node count: {metrics.get('node_count')}",
		f"Edge count: {metrics.get('edge_count')}",
		f"Loop count: {metrics.get('self_loops')}",
		f"Average degree: {metrics.get('avg_degree'):.2f}"
		if metrics.get("avg_degree") is not None
		else "Average degree: N/A",
		f"Average weighted degree: {metrics.get('avg_weighted_degree'):.2f}"
		if metrics.get("avg_weighted_degree") is not None
		else "Average weighted degree: N/A",
		f"Average clustering coefficient: {metrics.get('avg_clustering'):.4f}"
		if metrics.get("avg_clustering") is not None
		else "Average clustering coefficient: N/A",
		f"Diameter (largest component): {metrics.get('diameter')}",
		f"Connected components: {metrics.get('connected_components')}",
	]
	_add_block(lines, label, entries)


def add_notes(lines: list[str], title: str, notes: Sequence[str]) -> None:
	_add_block(lines, title, notes)


def write_log(lines: Sequence[str], log_path: str | Path | None, mirror_stdout: bool = True) -> None:
	text = "\n".join(lines).rstrip() + "\n"
	if mirror_stdout:
		print(text, end="")
	if log_path:
		Path(log_path).parent.mkdir(parents=True, exist_ok=True)
		Path(log_path).write_text(text)
