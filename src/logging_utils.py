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


def add_dataframe_info(
	lines: list[str], label: str, row_count: int, column_count: int
) -> None:
	_add_block(
		lines,
		label,
		[
			f"Rows: {row_count}",
			f"Columns: {column_count}",
		],
	)


def add_graph_metrics(
	lines: list[str], label: str, metrics: Mapping[str, object]
) -> None:
	entries = [
		f"Node count: {metrics.get('node_count')}",
		f"Edge count: {metrics.get('edge_count')}",
		f"Loop count: {metrics.get('self_loops')}",
		f"Bipartite density: {metrics.get('bipartite_density'):.6f}"
		if metrics.get("bipartite_density") is not None
		else "Bipartite density: N/A",
		f"Average degree: {metrics.get('avg_degree'):.2f}"
		if metrics.get("avg_degree") is not None
		else "Average degree: N/A",
		f"Average weighted degree: {metrics.get('avg_weighted_degree'):.2f}"
		if metrics.get("avg_weighted_degree") is not None
		else "Average weighted degree: N/A",
		f"Average clustering coefficient: {metrics.get('avg_clustering'):.4f}"
		if metrics.get("avg_clustering") is not None
		else "Average clustering coefficient: N/A",
		f"Average clustering coefficient ponderado: {metrics.get('avg_weighted_clustering'):.4f}"
		if metrics.get("avg_weighted_clustering") is not None
		else "Average clustering coefficient ponderado: N/A",
		f"Diameter (largest component): {metrics.get('diameter')}",
		f"Average path length (largest component): {metrics.get('avg_path_length'):.4f}"
		if metrics.get("avg_path_length") is not None
		else "Average path length (largest component): N/A",
		f"Connected components: {metrics.get('connected_components')}",
		f"Largest component size: {metrics.get('lcc_size')} ({metrics.get('lcc_percent'):.2f}%)"
		if metrics.get("lcc_size") is not None
		and metrics.get("lcc_percent") is not None
		else "Largest component size: N/A",
		f"Degree assortativity: {metrics.get('degree_assortativity'):.4f}"
		if metrics.get("degree_assortativity") is not None
		else "Degree assortativity: N/A",
		f"Degree assortativity (weighted): {metrics.get('degree_assortativity_weighted'):.4f}"
		if metrics.get("degree_assortativity_weighted") is not None
		else "Degree assortativity (weighted): N/A",
	]
	_add_block(lines, label, entries)


def add_bipartite_degree_strength_latex(
	lines: list[str], label: str, metrics: Mapping[str, object]
) -> None:
	partitions = metrics.get("bipartite_partitions")
	if not isinstance(partitions, dict) or not partitions:
		return

	def _fmt(value: object, digits: int = 2) -> str:
		if value is None:
			return "NA"
		try:
			return f"{float(value):.{digits}f}"
		except (TypeError, ValueError):
			return "NA"

	entries: list[str] = []
	entries.append(r"\begin{table}[H]")
	entries.append(r"\centering")
	entries.append(r"\begin{tabular}{lrrrrrrrrr}")
	entries.append(r"\hline")
	entries.append(
		r"Grupo & n & k\_mean & k\_median & k\_min & k\_max & s\_mean & s\_median & s\_min & s\_max \\"
	)
	entries.append(r"\hline")
	for group_label, group_data in partitions.items():
		if not isinstance(group_data, dict):
			continue
		degree_stats = group_data.get("degree_stats", {})
		strength_stats = group_data.get("strength_stats", {})
		row = (
			f"{group_label} & {group_data.get('size', 'NA')}"
			f" & {_fmt(degree_stats.get('mean'))}"
			f" & {_fmt(degree_stats.get('median'))}"
			f" & {_fmt(degree_stats.get('min'))}"
			f" & {_fmt(degree_stats.get('max'))}"
			f" & {_fmt(strength_stats.get('mean'))}"
			f" & {_fmt(strength_stats.get('median'))}"
			f" & {_fmt(strength_stats.get('min'))}"
			f" & {_fmt(strength_stats.get('max'))} \\"
		)
		entries.append(row)
	entries.append(r"\hline")
	entries.append(r"\end{tabular}")
	entries.append(r"\end{table}")
	_add_block(lines, label, entries)


def add_notes(lines: list[str], title: str, notes: Sequence[str]) -> None:
	_add_block(lines, title, notes)


def write_log(
	lines: Sequence[str], log_path: str | Path | None, mirror_stdout: bool = True
) -> None:
	text = "\n".join(lines).rstrip() + "\n"
	if mirror_stdout:
		print(text, end="")
	if log_path:
		Path(log_path).parent.mkdir(parents=True, exist_ok=True)
		Path(log_path).write_text(text)
