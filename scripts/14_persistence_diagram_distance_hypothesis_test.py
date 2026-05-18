from typing import Any, Iterable
import hashlib
import math
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from scripts import log

snakemake: Any

NULL_FAMILIES_DEFAULT = [
	"configuration_model",
	"watts_strogatz",
	"barabasi_albert",
	"stochastic_block_model",
	"erdos_renyi",
]


def _parse_null_label(label: str, families: Iterable[str]) -> tuple[str, int] | None:
	for family in families:
		prefix = f"{family}_"
		if label.startswith(prefix):
			suffix = label[len(prefix) :]
			if suffix.isdigit():
				return family, int(suffix)
	return None


def _stable_seed(base_seed: int, family: str, dimension: int, metric: str) -> int:
	token = f"{base_seed}|{family}|{dimension}|{metric}"
	digest = hashlib.md5(token.encode("utf-8")).hexdigest()
	return int(digest[:8], 16)


def _permutation_p_value(
	sample_a: np.ndarray,
	sample_b: np.ndarray,
	n_perm: int,
	seed: int,
	two_sided: bool,
) -> float:
	if sample_a.size == 0 or sample_b.size == 0:
		return float("nan")

	# Difference in means
	def statistic(a, b, axis):
		return np.mean(a, axis=axis) - np.mean(b, axis=axis)

	# Map your boolean to SciPy's alternative hypothesis string
	alt_hyp = 'two-sided' if two_sided else 'greater'

	rng = np.random.default_rng(seed)

	# Run the optimized permutation test
	res = permutation_test(
		data=(sample_a, sample_b),
		statistic=statistic,
		permutation_type='independent',
		n_resamples=n_perm,
		vectorized=True,
		alternative=alt_hyp,
		rng=rng,
	)

	return res.pvalue


def _sig_stars(p_value: float, n_tests: int) -> str:
	if not math.isfinite(p_value) or n_tests <= 0:
		return ""
	if p_value < 0.001 / n_tests:
		return "***"
	if p_value < 0.01 / n_tests:
		return "**"
	if p_value < 0.05 / n_tests:
		return "*"
	return ""

def main() -> None:
	df = pd.read_csv(snakemake.input[0])

	required_cols = {"model_1", "model_2", "dimension", "bottleneck", "wasserstein"}
	missing = required_cols - set(df.columns)
	if missing:
		raise KeyError(f"Missing required columns: {sorted(missing)}")

	null_families = list(
		getattr(snakemake.params, "null_families", None)
		or getattr(snakemake.params, "null_family", None)
		or NULL_FAMILIES_DEFAULT
	)
	alpha = float(getattr(snakemake.params, "alpha", 0.05))
	n_perm = int(getattr(snakemake.params, "n_perm", 10000))
	two_sided = bool(getattr(snakemake.params, "two_sided", True))
	seed = int(getattr(snakemake.params, "seed", 42))

	labels = set(df["model_1"]).union(df["model_2"])
	empirical_labels = sorted(
		label for label in labels if _parse_null_label(label, null_families) is None
	)
	if not empirical_labels:
		raise ValueError("No empirical labels found; check null family prefixes.")

	empirical_label = empirical_labels[0]
	skipped_empirical = [
		label for label in empirical_labels if label != empirical_label
	]

	dimensions = sorted(df["dimension"].unique())
	metrics = ["bottleneck", "wasserstein"]

	samples: dict[str, dict[int, dict[str, dict[str, list[float]]]]] = {}
	warnings: list[str] = []

	for _, row in df.iterrows():
		model_1 = row["model_1"]
		model_2 = row["model_2"]
		dim = int(row["dimension"])

		info_1 = _parse_null_label(model_1, null_families)
		info_2 = _parse_null_label(model_2, null_families)

		def _ensure(family: str, dimension: int) -> None:
			samples.setdefault(family, {}).setdefault(dimension, {}).setdefault(
				"emp_null", {metric: [] for metric in metrics}
			)
			samples[family][dimension].setdefault(
				"null_null", {metric: [] for metric in metrics}
			)

		if model_1 == empirical_label and info_2 is not None:
			family, _ = info_2
			_ensure(family, dim)
			for metric in metrics:
				samples[family][dim]["emp_null"][metric].append(float(row[metric]))
			continue

		if model_2 == empirical_label and info_1 is not None:
			family, _ = info_1
			_ensure(family, dim)
			for metric in metrics:
				samples[family][dim]["emp_null"][metric].append(float(row[metric]))
			continue

		if info_1 is not None and info_2 is not None:
			family_1, _ = info_1
			family_2, _ = info_2
			if family_1 == family_2:
				_ensure(family_1, dim)
				for metric in metrics:
					samples[family_1][dim]["null_null"][metric].append(
						float(row[metric])
					)
			continue

	if skipped_empirical:
		warnings.append(
			"Multiple empirical labels found. Using first label only: "
			f"{empirical_label}. Skipped: {', '.join(skipped_empirical)}"
		)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PERSISTENCE DIAGRAM DISTANCE HYPOTHESIS TEST")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)

	log.add_notes(
		log_lines,
		"DATA OVERVIEW",
		[
			f"Input: {snakemake.input[0]}",
			f"Empirical label: {empirical_label}",
			f"Null families: {', '.join(null_families)}",
			f"Dimensions: {', '.join(str(d) for d in dimensions)}",
			f"Metrics: {', '.join(metrics)}",
		],
	)

	log.add_notes(
		log_lines,
		"PARAMETERS",
		[
			f"Alpha: {alpha}",
			f"Permutation count: {n_perm}",
			f"Two-sided: {two_sided}",
			f"Seed: {seed}",
			"Significance stars (Bonferroni-style): *** p < 0.001/d, ** p < 0.01/d, * p < 0.05/d",
		],
	)

	if warnings:
		log.add_notes(log_lines, "WARNINGS", warnings)

	null_family_display_map = {
		"barabasi_albert": "Barabasi-Albert",
		"configuration_model": "Configuration Model",
		"erdos_renyi": "Erdos-Renyi",
		"stochastic_block_model": "Stochastic Block Model",
		"watts_strogatz": "Watts-Strogatz",
	}

	for metric in metrics:
		log_lines.append("")
		log_lines.append(f"METRIC: {metric}")
		log_lines.append("-" * (8 + len(metric)))

		rows: list[dict[str, object]] = []

		for family in sorted(samples.keys()):
			dims_for_family = samples[family]
			valid_dims = [
				dim
				for dim in dims_for_family
				if len(dims_for_family[dim]["emp_null"][metric]) > 0
				and len(dims_for_family[dim]["null_null"][metric]) > 0
			]
			n_tests = len(valid_dims)
			threshold = alpha / n_tests if n_tests > 0 else float("nan")

			for dim in sorted(dims_for_family.keys()):
				emp_vals = np.array(
					dims_for_family[dim]["emp_null"][metric], dtype=float
				)
				null_vals = np.array(
					dims_for_family[dim]["null_null"][metric], dtype=float
				)

				p_value = float("nan")
				if emp_vals.size > 0 and null_vals.size > 0:
					p_value = _permutation_p_value(
						emp_vals,
						null_vals,
						n_perm,
						_stable_seed(seed, family, dim, metric),
						two_sided,
					)

				mean_diff = (
					float(emp_vals.mean() - null_vals.mean())
					if emp_vals.size > 0 and null_vals.size > 0
					else float("nan")
				)

				rows.append(
					{
						"Null Family": null_family_display_map.get(family, family),
						"Dim.": dim,
						"n(E, N)": int(emp_vals.size),
						"Mean(E, N)": float(emp_vals.mean())
						if emp_vals.size
						else float("nan"),
						"n(N, N)": int(null_vals.size),
						"Mean(N, N)": float(null_vals.mean())
						if null_vals.size
						else float("nan"),
						"Mean Diff.": mean_diff,
						"p-value": p_value,
						"Sig.": _sig_stars(p_value, n_tests),
						"Bonf.": threshold,
					}
				)

		results = pd.DataFrame(rows)
		if results.empty:
			log_lines.append("No samples available for this metric.")
			continue

		summary_rows = (
			results.dropna(subset=["p-value"])
			.assign(rejected=lambda df_: df_["p-value"] < df_["Bonf."])
			.groupby("Null Family", as_index=False)["rejected"]
			.agg(Rejected="sum", Total="count")
			.reset_index()
		)

		if not summary_rows.empty:
			summary_rows["Rejected %"] = (
				100 * summary_rows["Rejected"] / summary_rows["Total"]
			)
			log_lines.append("")
			log_lines.append("SUMMARY BY NULL FAMILY (LATEX)")
			log_lines.append(
				summary_rows.to_latex(
					index=False,
					float_format="%.2f",
					formatters={"p-value": "{:,.4f}".format} if "p-value" in summary_rows.columns else None
				)
			)

		log_lines.append("")
		log_lines.append("PER-DIMENSION RESULTS (LATEX)")
		log_lines.append(
			results.to_latex(
				index=False,
				float_format="%.2f",
				formatters={"p-value": "{:,.4f}".format} if "p-value" in results.columns else None
			)
		)

	if hasattr(snakemake, "log") and snakemake.log:
		log_path = snakemake.log[0]
	else:
		log_path = snakemake.output[0] if snakemake.output else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
