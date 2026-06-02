from typing import Any
from pathlib import Path

from scripts import *
import pandas as pd
import src.topology as topo

snakemake: Any


def _extract_null_meta(path: str) -> tuple[str, int]:
	path_obj = Path(path)
	null_model = path_obj.parent.name
	stem = path_obj.stem
	parts = stem.split("_")
	index = -1
	for part in reversed(parts):
		if part.isdigit():
			index = int(part)
			break
	return null_model, index


def main() -> None:
	empirical_path = snakemake.input[0]
	null_paths = list(snakemake.input[1:])
	null_meta = [_extract_null_meta(path) for path in null_paths]
	null_model_labels = [f"{model}_{index}" for model, index in null_meta]
	null_by_dims = [topo.load_diagrams_by_dimension(path) for path in null_paths]

	empirical_by_dim = topo.load_diagrams_by_dimension(empirical_path)
	rows: list[dict[str, object]] = []

	for i, null_path in enumerate(null_paths):
		empirical_dgms, null_dgms = topo.align_diagrams(empirical_by_dim, null_by_dims[i])

		print(f"Comparing empirical diagram to null model {null_model_labels[i]}...")

		for dim in range(len(empirical_dgms)):
			if empirical_dgms[dim].size == 0 and null_dgms[dim].size == 0:
				# Both diagrams are empty; distance is zero.
				#bottleneck = 0.0
				wasserstein = 0.0
			else:
				#bottleneck = topo.bottleneck_distance(empirical_dgms[dim], null_dgms[dim])
				wasserstein = topo.wasserstein_distance(empirical_dgms[dim], null_dgms[dim])

			rows.append(
				{
					"model_1": Path(empirical_path).name,
					"model_2": null_model_labels[i],
					"dimension": dim,
					#"bottleneck": bottleneck,
					"wasserstein": wasserstein,
				}
			)

		print(f"Comparing null model {null_model_labels[i]} to other null models...")

		for j, null_path_2 in enumerate(null_paths):
			if i >= j:
				continue
			null_dgms_2, null_dgms_1 = topo.align_diagrams(null_by_dims[i], null_by_dims[j])
			for dim in range(len(null_dgms_1)):
				#bottleneck = topo.bottleneck_distance(null_dgms_1[dim], null_dgms_2[dim])
				wasserstein = topo.wasserstein_distance(null_dgms_1[dim], null_dgms_2[dim])
				rows.append(
					{
						"model_1": null_model_labels[i],
						"model_2": null_model_labels[j],
						"dimension": dim,
						#"bottleneck": bottleneck,
						"wasserstein": wasserstein,
					}
				)

	output_path = Path(snakemake.output[0])
	pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
	main()
