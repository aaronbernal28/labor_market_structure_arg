from scripts import *
import pandas as pd

snakemake: any


def main() -> None:
	dataset = snakemake.params[0]

	dataset_input = snakemake.config["datasets"][dataset]
	dataset_default = snakemake.config["enes"]
	default_cfg = snakemake.config["datasets"][dataset_default]

	SOURCE = dataset_input["source"]
	URL = dataset_input["url"]
	ID_1 = dataset_input["id_1"]
	ID_2 = dataset_input["id_2"]
	ID_CAES = dataset_input["id_caes"]
	ID_CIUO = dataset_input["id_ciuo"]

	FEATURES = [
		dl.Feature(**feature) for feature in snakemake.config["datasets"]["features"]
	]
	feature_names = [feature.name for feature in FEATURES]
	dataset_features = dataset_input.get("features", {})
	default_features = default_cfg.get("features", {})

	df_enes = pd.read_csv(
		SOURCE if SOURCE else URL, sep=";" if dataset == "enes_2019" else ","
	)
	df_enes = dl.lcd_load_enes_base(
		df_enes=df_enes,
		id_1=ID_1,
		id_2=ID_2,
		id_caes=ID_CAES,
		id_ciuo=ID_CIUO,
		features=FEATURES,
	)

	# Normalize IDs and features to the default dataset schema (enes_2019 in this project).
	rename_cols = {}
	if ID_CAES and default_cfg.get("id_caes") and ID_CAES != default_cfg["id_caes"]:
		rename_cols[ID_CAES] = default_cfg["id_caes"]
	if ID_CIUO and default_cfg.get("id_ciuo") and ID_CIUO != default_cfg["id_ciuo"]:
		rename_cols[ID_CIUO] = default_cfg["id_ciuo"]
	if ID_1 and default_cfg.get("id_1") and ID_1 != default_cfg["id_1"]:
		rename_cols[ID_1] = default_cfg["id_1"]
	if ID_2 and default_cfg.get("id_2") and ID_2 != default_cfg["id_2"]:
		rename_cols[ID_2] = default_cfg["id_2"]

	for feature_name in feature_names:
		source_col = dataset_features.get(feature_name)
		target_col = default_features.get(feature_name)
		if (
			source_col
			and target_col
			and source_col != target_col
			and source_col in df_enes.columns
		):
			rename_cols[source_col] = target_col

	if rename_cols:
		df_enes = df_enes.rename(columns=rename_cols)

	# Convert default feature source columns to canonical names from config datasets.features.
	for feature_name in feature_names:
		default_source_col = default_features.get(feature_name)
		if (
			default_source_col
			and default_source_col in df_enes.columns
			and default_source_col != feature_name
		):
			df_enes = df_enes.rename(columns={default_source_col: feature_name})

	df_enes = dl.clean_enes_base_features(df_enes, FEATURES)

	output_id_1 = default_cfg["id_1"]
	output_id_2 = default_cfg["id_2"]
	output_id_caes = default_cfg["id_caes"]
	output_id_ciuo = default_cfg["id_ciuo"]

	if output_id_2 and output_id_2 not in df_enes.columns:
		df_enes[output_id_2] = pd.NA

	output_cols = [
		output_id_1,
		output_id_2,
		output_id_caes,
		output_id_ciuo,
	] + feature_names
	output_cols = [col for col in output_cols if col is not None]
	for col in output_cols:
		if col not in df_enes.columns:
			df_enes[col] = pd.NA
	df_enes = df_enes[output_cols]

	df_enes.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
