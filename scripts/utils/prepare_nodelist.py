from typing import Any
from scripts import *
import pandas as pd

snakemake: Any


CHARACTERISTIC_COLUMNS = [
	"n_obs",
	"total_workers_weighted",
	"income_mean",
	"income_std",
	"income_median",
	"female_pct",
	"male_pct",
	"public_sector_pct",
	"nivel_ed_mean",
	"nivel_ed_std",
	"age_mean",
	"age_std",
	"age_median",
]


def main() -> None:
	df_enes = pd.read_csv(
		snakemake.input[0],
		dtype={
			snakemake.config["caes"]["id"]: int,
			snakemake.config["ciuo"]["id"]: int,
		},
	)
	class_name = snakemake.wildcards["class_"]
	nodelist = snakemake.config[class_name]
	id = nodelist["id"]
	max_caes_id = snakemake.config["max_caes_id"]
	palette_map = utils.get_config_section(snakemake.config, "palette")

	if class_name == "caes":
		df_nodelist = dl.load_nodelist_caes(
			caes_path=nodelist["source"],
			caes_id=nodelist["id"],
			caes_letra_col=nodelist["letra"],
			caes_ag_col=nodelist["grupo"],
			caes_label_color_col=nodelist["label_color"],
			caes_letra_color_col=nodelist["letra_color"],
			caes_ag_color_col=nodelist["grupo_color"],
		)
	elif class_name == "ciuo":
		df_nodelist = dl.load_nodelist_ciuo(
			ciuo_path=nodelist["source"],
			ciuo_id=nodelist["id"],
			max_caes_id=max_caes_id,
			ciuo_letra_col=nodelist["letra"],
			ciuo_3cat_col=nodelist["grupo"],
			ciuo_label_color_col=nodelist["label_color"],
			ciuo_letra_color_col=nodelist["letra_color"],
			ciuo_3cat_color_col=nodelist["grupo_color"],
			palette=palette_map,
		)
	else:
		raise ValueError(f"Unsupported nodelist type: {class_name}")

	new_features = nc.compute_group_characteristics(enes_df=df_enes, col_group=id)
	for col in CHARACTERISTIC_COLUMNS:
		if col not in new_features.columns and col not in df_nodelist.columns:
			new_features[col] = pd.NA

	df_nodelist = nc.attach_group_characteristics(
		nodelist_df=df_nodelist,
		features_df=new_features,
		keep_columns=CHARACTERISTIC_COLUMNS + ["original_" + id],
	)
	df_nodelist = df_nodelist.reset_index()

	df_nodelist[id] = df_nodelist[id].astype(int)

	if "original_" + id in df_nodelist.columns:
		cols = [id, "original_" + id] + [
			col for col in df_nodelist.columns if col not in (id, "original_" + id)
		]
		df_nodelist = df_nodelist[cols]

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PREPARE NODELIST")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SETTINGS",
		[
			f"Class: {class_name}",
			f"ID column: {id}",
			f"Characteristic columns: {len(CHARACTERISTIC_COLUMNS)}",
		],
	)
	log.add_dataframe_info(
		log_lines,
		"ENES DATA",
		row_count=len(df_enes),
		column_count=len(df_enes.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"NODELIST OUTPUT",
		row_count=len(df_nodelist),
		column_count=len(df_nodelist.columns),
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	df_nodelist.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
