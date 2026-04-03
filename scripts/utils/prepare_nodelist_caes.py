from scripts import *
import pandas as pd

snakemake: any


CHARACTERISTIC_COLUMNS = [
	"n_obs",
	"total_workers_weighted",
	"income_mean",
	"income_min",
	"income_q1",
	"income_median",
	"income_q3",
	"income_max",
	"income_std",
	"female_pct",
	"male_pct",
	"public_sector_pct",
]


def main() -> None:
	df_enes = pd.read_csv(snakemake.input[0])
	class_name = snakemake.params[0]
	nodelist = snakemake.config[class_name]
	id = nodelist["id"]
	max_caes_id = snakemake.config["max_caes_id"]

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
		)
	else:
		raise ValueError(f"Unsupported nodelist type: {class_name}")

	new_features = nc.compute_group_characteristics(enes_df=df_enes, col_group=id)
	for col in CHARACTERISTIC_COLUMNS:
		if col not in new_features.columns:
			new_features[col] = pd.NA

	df_nodelist = nc.attach_group_characteristics(
		nodelist_df=df_nodelist,
		features_df=new_features,
		keep_columns=CHARACTERISTIC_COLUMNS,
	)
	df_nodelist = df_nodelist.reset_index()

	df_nodelist.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
