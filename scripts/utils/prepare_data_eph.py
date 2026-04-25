from scripts import *
import pandas as pd
from src.seeding import initialize_seeds, get_seed_from_config

snakemake: any


def _resolve_column(columns: list[str], candidates: list[str]) -> str | None:
	for candidate in candidates:
		if candidate in columns:
			return candidate
	return None


def _resolve_feature_source(
	columns: list[str],
	configured_source: str | None,
	fallback_candidates: list[str],
) -> str | None:
	candidates: list[str] = []
	if configured_source:
		candidates.append(configured_source)
	for col in fallback_candidates:
		if col not in candidates:
			candidates.append(col)
	return _resolve_column(columns, candidates)


def main() -> None:
	initialize_seeds(get_seed_from_config(snakemake.config))
	dataset_cfg = snakemake.config["datasets"]["eph_generic"]

	id_1 = dataset_cfg["id_1"]
	id_2 = dataset_cfg["id_2"]
	id_3 = dataset_cfg["id_3"]

	df_eph = pd.read_csv(snakemake.input[0], sep=",")
	columns = df_eph.columns.tolist()

	id_caes = _resolve_column(
		columns,
		[dataset_cfg.get("id_caes"), "PP04B_COD", "PP04B COD"],
	)
	id_cno = _resolve_column(
		columns,
		[dataset_cfg.get("id_cno"), "PP04D_COD", "PP04D COD"],
	)
	if id_caes is None or id_cno is None:
		raise KeyError("Could not resolve EPH id_caes/id_cno columns in input file.")

	features = [
		dl.Feature(**feature) for feature in snakemake.config["datasets"]["features"]
	]
	feature_names = [feature.name for feature in features]
	dataset_features = dataset_cfg.get("features", {})

	df_eph = dl.lcd_load_eph_base(
		df_eph=df_eph,
		id_1=id_1,
		id_2=id_2,
		id_3=id_3,
		id_caes=id_caes,
		id_cno=id_cno,
		features=features,
		max_caes_id=10000,
	)

	feature_fallbacks: dict[str, list[str]] = {
		"sex_id": ["PP03G_COD", "PP03G"],
		"public_worker": ["CH04"],
		"total_income": ["P47T"],
		"age": ["CH06"],
		"nivel_ed": ["CH12", "NIVEL_ED"],
	}

	rename_cols: dict[str, str] = {}
	for feature_name in feature_names:
		source_col = _resolve_feature_source(
			columns=df_eph.columns.tolist(),
			configured_source=dataset_features.get(feature_name),
			fallback_candidates=feature_fallbacks.get(feature_name, []),
		)
		if source_col and source_col in df_eph.columns and source_col != feature_name:
			rename_cols[source_col] = feature_name

	id_caes_out = snakemake.config["caes"]["id"]
	id_cno_out = snakemake.config["cno"]["id"]

	rename_cols[id_caes] = id_caes_out
	rename_cols[id_cno] = id_cno_out

	if rename_cols:
		df_eph = df_eph.rename(columns=rename_cols)

	df_eph = dl.clean_enes_base_features(df_eph, features)

	output_cols = [id_1, id_2, id_3, id_caes_out, id_cno_out] + feature_names
	output_cols = [col for col in output_cols if col is not None]
	for col in output_cols:
		if col not in df_eph.columns:
			df_eph[col] = pd.NA

	df_eph = df_eph[output_cols]
	df_eph[id_caes_out] = (
		pd.to_numeric(df_eph[id_caes_out], errors="coerce").fillna(0).astype(int)
	)
	df_eph[id_cno_out] = (
		pd.to_numeric(df_eph[id_cno_out], errors="coerce").fillna(0).astype(int)
	)

	# Special case for total income which has -9 for "no income" and -9.0 for "missing"
	df_eph["total_income"] = (
		df_eph["total_income"].replace(-9.0, pd.NA).replace(-9, pd.NA)
	)

	df_eph.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
