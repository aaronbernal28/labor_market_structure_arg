from typing import Any
from scripts import *
import pandas as pd

snakemake: Any

WHITE_COLLAR = "1. Trabajadores  no manuales"
BLUE_COLLAR = "3. Trabajadores manuales"


def _filter_by_collar(df: pd.DataFrame, ciuo3cat_value: str) -> pd.DataFrame:
	return df[df["ciuo3cat"] == ciuo3cat_value].copy()


def main() -> None:
	df = pd.read_csv(snakemake.input[0])

	ciuo_id_col = snakemake.config["ciuo"]["id"]  # "v183ciuo"

	# Use the processed enes_all nodelist which carries matching 5-digit CIUO codes
	# (the raw nodelist uses a different 4-digit encoding that doesn't overlap with enes_all)
	nodelist_ciuo = pd.read_csv(
		"data/processed/enes_all/nodelist_ciuo.csv",
		usecols=[ciuo_id_col, "ciuo3cat"],
		dtype={ciuo_id_col: int},
	)

	df = df.merge(nodelist_ciuo, on=ciuo_id_col, how="left")

	valid_cats = [WHITE_COLLAR, BLUE_COLLAR]
	df = df[df["ciuo3cat"].isin(valid_cats)].copy()

	white_collar_df = _filter_by_collar(df, WHITE_COLLAR)
	blue_collar_df = _filter_by_collar(df, BLUE_COLLAR)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PREPARE ENES ALL BY COLLAR")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SETTINGS",
		[
			f"White collar filter: ciuo3cat == '{WHITE_COLLAR}'",
			f"Blue collar filter:  ciuo3cat == '{BLUE_COLLAR}'",
			"Rows with other ciuo3cat values (e.g. '2. Trabajadores de servicios y ventas') are dropped.",
			"ciuo3cat mapped from data/processed/enes_all/nodelist_ciuo.csv",
		],
	)
	log.add_dataframe_info(
		log_lines,
		"ENES ALL (FILTERED)",
		row_count=len(df),
		column_count=len(df.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"ENES ALL WHITE COLLAR",
		row_count=len(white_collar_df),
		column_count=len(white_collar_df.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"ENES ALL BLUE COLLAR",
		row_count=len(blue_collar_df),
		column_count=len(blue_collar_df.columns),
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	white_collar_df.to_csv(snakemake.output[0], index=False)
	blue_collar_df.to_csv(snakemake.output[1], index=False)


if __name__ == "__main__":
	main()
