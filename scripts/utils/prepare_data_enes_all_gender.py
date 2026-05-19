from typing import Any
from scripts import *
import pandas as pd

snakemake: Any


def _filter_by_sex(df: pd.DataFrame, sex_value: int) -> pd.DataFrame:
	return df[df["sex_id"] == sex_value].copy()


def main() -> None:
	df = pd.read_csv(snakemake.input[0])
	if "sex_id" not in df.columns:
		raise KeyError("Missing 'sex_id' in ENES data.")

	df = df[df["sex_id"].isin([1, 2])].copy()
	male_df = _filter_by_sex(df, 1)
	female_df = _filter_by_sex(df, 2)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PREPARE ENES ALL BY GENDER")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SETTINGS",
		[
			"sex_id mapping: male=1, female=2",
			"rows with other values are dropped",
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
		"ENES ALL MALE",
		row_count=len(male_df),
		column_count=len(male_df.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"ENES ALL FEMALE",
		row_count=len(female_df),
		column_count=len(female_df.columns),
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	male_df.to_csv(snakemake.output[0], index=False)
	female_df.to_csv(snakemake.output[1], index=False)


if __name__ == "__main__":
	main()
