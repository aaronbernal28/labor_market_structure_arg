from typing import Any
from scripts import *
import pandas as pd

snakemake: Any


def main() -> None:
	df_2019 = pd.read_csv(snakemake.input[0])
	df_2021 = pd.read_csv(snakemake.input[1])
	df = pd.concat([df_2019, df_2021], ignore_index=True)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PREPARE ENES ALL")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_dataframe_info(
		log_lines,
		"ENES 2019",
		row_count=len(df_2019),
		column_count=len(df_2019.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"ENES 2021",
		row_count=len(df_2021),
		column_count=len(df_2021.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"ENES ALL",
		row_count=len(df),
		column_count=len(df.columns),
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
