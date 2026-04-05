from scripts import *
import matplotlib.pyplot as plt
import pandas as pd

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	enes_df = pd.read_csv(
		snakemake.input[0],
		dtype={snakemake.config["id_caes"]: int, snakemake.config["id_ciuo"]: int},
	)
	caes_id = snakemake.config["caes"]["id"]
	ciuo_id = snakemake.config["ciuo"]["id"]
	caes_df = pd.read_csv(snakemake.input[1], dtype={caes_id: int})
	ciuo_df = pd.read_csv(snakemake.input[2], dtype={ciuo_id: int})
	letra_caes = snakemake.config["caes"]["letra"]
	letra_ciuo = snakemake.config["ciuo"]["letra"]

	enes_df = pd.merge(
		enes_df,
		caes_df[[caes_id, letra_caes]],
		left_on=snakemake.config["id_caes"],
		right_on=caes_id,
		how="left",
	)
	enes_df = pd.merge(
		enes_df,
		ciuo_df[[ciuo_id, letra_ciuo]],
		left_on=snakemake.config["id_ciuo"],
		right_on=ciuo_id,
		how="left",
	)

	biadjacency = gc.build_biadjacency(
		enes_df,
		letra_caes,
		letra_ciuo,
		logscale=False,
	)

	pl.plot_heatmap(
		biadjacency,
		output_path=snakemake.output[0],
		save=True,
		figsize=tuple(snakemake.config["figsizes"]["heatmap"]),
	)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("BIADJACENCY HEATMAP")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_dataframe_info(
		log_lines,
		"ENES DATA",
		row_count=len(enes_df),
		column_count=len(enes_df.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"CAES NODELIST",
		row_count=len(caes_df),
		column_count=len(caes_df.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"CIUO NODELIST",
		row_count=len(ciuo_df),
		column_count=len(ciuo_df.columns),
	)
	log.add_notes(
		log_lines,
		"BIADJACENCY",
		[
			f"Rows (CAES labels): {biadjacency.shape[0]}",
			f"Columns (CIUO labels): {biadjacency.shape[1]}",
		],
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
