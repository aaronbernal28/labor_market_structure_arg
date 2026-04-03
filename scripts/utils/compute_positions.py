from scripts import *
import networkx as nx
import pandas as pd

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]
	seed = int(snakemake.config["seed"])
	dataset_df = pd.read_csv(snakemake.input[1], dtype={id_col: int})

	pos = gc.get_projection_positions(
		graph,
		seed=seed,
		spring_layout_iterations=1000,
		spring_layout_k=None,
		rotate=False,
		method="auto",
	)
	dataset_df = dl.insert_positions(dataset_df, pos, id_col=id_col)

	dataset_df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
