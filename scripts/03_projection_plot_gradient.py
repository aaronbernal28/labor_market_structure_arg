from scripts import *
import networkx as nx


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])
	layout = getattr(snakemake.params, "layout", None) or snakemake.wildcards.layout

	fig = lcd_plot_projection_gradient(
		graph, layout=layout, title=f"Projection gradient ({layout})"
	)
	fig.savefig(snakemake.output[0], bbox_inches="tight")


if __name__ == "__main__":
	main()
