configfile: "config.yaml"

DATASETS = ["enes_all", "enes_2019", "enes_2021"]
METADATA = ["nodelist_caes", "nodelist_ciuo"]
WEIGHTS = ["vecinos_compartidos", "hub_depressed", "dot_product", "hidalgo"]
ALGORITHMS = ["louvain", "leiden", "infomap"]
FEATURES = ["sex_id", "public_worker", "total_income"]
LAYOUTS = ["spring_layout"]

rule all:
	input:
		expand(
			"images/02_projection_by_groups/02_projection_{dataset}_{layout}.png",
			dataset=DATASETS,
		)


rule _00_aed_report:
	'''AED: Análisis Exploratorio de Datos on ENES datasets'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{METADATA[0]}_{dataset}.csv",
		"data/processed/{METADATA[1]}_{dataset}.csv"
	output:
		"images/00_aed_report/aed_{dataset}_top_sectors.png",
		"images/00_aed_report/aed_{dataset}_top_occupations.png",
		"reports/00_aed_report/aed_{dataset}.csv"
	script:
		"scripts/00_aed_report.py"


rule _01_biadjacency_matrix_heatmap:
	'''Cross tabular matrix on frequency in the ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{METADATA[0]}.csv",
		"data/processed/{METADATA[1]}.csv"
	output:
		"images/01_biadjacency_matrix_heatmap/biadjacency_matrix_heatmap_{dataset}.png"
	script:
		"scripts/01_biadjacency_matrix_heatmap.py"


rule _02_bipartite_plot_by_groups:
	'''Plot bipartite graph from graph.'''
	input:
		"data/graphs/bipartite_{dataset}.gexf"
	output:
		"images/02_bipartite_plot_by_groups/bipartite_layout_by_groups{dataset}_{layout}.png",
		"images/02_bipartite_plot_by_groups/bipartite_degree_dist_{dataset}_{layout}.png"
	script:
		"scripts/02_bipartite_plot_by_groups.py"


rule _03_projection_plot_by_groups:
	'''Plot projection graph from graph.'''
	input:
		"data/graphs/projection_{dataset}_{class}_{logscale}_{weight}_{algorithm}_{alpha}.gexf"
	output:
		"images/03_projection_plot_by_groups/projection_plot_by_groups_{dataset}_{logscale}_{weight}_{algorithm}_{layout}_{discrete_feature}.png"
	script:
		"scripts/03_projection_plot_by_groups.py"


rule _03_projection_plot_gradient:
	'''Plot projection graph from graph.'''
	input:
		"data/graphs/projection_{dataset}_{class}_{logscale}_{weight}_{algorithm}_{alpha}.gexf"
	output:
		"images/03_projection_plot_gradient/projection_plot_gradient_{dataset}_{logscale}_{weight}_{algorithm}_{layout}_{continuous_feature}.png"
	script:
		"scripts/03_projection_plot_gradient.py"


rule _04_walt_test:
	'''Walt test on datasets enes_2019 vs enes_2021.'''
	input:
		"data/processed/enes_2019.csv",
		"data/processed/enes_2021.csv"
	output:
		"images/04_walt_test/walt_test_bootstrap_se.csv",
		"images/04_walt_test/walt_test_delta.csv",
		"images/04_walt_test/walt_test_pvalue_detailed.csv"
	script:
		"scripts/04_walt_test.py"


rule _05_edge_weight_correlation:
	'''Correlation between edge weights in projection graphs.'''
	input:
		"data/graphs/projection_{dataset}_{class}_{logscale}_{weight}_{algorithm}_{alpha}.gexf"
	output:
		"images/05_edge_weight_correlation/edge_weight_correlation_{dataset}_{class}_{logscale}_{weight}_{algorithm}_{alpha}_{feature}.png"
	script:
		"scripts/05_edge_weight_correlation.py"


rule _06_sankey_plot:
	'''Sankey plot of communities.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{METADATA[0]}.csv",
		"data/processed/{METADATA[1]}.csv"
	output:
		"images/06_sankey_plot/sankey_plot_{dataset}.png"
	script:
		"scripts/06_sankey_plot.py"


include: "rules/00_prepare.smk"
include: "rules/01_bipartite.smk"
include: "rules/02_projection.smk"
include: "rules/03_communities.smk"
