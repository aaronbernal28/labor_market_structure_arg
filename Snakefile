configfile: "config.yaml"

DATASETS = ["enes_all"]
NODELIST = ["caes", "ciuo"]
WEIGHT_FUNCTIONS = ["dot_product", "weighted_hidalgo_weight"]
ALGORITHMS = ["louvain"]
VARIABLES = ["sex_id", "public_worker", "total_income"]
DISCRETE_FEATURES = ["grupo", "louvain"] # in nodelist data
CONTINUOUS_FEATURES = ["income_mean", "female_pct", "public_sector_pct"]

LAYOUTS = ["spring_layout"]
CLASSES = ["caes", "ciuo"]
LOGSCALES = ["false"]
ALPHAS = ["0.05","1.00"]

wildcard_constraints:
	dataset = "|".join(DATASETS),
	class_ = "|".join(CLASSES),
	logscale = "|".join(LOGSCALES),
	weight_function = "|".join(WEIGHT_FUNCTIONS),
	algorithm = "|".join(ALGORITHMS),
	alpha = "|".join(ALPHAS)

rule all:
	input:
		expand(
			"images/00_aed_report/aed_{dataset}_top_sectors.png",
			dataset=DATASETS,
		),
		expand(
			"images/00_aed_report/aed_{dataset}_top_occupations.png",
			dataset=DATASETS,
		),
		expand(
			"images/01_biadjacency_matrix_heatmap/biadjacency_matrix_heatmap_{dataset}.png",
			dataset=DATASETS,
		),
		expand(
			"images/02_bipartite_plot_by_groups/bipartite_plot_by_groups_{dataset}_{logscale}.png",
			dataset=DATASETS,
			logscale=LOGSCALES,
		),
		expand(
			"images/02_bipartite_plot_by_groups/bipartite_degree_dist_{dataset}_{logscale}.png",
			dataset=DATASETS,
			logscale=LOGSCALES,
		),
		expand(
			"images/03_projection_plot_by_groups/projection_plot_by_groups_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}_pos_{algorithm}_{discrete_feature}.png",
			dataset=DATASETS,
			logscale=LOGSCALES,
			class_=CLASSES,
			weight_function=WEIGHT_FUNCTIONS,
			alpha=ALPHAS,
			algorithm=ALGORITHMS,
			discrete_feature=DISCRETE_FEATURES,
		),
		expand(
			"images/03_projection_plot_gradient/projection_plot_gradient_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}_pos_{discrete_feature}.png",
			dataset=DATASETS,
			logscale=LOGSCALES,
			class_=CLASSES,
			weight_function=WEIGHT_FUNCTIONS,
			alpha=ALPHAS,
			discrete_feature=CONTINUOUS_FEATURES,
		),
		expand(
			"images/_07_alpha_sensitivity/filtered_alpha_sensitivity_{dataset}_{logscale}_{class_}_{weight_function}.png",
			dataset=DATASETS,
			class_=CLASSES,
			logscale=LOGSCALES,
			weight_function=WEIGHT_FUNCTIONS,
		),
		expand(
			"images/05_edge_weight_correlation/edge_weight_correlation_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}_pos_{algorithm}_{continuous_feature}.png",
			dataset=DATASETS,
			class_=CLASSES,
			logscale=LOGSCALES,
			weight_function=WEIGHT_FUNCTIONS,
			algorithm=ALGORITHMS,
			alpha=ALPHAS,
			continuous_feature=CONTINUOUS_FEATURES,
		),
		expand(
			"images/06_sankey_plot/sankey_plot_{dataset}.png",
			dataset=DATASETS,
		)


rule _00_aed_report:
	'''AED: Análisis Exploratorio de Datos on ENES datasets'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/nodelist_caes_{dataset}.csv",
		"data/processed/nodelist_ciuo_{dataset}.csv"
	output:
		"images/00_aed_report/aed_{dataset}_top_sectors.png",
		"images/00_aed_report/aed_{dataset}_top_occupations.png"
	script:
		"scripts/00_aed_report.py"


rule _01_biadjacency_matrix_heatmap:
	'''Cross tabular matrix on frequency in the ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/nodelist_caes_{dataset}.csv",
		"data/processed/nodelist_ciuo_{dataset}.csv"
	output:
		"images/01_biadjacency_matrix_heatmap/biadjacency_matrix_heatmap_{dataset}.png"
	script:
		"scripts/01_biadjacency_matrix_heatmap.py"


rule _02_bipartite_plot_by_groups:
	'''Plot bipartite graph from graph.'''
	input:
		"data/graphs/bipartite_{dataset}_{logscale}.gexf",
		"data/processed/nodelist_caes_{dataset}.csv",
		"data/processed/nodelist_ciuo_{dataset}.csv",
	output:
		"images/02_bipartite_plot_by_groups/bipartite_plot_by_groups_{dataset}_{logscale}.png",
		"images/02_bipartite_plot_by_groups/bipartite_degree_dist_{dataset}_{logscale}.png"
	script:
		"scripts/02_bipartite_plot_by_groups.py"


rule _03_projection_plot_by_groups:
	'''Plot projection graph from graph.
	Example:
	snakemake -j1 images/03_projection_plot_by_groups/projection_plot_by_groups_enes_all_false_ciuo_weighted_hidalgo_weight_1.0000_pos_louvain_louvain.png'''
	input:
		"data/processed/nodelist_{class_}_{dataset}_{logscale}_{weight_function}_{alpha}_pos_{algorithm}.csv",
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}.gexf"
	output:
		"images/03_projection_plot_by_groups/projection_plot_by_groups_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}_pos_{algorithm}_{discrete_feature}.png"
	script:
		"scripts/03_projection_plot_by_groups.py"


rule _03_projection_plot_gradient:
	'''Plot projection graph from graph.'''
	input:
		"data/processed/nodelist_{class_}_{dataset}_{logscale}_{weight_function}_{alpha}_pos.csv",
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}.gexf"
	output:
		"images/03_projection_plot_gradient/projection_plot_gradient_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}_pos_{discrete_feature}.png"
	script:
		"scripts/03_projection_plot_gradient.py"


rule _04_walt_test:
	'''Walt test on datasets enes_2019 vs enes_2021.'''
	input:
		"data/processed/enes_2019.csv",
		"data/processed/enes_2021.csv",
		"data/processed/nodelist_caes_enes_2021.csv", # irrelevant which enes dataset we use
		"data/processed/nodelist_ciuo_enes_2021.csv"
	output:
		"images/04_walt_test/walt_test_bootstrap_se.png",
		"images/04_walt_test/walt_test_delta.png",
		"images/04_walt_test/walt_test_pvalue_detailed.png",
		"images/04_walt_test/walt_test_pvalue_summary.log"
	script:
		"scripts/04_walt_test.py"


rule _05_edge_weight_correlation:
	'''Correlation between edge weights in projection graphs.'''
	input:
		"data/processed/nodelist_{class_}_{dataset}_{logscale}_{weight_function}_{alpha}_pos_{algorithm}.csv", # get community class from colum louvain
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}.gexf" # extract edge weights from graph
	output:
		"images/05_edge_weight_correlation/edge_weight_correlation_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}_pos_{algorithm}_{continuous_feature}.png"
	script:
		"scripts/05_edge_weight_correlation.py"


rule _06_sankey_plot:
	'''Sankey plot of communities.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/nodelist_caes_{dataset}.csv",
		"data/processed/nodelist_ciuo_{dataset}.csv"
	output:
		"images/06_sankey_plot/sankey_plot_{dataset}.png"
	script:
		"scripts/06_sankey_plot.py"


rule _07_alpha_sensitivity:
	input:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}.gexf"
	output:
		"images/_07_alpha_sensitivity/filtered_alpha_sensitivity_{dataset}_{logscale}_{class_}_{weight_function}.png"
	params:
		alpha=0.05,
		algorithm="louvain"
	script:
		"scripts/07_alpha_sensitivity.py"

include: "rules/00_prepare.smk"
include: "rules/01_bipartite.smk"
include: "rules/02_projection.smk"
include: "rules/03_communities.smk"
