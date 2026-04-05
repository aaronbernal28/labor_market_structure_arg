configfile: "config.yaml"


DATASETS = ["enes_all"]
NODELIST = ["caes", "ciuo"]
WEIGHT_FUNCTIONS = ["hidalgo"] # "unweighted_hidalgo", "dot_product", "cosine_similarity"
ALGORITHMS = ["louvain"]#, "infomap"]
#VARIABLES = ["sex_id", "public_worker", "total_income"]
DISCRETE_FEATURES = ["grupo", "community"] # in nodelist data
CONTINUOUS_FEATURES = ["female_pct"] # in nodelist data

LAYOUTS = ["spring_layout"]
CLASSES = ["caes", "ciuo"]
ALPHAS = ["0.30","1.00"]

wildcard_constraints:
	dataset = "|".join(DATASETS),
	class_ = "|".join(CLASSES),
	weight_function = "|".join(WEIGHT_FUNCTIONS),
	algorithm = "|".join(ALGORITHMS),
	alpha = "|".join(ALPHAS)

rule all:
	input:
		expand(
			"images/{dataset}/00_aed_report/aed_top_sectors.png",
			dataset=DATASETS,
		),
		expand(
			"images/{dataset}/00_aed_report/aed_top_occupations.png",
			dataset=DATASETS,
		),
		expand(
			"images/{dataset}/01_biadjacency_matrix_heatmap/biadjacency_matrix_heatmap.png",
			dataset=DATASETS,
		),
		expand(
			"images/{dataset}/02_bipartite_plot_by_groups/bipartite_plot_by_groups.png",
			dataset=DATASETS,
		),
		expand(
			"images/{dataset}/02_bipartite_plot_by_groups/bipartite_degree_dist.png",
			dataset=DATASETS,
		),
		expand(
			"images/{dataset}/{class_}/03_projection_plot_by_groups/_{weight_function}_{alpha}_pos_{algorithm}_{discrete_feature}.png",
			dataset=DATASETS,
			class_=CLASSES,
			weight_function=WEIGHT_FUNCTIONS,
			alpha=ALPHAS,
			algorithm=ALGORITHMS,
			discrete_feature=DISCRETE_FEATURES,
		),
		expand(
			"images/{dataset}/{class_}/03_projection_plot_gradient/_{weight_function}_{alpha}_pos_{discrete_feature}.png",
			dataset=DATASETS,
			class_=CLASSES,
			weight_function=WEIGHT_FUNCTIONS,
			alpha=ALPHAS,
			discrete_feature=CONTINUOUS_FEATURES,
		),
		"images/enes_all/04_walt_test/walt_test_bootstrap_se.png",
		expand(
			"images/{dataset}/{class_}/07_alpha_sensitivity/_{weight_function}.png",
			dataset=DATASETS,
			class_=CLASSES,
			weight_function=WEIGHT_FUNCTIONS,
		),
		expand(
			"images/{dataset}/{class_}/05_edge_weight_correlation/_{weight_function}_{alpha}_pos_{algorithm}_{continuous_feature}.png",
			dataset=DATASETS,
			class_=CLASSES,
			weight_function=WEIGHT_FUNCTIONS,
			algorithm=ALGORITHMS,
			alpha=ALPHAS,
			continuous_feature=CONTINUOUS_FEATURES,
		),
		expand(
			"images/{dataset}/{class_}/03_communities/_distribution_{weight_function}_{alpha}_{algorithm}.png",
			dataset=DATASETS,
			class_=CLASSES,
			weight_function=WEIGHT_FUNCTIONS,
			alpha=ALPHAS,
			algorithm=ALGORITHMS,
		),
		expand(
			"images/{dataset}/06_sankey_plot/sankey_plot.png",
			dataset=DATASETS,
		)


rule _00_aed_report:
	'''AED: Análisis Exploratorio de Datos on ENES datasets'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{dataset}/nodelist_caes.csv",
		"data/processed/{dataset}/nodelist_ciuo.csv"
	output:
		"images/{dataset}/00_aed_report/aed_top_sectors.png",
		"images/{dataset}/00_aed_report/aed_top_occupations.png"
	log:
		"images/{dataset}/00_aed_report/aed_report.log"
	script:
		"scripts/00_aed_report.py"


rule _01_biadjacency_matrix_heatmap:
	'''Cross tabular matrix on frequency in the ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{dataset}/nodelist_caes.csv",
		"data/processed/{dataset}/nodelist_ciuo.csv"
	output:
		"images/{dataset}/01_biadjacency_matrix_heatmap/biadjacency_matrix_heatmap.png"
	log:
		"images/{dataset}/01_biadjacency_matrix_heatmap/biadjacency_matrix_heatmap.log"
	script:
		"scripts/01_biadjacency_matrix_heatmap.py"


rule _02_bipartite_plot_by_groups:
	'''Plot bipartite graph from graph.'''
	input:
		"data/graphs/{dataset}/bipartite.gexf",
		"data/processed/{dataset}/nodelist_caes.csv",
		"data/processed/{dataset}/nodelist_ciuo.csv",
	output:
		"images/{dataset}/02_bipartite_plot_by_groups/bipartite_plot_by_groups.png",
		"images/{dataset}/02_bipartite_plot_by_groups/bipartite_degree_dist.png"
	log:
		"images/{dataset}/02_bipartite_plot_by_groups/bipartite_plot_by_groups.log"
	script:
		"scripts/02_bipartite_plot_by_groups.py"


rule _03_projection_plot_by_groups:
	'''Plot projection graph from graph.
	Example:
	snakemake -j1 images/03_projection_plot_by_groups/_enes_all_false_ciuo_weighted_hidalgo_weight_1.0000_pos_louvain_community.png'''
	input:
		"data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos_{algorithm}.csv",
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf"
	output:
		"images/{dataset}/{class_}/03_projection_plot_by_groups/_{weight_function}_{alpha}_pos_{algorithm}_{discrete_feature}.png"
	log:
		"images/{dataset}/{class_}/03_projection_plot_by_groups/_{weight_function}_{alpha}_pos_{algorithm}_{discrete_feature}.log"
	script:
		"scripts/03_projection_plot_by_groups.py"


rule _03_projection_plot_gradient:
	'''Plot projection graph from graph.'''
	input:
		"data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos.csv",
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf"
	output:
		"images/{dataset}/{class_}/03_projection_plot_gradient/_{weight_function}_{alpha}_pos_{discrete_feature}.png"
	log:
		"images/{dataset}/{class_}/03_projection_plot_gradient/_{weight_function}_{alpha}_pos_{discrete_feature}.log"
	script:
		"scripts/03_projection_plot_gradient.py"


rule _04_walt_test:
	'''Walt test on datasets enes_2019 vs enes_2021.'''
	input:
		"data/processed/enes_2019.csv",
		"data/processed/enes_2021.csv",
		"data/processed/enes_all/nodelist_caes.csv", # irrelevant which enes dataset we use
		"data/processed/enes_all/nodelist_ciuo.csv"
	output:
		"images/enes_all/04_walt_test/walt_test_bootstrap_se.png",
		"images/enes_all/04_walt_test/walt_test_delta.png",
		"images/enes_all/04_walt_test/walt_test_pvalue_detailed.png"
	log:
		"images/enes_all/04_walt_test/walt_test_pvalue_summary.log"
	script:
		"scripts/04_walt_test.py"


rule _05_edge_weight_correlation:
	'''Correlation between edge weights in projection graphs.'''
	input:
		"data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos_{algorithm}.csv", # get community class from column louvain
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf" # extract edge weights from graph
	output:
		"images/{dataset}/{class_}/05_edge_weight_correlation/_{weight_function}_{alpha}_pos_{algorithm}_{continuous_feature}.png"
	log:
		"images/{dataset}/{class_}/05_edge_weight_correlation/_{weight_function}_{alpha}_pos_{algorithm}_{continuous_feature}.log"
	script:
		"scripts/05_edge_weight_correlation.py"


rule _06_sankey_plot:
	'''Sankey plot of communities.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{dataset}/nodelist_caes.csv",
		"data/processed/{dataset}/nodelist_ciuo.csv"
	output:
		"images/{dataset}/06_sankey_plot/sankey_plot.png"
	log:
		"images/{dataset}/06_sankey_plot/sankey_plot.log"
	script:
		"scripts/06_sankey_plot.py"


rule _07_alpha_sensitivity:
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		"images/{dataset}/{class_}/07_alpha_sensitivity/_{weight_function}.png"
	params:
		alpha=0.30,
		algorithm="louvain"
	log:
		"images/{dataset}/{class_}/07_alpha_sensitivity/_{weight_function}.log"
	script:
		"scripts/07_alpha_sensitivity.py"

include: "rules/00_prepare.smk"
include: "rules/01_bipartite.smk"
include: "rules/02_projection.smk"
include: "rules/03_communities.smk"
