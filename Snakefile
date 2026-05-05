configfile: "config.yaml"
from numpy import logspace


DATASETS = ["enes_all", "enes_2019", "enes_2021"]
NODELIST = ["caes", "ciuo"]
WEIGHT_FUNCTIONS = ["hidalgo", "unweighted_hidalgo", "dot_product", "cosine"]
ALGORITHMS = ["leiden"]
ALGORITHMS_ALL = ["louvain", "leiden", "infomap"]
VARIABLES = ["sex_id", "public_worker", "total_income", "education_mean"]
DISCRETE_FEATURES = ["grupo", "community"] # in nodelist data
CONTINUOUS_FEATURES = ["female_pct", "public_sector_pct", "income_median", "income_mean", "nivel_ed_mean", "age_mean"] # in nodelist data
LAYOUTS = ["spring_layout"]
CLASSES = ["caes", "ciuo"]
CLASSES_ALL = ["caes", "ciuo", "cno"]
ALPHA_CAES = ["0.0043"]
ALPHA_CIUO = ["0.0093"]
ALPHA_EPH = (logspace(-10, 0, 60).round(4).astype(str)).tolist()
ALPHAS_ALL = ALPHA_CAES + ALPHA_CIUO + ALPHA_EPH
TOPO_METHOD = ["shortest_path", "disparity_filtration"]
EPH_YEARS = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
EPH_FILES = glob_wildcards("data/raw/eph/{eph_file}.csv").eph_file
DATASETS_ALL = DATASETS + EPH_FILES

wildcard_constraints:
	dataset = "|".join(DATASETS_ALL),
	class_ = "|".join(CLASSES_ALL),
	weight_function = "|".join(WEIGHT_FUNCTIONS),
	algorithm = "|".join(ALGORITHMS_ALL),
	alpha = "|".join(ALPHAS_ALL),
	topo_method = "|".join(TOPO_METHOD),
	eph_file = "|".join(EPH_FILES)


rule all:
	input:
		"images/enes_all/00_aed_report/aed_top_sectors.png",
		"images/enes_all/00_aed_report/aed_top_occupations.png",
		"images/enes_all/00_aed_report/aed_distributions.png",
		"images/enes_all/00_aed_report/aed_correlation_matrix.png",
		"images/enes_all/01_biadjacency_matrix_heatmap/biadjacency_matrix_heatmap.png",
		"images/enes_all/02_bipartite_plot_by_groups/bipartite_plot_by_groups.png",
		"images/enes_all/02_bipartite_plot_by_groups/bipartite_degree_dist.png",
		"images/enes_all/04_walt_test/walt_test_bootstrap_se.png",
		"images/enes_all/06_sankey_plot/sankey_plot.png",
		expand(
			"images/enes_all/{class_}/07_alpha_sensitivity/_{weight_function}.png",
			class_=CLASSES,
			weight_function=["hidalgo"],
			algorithm=ALGORITHMS,
		),
		expand(
			["images/enes_all/caes/03_resolution_sensitivity/_catplots_{weight_function}_{alpha_caes}.png",
			"images/enes_all/ciuo/03_resolution_sensitivity/_catplots_{weight_function}_{alpha_ciuo}.png"],
			class_=CLASSES,
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
		),
		expand(
			["images/enes_all/caes/03_projection_plot_by_groups/_{weight_function}_{alpha_caes}_pos_{algorithm}_{discrete_feature}.png",
			"images/enes_all/ciuo/03_projection_plot_by_groups/_{weight_function}_{alpha_ciuo}_pos_{algorithm}_{discrete_feature}.png"],
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			algorithm=ALGORITHMS,
			discrete_feature=["community"],
		),
		expand(
			["images/enes_all/caes/03_projection_plot_by_groups/_{weight_function}_{alpha_caes}_pos_{algorithm}_{discrete_feature}.png",
			"images/enes_all/ciuo/03_projection_plot_by_groups/_{weight_function}_{alpha_ciuo}_pos_{algorithm}_{discrete_feature}.png"],
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			algorithm=["leiden"], # Any algorithm will do since we're only plotting discrete feature groups
			discrete_feature=["grupo"],
		),
		expand(
			["images/enes_all/caes/03_projection_plot_gradient/_{weight_function}_{alpha_caes}_pos_{discrete_feature}.png",
			"images/enes_all/ciuo/03_projection_plot_gradient/_{weight_function}_{alpha_ciuo}_pos_{discrete_feature}.png"],
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			discrete_feature=CONTINUOUS_FEATURES,
		),
		expand(
			["images/enes_all/caes/03_communities/_distribution_{weight_function}_{alpha_caes}_{algorithm}.png",
			"images/enes_all/ciuo/03_communities/_distribution_{weight_function}_{alpha_ciuo}_{algorithm}.png"],
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			algorithm=ALGORITHMS,
		),
		expand(
			["images/enes_all/caes/05_edge_weight_correlation/_{weight_function}_{alpha_caes}_pos_{algorithm}_{continuous_feature}.png",
			"images/enes_all/ciuo/05_edge_weight_correlation/_{weight_function}_{alpha_ciuo}_pos_{algorithm}_{continuous_feature}.png"],
			weight_function=["hidalgo"],
			algorithm=ALGORITHMS,
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			continuous_feature=CONTINUOUS_FEATURES,
		),
		expand(
			"images/enes_all/{class_}/08_persistence_diagram/_{weight_function}_{topo_method}.png",
			weight_function=["hidalgo"],
			class_=CLASSES,
			topo_method=TOPO_METHOD,
		),
		expand(
			"data/processed/eph/{eph_file}.csv",
			eph_file=EPH_FILES
		),
		expand(
			"images/eph/{class_}/09_alpha_sensitivity/_{weight_function}.png",
			class_=["caes", "cno"],
			weight_function=["hidalgo"],
			algorithm=ALGORITHMS,
		),
		expand(
			"images/eph/{class_}/10_edge_weight_correlation/_{weight_function}_{feature}.png",
			class_=["caes", "cno"],
			weight_function=["hidalgo"],
			feature=["female_pct"],
		),


rule _00_aed_report:
	'''AED: Análisis Exploratorio de Datos on ENES datasets'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{dataset}/nodelist_caes.csv",
		"data/processed/{dataset}/nodelist_ciuo.csv"
	output:
		"images/{dataset}/00_aed_report/aed_top_sectors.png",
		"images/{dataset}/00_aed_report/aed_top_occupations.png",
		"images/{dataset}/00_aed_report/aed_distributions.png",
		"images/{dataset}/00_aed_report/aed_correlation_matrix.png"
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


rule _03_resolution_sensitivity:
	"""Sensitivity of community detection to resolution parameter alpha. This will compare all algorithms also."""
	input:
		"data/processed/{dataset}/{class_}/_compute_resolution_sensitivity/_df_{weight_function}_{alpha}.csv",
		"data/processed/{dataset}/{class_}/_compute_resolution_sensitivity/_df_scores_{weight_function}_{alpha}.csv"
	output:
		"images/{dataset}/{class_}/03_resolution_sensitivity/_catplots_{weight_function}_{alpha}.png",
		"images/{dataset}/{class_}/03_resolution_sensitivity/_catplots_{weight_function}_{alpha}_AMI.png",
		"images/{dataset}/{class_}/03_resolution_sensitivity/_catplots_{weight_function}_{alpha}_NMI.png"
	#log:
	#	"images/{dataset}/{class_}/03_resolution_sensitivity/_catplots_{weight_function}_{alpha}.log"
	script:
		"scripts/03_resolution_sensitivity.py"
	#shell:
	#	"export CUDA_PATH=/usr"


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
		"data/processed/{dataset}/{class_}/_alpha_sensitivity/_{weight_function}.json"
	output:
		"images/{dataset}/{class_}/07_alpha_sensitivity/_{weight_function}.png"
	log:
		"images/{dataset}/{class_}/07_alpha_sensitivity/_{weight_function}.log"
	script:
		"scripts/07_alpha_sensitivity.py"


rule _08_persistence_diagram:
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		"images/{dataset}/{class_}/08_persistence_diagram/_{weight_function}_{topo_method}.png"
	#log:
	#	"images/{dataset}/{class_}/08_persistence_diagram/_{weight_function}.log"
	script:
		"scripts/08_persistence_diagram.py"


rule _09_alpha_sensitivity_eph:
	"""Alpha-sensitivity sweep across all EPH projection graphs.
	Produces a single combined plot per (class_, weight_function), overlaying
	all EPH series with colormap gradients.
	"""
	resources:
		limited_slots = 7
	wildcard_constraints:
		class_ = "caes|cno"
	input:
		lambda wildcards: expand(
			"data/processed/eph/{eph_file}/{class_}/_alpha_sensitivity/_{weight_function}.json",
			dataset=EPH_FILES,
			eph_file=EPH_FILES,
			class_=wildcards.class_,
			weight_function=wildcards.weight_function,
		)
	output:
		"images/eph/{class_}/09_alpha_sensitivity/_{weight_function}.png"
	log:
		"images/eph/{class_}/09_alpha_sensitivity/_{weight_function}.log"
	script:
		"scripts/09_alpha_sensitivity_eph.py"


rule _10_edge_weight_correlation_eph:
	"""Time-series assortativity (Pearson r) across EPH waves.

	Produces a single combined plot per (class_, weight_function, feature), overlaying
	unfiltered projections plus disparity-filtered backbones for a small alpha grid.
	"""
	wildcard_constraints:
		class_ = "caes|cno"
	input:
		projections=lambda wildcards: expand(
			"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}.gexf",
			eph_file=EPH_FILES,
			class_=wildcards.class_,
			weight_function=wildcards.weight_function,
		),
		processed=lambda wildcards: expand(
			"data/processed/eph/{eph_file}.csv",
			eph_file=EPH_FILES,
		),
	output:
		"images/eph/{class_}/10_edge_weight_correlation/_{weight_function}_{feature}.png"
	log:
		"images/eph/{class_}/10_edge_weight_correlation/_{weight_function}_{feature}.log"
	script:
		"scripts/10_edge_weight_correlation_eph.py"


include: "rules/00_prepare.smk"
include: "rules/01_bipartite.smk"
include: "rules/02_projection.smk"
include: "rules/03_communities.smk"
