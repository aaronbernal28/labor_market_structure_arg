configfile: "config.yaml"
from numpy import logspace, geomspace


DATASETS = ["enes_all", "enes_2019", "enes_2021", "enes_all_male", "enes_all_female"]
NODELIST = ["caes", "ciuo"]
WEIGHT_FUNCTIONS = ["hidalgo", "unweighted_hidalgo", "dot_product", "cosine"]
ALGORITHMS_CAES = ["infomap"]
ALGORITHMS_CIUO = ["infomap"]
ALGORITHMS_ALL = ["louvain", "leiden", "infomap"]
VARIABLES = ["sex_id", "public_worker", "total_income", "education_mean"]
DISCRETE_FEATURES = ["grupo", "community"] # in nodelist data
CONTINUOUS_FEATURES = ["female_pct", "public_sector_pct", "income_median", "income_mean", "nivel_ed_mean", "age_mean",
	"income_std", "nivel_ed_std", "age_std"] # in nodelist data
LAYOUTS = ["spring_layout"]
CLASSES = ["caes", "ciuo"]
CLASSES_ALL = ["caes", "ciuo", "cno"]
ALPHA_CAES = ["0.05"]
ALPHA_CIUO = ["0.05"]
ALPHAS_ALL = ALPHA_CAES + ALPHA_CIUO
TOPO_METHOD = ["shortest_path", "disparity_filtration"]
EPH_YEARS = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
EPH_FILES = glob_wildcards("data/raw/eph/{eph_file}.csv").eph_file
DATASETS_ALL = DATASETS + EPH_FILES
NULL_GRAPH_MODELS = ["configuration_model", "enhanced_configuration_model"]
DISTANCE_DIAGRAMS = ["bottleneck", "wasserstein"]
RESOLUTIONS = geomspace(0.1, 30, num=40).round(4).tolist()

wildcard_constraints:
	dataset = "|".join(DATASETS_ALL),
	class_ = "|".join(CLASSES_ALL),
	weight_function = "|".join(WEIGHT_FUNCTIONS),
	algorithm = "|".join(ALGORITHMS_ALL),
	alpha = "|".join(ALPHAS_ALL),
	topo_method = "|".join(TOPO_METHOD),
	eph_file = "|".join(EPH_FILES),
	distance_diagrams = "|".join(DISTANCE_DIAGRAMS)


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
		),
		#expand(
		#	["images/enes_all/caes/03_resolution_sensitivity/_catplots_{weight_function}_{alpha_caes}.png",
		#	"images/enes_all/ciuo/03_resolution_sensitivity/_catplots_{weight_function}_{alpha_ciuo}.png"],
		#	weight_function=["hidalgo"],
		#	alpha_caes=ALPHA_CAES,
		#	alpha_ciuo=ALPHA_CIUO,
		#),
		expand(
			["images/enes_all/caes/03_projection_plot_by_groups/_{weight_function}_{alpha_caes}_pos_{algorithm_caes}_{discrete_feature}.png",
			"images/enes_all/ciuo/03_projection_plot_by_groups/_{weight_function}_{alpha_ciuo}_pos_{algorithm_ciuo}_{discrete_feature}.png"],
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			algorithm_caes=ALGORITHMS_CAES,
			algorithm_ciuo=ALGORITHMS_CIUO,
			discrete_feature=["community"],
		),
		expand(
			["images/enes_all/caes/03_projection_plot_by_groups/_{weight_function}_{alpha_caes}_pos_{algorithm_caes}_{discrete_feature}.png",
			"images/enes_all/ciuo/03_projection_plot_by_groups/_{weight_function}_{alpha_ciuo}_pos_{algorithm_ciuo}_{discrete_feature}.png"],
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			algorithm_caes=ALGORITHMS_CAES, # Any caes algorithm, since discrete feature is grupo which is independent of communities
			algorithm_ciuo=ALGORITHMS_CIUO,
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
			["images/enes_all/caes/03_communities/_distribution_{weight_function}_{alpha_caes}_{algorithm_caes}.png",
			"images/enes_all/ciuo/03_communities/_distribution_{weight_function}_{alpha_ciuo}_{algorithm_ciuo}.png"],
			weight_function=["hidalgo"],
			alpha_caes=ALPHA_CAES,
			alpha_ciuo=ALPHA_CIUO,
			algorithm_caes=ALGORITHMS_CAES,
			algorithm_ciuo=ALGORITHMS_CIUO,
		),
		expand(
			["images/enes_all/caes/05_edge_weight_correlation/_{weight_function}_{alpha_caes}_pos_{algorithm_caes}_{continuous_feature}.png",
			"images/enes_all/ciuo/05_edge_weight_correlation/_{weight_function}_{alpha_ciuo}_pos_{algorithm_ciuo}_{continuous_feature}.png"],
			weight_function=["hidalgo"],
			algorithm_caes=ALGORITHMS_CAES,
			algorithm_ciuo=ALGORITHMS_CIUO,
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
		),
		expand(
			"images/eph/{class_}/10_edge_weight_correlation/_{weight_function}_{feature}.png",
			class_=["caes", "cno"],
			weight_function=["hidalgo"],
			feature=["female_pct"],
		),
		expand(
			"images/eph/cno/12_betweenness_centrality_ai/betweenness_centrality_ai_{weight_function}_backbone.png",
			weight_function=["hidalgo"],
		),
		expand(
			"images/eph/{class_}/13_preferential_attachment/_{weight_function}.png",
			weight_function=["hidalgo"],
			class_=["cno"],
		),
		"images/enes_all/ciuo/14_persistence_diagram_distance_hypothesis_test/_hidalgo_disparity_filtration.log",
		expand(
			"images/enes_all/{class_}/08_persistence_diagram/_{weight_function}_{topo_method}_gender.png",
			weight_function=["hidalgo"],
			class_=CLASSES,
			topo_method=["disparity_filtration"],
		),
		expand(
			"images/eph/{class_}/16_persistence_diagram_distance/_{weight_function}_{topo_method}_heatmap_{distance_diagrams}.png",
			class_=["caes", "cno"],
			weight_function=["hidalgo"],
			topo_method=TOPO_METHOD,
			distance_diagrams=DISTANCE_DIAGRAMS,
		),
		expand(
			"images/enes_all/ciuo/18_disparity_filtration_subgraph/_hidalgo_0.05_pos_{algorithm}_filtration.png",
			class_=["caes", "ciuo"],
			algorithm=ALGORITHMS_CIUO,
		),
		#expand(
		#	"images/eph/{eph_file}/{class_}/14_persistence_diagram_distance_hypothesis_test/_{weight_function}_{topo_method}.log",
		#	eph_file=EPH_FILES,
		#	class_=["cno"],
		#	weight_function=["hidalgo"],
		#	topo_method=["disparity_filtration"],
		#),
		"images/20_persistence_diagram_umap_all/_hidalgo_disparity_filtration_wasserstein_umap_H1.png",


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
		lambda wc: expand(
			["data/processed/{dataset}/{class_}/_compute_resolution_sensitivity/_df_{weight_function}_{alpha}_{resolution}.csv",
			"data/processed/{dataset}/{class_}/_compute_resolution_sensitivity/_df_scores_{weight_function}_{alpha}_{resolution}.csv"],
			resolution=RESOLUTIONS,
			dataset=wc.dataset,
			class_=wc.class_,
			weight_function=wc.weight_function,
			alpha=wc.alpha,
		)
	output:
		"images/{dataset}/{class_}/03_resolution_sensitivity/_catplots_{weight_function}_{alpha}.png",
		"images/{dataset}/{class_}/03_resolution_sensitivity/_catplots_{weight_function}_{alpha}_AMI.png",
		"images/{dataset}/{class_}/03_resolution_sensitivity/_catplots_{weight_function}_{alpha}_NMI.png",
		"images/{dataset}/{class_}/03_resolution_sensitivity/_{weight_function}_{alpha}_modularity.png"
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


rule _15_persistence_diagram_gender:
	"""Compare male vs female persistence diagrams side-by-side."""
	input:
		male="data/diagrams/enes_all_male/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv",
		female="data/diagrams/enes_all_female/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv"
	output:
		"images/enes_all/{class_}/08_persistence_diagram/_{weight_function}_{topo_method}_gender.png"
	script:
		"scripts/15_persistence_diagram_gender.py"


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


rule _12_betweenness_centrality_ai:
	"""Time-series of betweenness centrality for AI-related occupations in CNO projections."""
	input:
		projections=lambda wildcards: expand(
			"data/graphs/eph/{eph_file}/cno/projection_{weight_function}_backbone.gexf",
			eph_file=EPH_FILES,
			weight_function=wildcards.weight_function,
		),
		nodelist="data/raw/nodelist_cno.csv",
	output:
		"images/eph/cno/12_betweenness_centrality_ai/betweenness_centrality_ai_{weight_function}_backbone.png"
	log:
		"images/eph/cno/12_betweenness_centrality_ai/betweenness_centrality_ai_{weight_function}_backbone.log"
	script:
		"scripts/12_betweenness_centrality_ai.py"


rule _13_preferential_attachment:
	"""Estimate the preferential attachment exponent alpha for EPH projections."""
	input:
		projections=lambda wildcards: sorted(expand(
			"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}_backbone.gexf",
			eph_file=EPH_FILES,
			class_=wildcards.class_,
			weight_function=wildcards.weight_function,
		))
	output:
		plot="images/eph/{class_}/13_preferential_attachment/_{weight_function}.png"
	log:
		"images/eph/{class_}/13_preferential_attachment/_{weight_function}.log"
	script:
		"scripts/13_preferential_attachment.py"


rule _14_persistence_diagram_distance_hypothesis_test:
	input:
		"data/diagrams/{dataset}/{class_}/_persistence_diagram_distance/_{weight_function}_{topo_method}.csv"
	output:
		"images/{dataset}/{class_}/14_persistence_diagram_distance_hypothesis_test/_{weight_function}_{topo_method}.log"
	params:
		alpha=0.05,
		n_perm=10000,
		two_sided=True,
		seed=42,
		null_families=NULL_GRAPH_MODELS,
	script:
		"scripts/14_persistence_diagram_distance_hypothesis_test.py"


rule _14_persistence_diagram_distance_hypothesis_test_eph:
	input:
		"data/diagrams/eph/{eph_file}/{class_}/_persistence_diagram_distance/_{weight_function}_{topo_method}.csv"
	output:
		"images/eph/{eph_file}/{class_}/14_persistence_diagram_distance_hypothesis_test/_{weight_function}_{topo_method}.log"
	params:
		alpha=0.05,
		n_perm=10000,
		two_sided=True,
		seed=42,
		null_families=NULL_GRAPH_MODELS,
	script:
		"scripts/14_persistence_diagram_distance_hypothesis_test.py"


rule _16_persistence_diagram_distance_eph:
	"""Compute distance between persistence diagrams between EPH waves.
	"""
	input:
		lambda wildcards: expand(
			"data/diagrams/eph/{eph_file}/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv",
			eph_file=EPH_FILES,
			class_=wildcards.class_,
			weight_function=wildcards.weight_function,
			topo_method=wildcards.topo_method,
		)
	output:
		"images/eph/{class_}/16_persistence_diagram_distance/_{weight_function}_{topo_method}_heatmap_{distance_diagrams}.png",
	script:
		"scripts/16_persistence_diagram_distance_eph.py"


rule _17_phi_proximity:
	"""Phi proximity plots from bipartite graphs."""
	input:
		"data/graphs/{dataset}/bipartite.gexf"
	output:
		"images/{dataset}/{class_}/17_phi_proximity/_phi_weighted_vs_unweighted.png",
		"images/{dataset}/{class_}/17_phi_proximity/_puv_vs_maxp_weighted_unweighted.png"
	log:
		"images/{dataset}/{class_}/17_phi_proximity/_phi_proximity.log"
	script:
		"scripts/17_phi_proximity.py"


rule _18_disparity_filtration_subgraph:
	"""Extract disparity-filtration subgraph for a given graph.
	Steps:
	1. Load the projection graph and corresponding nodelist.
	2. Filter the graph using one community (eg. C12) as the focal class.
	3. For alpha in a small grid (eg. 0.0001, 0.001, 0.01, None):
		a. Apply the disparity filtration to the subgraph.
	4. Plot the resulting subgraphs side-by-side where each is
		a. Colored by original community
		b. Constant node size
		c. Edge widths proportional to original weights (not disparity-filtered weights)
		b. Layout is fixed across all subgraphs for comparability (eg. original graph position)
		d. Labeled with code (as_display) nodes, only the new nodes respect the last plot.
			(eg. Plot 1: alpha = 0.0001, nodes={1, 2}, labels={1, 2},
			 Plot 2: alpha = 0.001, nodes={1, 2, 3}, labels={3},
			 Plot 3: alpha = None, nodes={1, 2, 3, 4, 5}, labels={4, 5})
	"""
	input:
		nodelist="data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos_{algorithm}.csv",
		graph="data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		"images/{dataset}/{class_}/18_disparity_filtration_subgraph/_{weight_function}_{alpha}_pos_{algorithm}_filtration.png"
	script:
		"scripts/18_disparity_filtration_subgraph.py"


rule _19_public_policy_by_communities:
	"""Initial exploration of public policy variables by communities.
	Steps:
	1. Load the projection graph and corresponding nodelist with community labels.
	2. Filter the graph to only include nodes from two communities (eg. C12 vs C13).
	3. Compute betweenness centrality for all nodes in the filtered graph.
	4. Log the top 5 nodes by betweenness centrality Label (key).
	5. Plot the betweenness centrality distribution as a histogram, with separate colors for each community. Highlight the top 5 nodes with labels on the plot.
	"""
	wildcard_constraints:
		c1 = "C(?:0[1-9]|[12][0-9]|3[0-9])",
		c2 = "C(?:0[1-9]|[12][0-9]|3[0-9])",
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf",
		"data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos_{algorithm}.csv",
	output:
		"images/{dataset}/{class_}/19_public_policy_by_communities/_{weight_function}_{alpha}_{algorithm}_{c1}_{c2}_betweenness_centrality.png"
	log:
		"images/{dataset}/{class_}/19_public_policy_by_communities/_{weight_function}_{alpha}_{algorithm}_{c1}_{c2}_top_betweenness_centrality.log"
	script:
		"scripts/19_public_policy_by_communities.py"


rule _20_persistence_diagram_umap_all:
	"""UMAP projection of persistence diagram distances across all datasets and EPH waves.
	Steps:
	1. Load all persistence diagram distance matrices for all datasets and EPH waves.
	2. For each distance matrix, extract the distance values for a specific weight function and topological method (eg. Hidalgo + disparity filtration).
	3. Combine all distance values into a single dataframe, with columns for dataset/eph_file, class, weight function, topological method, and distance values.
	4. Apply UMAP to the combined distance values to reduce them to 2 dimensions.
	5. Plot the UMAP projection, coloring points by dataset/eph_file and shaping points by class. Optionally, annotate points with weight function and topological method.

	Notes:
	1. The real model must have high alpha and more vibrant colors.
	2. Each null model should have a consistent color with its linked real model, but with lower alpha for better visualization of the real vs null contrast.
	3. EPH palette: pal_observable("observable10", alpha=0.7)(10) from ggsci.observable import pal_observable
	"""
	input:
		lambda wc:
			expand(
				"data/diagrams/{dataset}/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv",
				dataset=["enes_all", "enes_2019", "enes_2021"],
				class_=["ciuo"],
				weight_function=wc.weight_function,
				topo_method=wc.topo_method,
				allow_missing=True,
			)
			+
			expand(
				"data/diagrams/{dataset}/{class_}/_persistence_diagram/{null_model}/_{weight_function}_{i}_{topo_method}.csv",
				null_model=NULL_GRAPH_MODELS,
				dataset=["enes_all", "enes_2019", "enes_2021"],
				class_=["ciuo"],
				weight_function=wc.weight_function,
				topo_method=wc.topo_method,
				i=range(10),
				allow_missing=True,
			)
			+
			expand(
				"data/diagrams/eph/{eph_file}/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv",
				eph_file=EPH_FILES,
				class_=["cno"],
				weight_function=wc.weight_function,
				topo_method=wc.topo_method,
			)
			+
			expand(
				"data/diagrams/eph/{eph_file}/{class_}/_persistence_diagram/{null_model}/_{weight_function}_{i}_{topo_method}.csv",
				null_model=NULL_GRAPH_MODELS,
				eph_file=EPH_FILES,
				class_=["cno"],
				weight_function=wc.weight_function,
				topo_method=wc.topo_method,
				i=range(10),
			)
	output:
		"images/20_persistence_diagram_umap_all/_{weight_function}_{topo_method}_wasserstein_umap_H0.png",
		"images/20_persistence_diagram_umap_all/_{weight_function}_{topo_method}_wasserstein_umap_H1.png",
		"images/20_persistence_diagram_umap_all/_{weight_function}_{topo_method}_wasserstein_umap_H2.png"
	script:
		"scripts/20_persistence_diagram_umap_all.py"


include: "rules/00_prepare.smk"
include: "rules/01_bipartite.smk"
include: "rules/02_projection.smk"
include: "rules/03_communities.smk"
include: "rules/04_diagrams.smk"
