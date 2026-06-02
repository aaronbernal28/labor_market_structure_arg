rule compute_projection:
	'''Build projection graph from bipartite graph based on caes or ciuo.'''
	input:
		"data/graphs/{dataset}/bipartite.gexf"
	output:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	log:
		"images/{dataset}/{class_}/compute_projection_{weight_function}.log"
	wildcard_constraints:
		dataset = "|".join(["enes_2019", "enes_2021", "enes_all", "enes_all_male", "enes_all_female"])
	script:
		"../scripts/utils/build_projection.py"


rule compute_projection_eph:
	'''Build projection graph from EPH bipartite graph based on caes or cno.'''
	input:
		"data/graphs/eph/{eph_file}/bipartite_eph.gexf"
	output:
		"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}.gexf"
	log:
		"images/eph/{eph_file}/{class_}/compute_projection_{weight_function}.log"
	wildcard_constraints:
		class_ = "caes|cno"
	script:
		"../scripts/utils/build_projection.py"


rule filter_projection:
	'''Filter projection graph by degree.'''
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf",
		"images/{dataset}/{class_}/backbone_weight_histogram_{weight_function}_{alpha}.png"
	log:
		"images/{dataset}/{class_}/backbone_weight_histogram_{weight_function}_{alpha}.log"
	script:
		"../scripts/utils/filter_projection.py"


rule filter_projection_eph:
	'''Filter projection graph by degree.'''
	input:
		"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}_backbone.gexf",
		"images/eph/{eph_file}/{class_}/backbone_weight_histogram_{weight_function}.png"
	log:
		"images/eph/{eph_file}/{class_}/backbone_weight_histogram_{weight_function}.log"
	script:
		"../scripts/utils/filter_projection.py"


rule _compute_alpha_sensitivity:
	"""Worker for alpha sensitivity: sweep a single projection graph and save metrics to JSON."""
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/processed/{dataset}/{class_}/_alpha_sensitivity/_{weight_function}.json"
	log:
		"data/processed/{dataset}/{class_}/_alpha_sensitivity/_{weight_function}.log"
	script:
		"../scripts/utils/compute_alpha_sensitivity.py"


rule _compute_alpha_sensitivity_eph:
	"""Worker for alpha sensitivity: sweep a single EPH projection graph and save metrics to JSON."""
	input:
		"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}.gexf" # NOTE: unfiltered projection graph
	output:
		"data/processed/eph/{eph_file}/{class_}/_alpha_sensitivity/_{weight_function}.json"
	log:
		"data/processed/eph/{eph_file}/{class_}/_alpha_sensitivity/_{weight_function}.log"
	script:
		"../scripts/utils/compute_alpha_sensitivity.py"


rule _compute_resolution_sensitivity:
	"""Sensitivity of community detection to resolution parameter alpha. This will compare all algorithms also."""
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf",
	params:
		algorithms = ALGORITHMS_ALL,
	output:
		"data/processed/{dataset}/{class_}/_compute_resolution_sensitivity/_df_{weight_function}_{alpha}.csv",
		"data/processed/{dataset}/{class_}/_compute_resolution_sensitivity/_df_scores_{weight_function}_{alpha}.csv"
	script:
		"../scripts/utils/compute_resolution_sensitivity.py"
	#shell:
	#	"export CUDA_PATH=/usr"


rule _compute_persistence_diagram:
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/diagrams/{dataset}/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv"
	script:
		"../scripts/utils/compute_persistence_diagram.py"


rule _compute_random_graphs:
	'''Compute random graphs for a given projection graph.'''
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		cm ="data/graphs/{dataset}/{class_}/_random_graphs/configuration_model/_{weight_function}_{i}.gexf",
		ecm ="data/graphs/{dataset}/{class_}/_random_graphs/enhanced_configuration_model/_{weight_function}_{i}.gexf"
	log:
		"data/null_graphs/{dataset}/{class_}/_{weight_function}_random_graphs_{i}.log"
	wildcard_constraints:
		i = "\\d+"
	script:
		"../scripts/utils/compute_random_graphs.py"


rule _compute_random_graphs_eph:
	'''Compute random graphs for a given projection graph.'''
	input:
		"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}.gexf"
	output:
		cm ="data/graphs/eph/{eph_file}/{class_}/_random_graphs/configuration_model/_{weight_function}_{i}.gexf",
		ecm ="data/graphs/eph/{eph_file}/{class_}/_random_graphs/enhanced_configuration_model/_{weight_function}_{i}.gexf"
	log:
		"data/null_graphs/eph/{eph_file}/{class_}/_{weight_function}_random_graphs_{i}.log"
	wildcard_constraints:
		i = "\\d+"
	script:
		"../scripts/utils/compute_random_graphs.py"


rule compute_persistence_diagram_null_graphs:
	input:
		"data/graphs/{dataset}/{class_}/_random_graphs/{null_model}/_{weight_function}_{i}.gexf",
	output:
		"data/diagrams/{dataset}/{class_}/_persistence_diagram/{null_model}/_{weight_function}_{i}_{topo_method}.csv"
	wildcard_constraints:
		i = "\\d+"
	script:
		"../scripts/utils/compute_persistence_diagram.py"


rule _compute_persistence_diagram_eph:
	'''Compute persistence diagram for EPH projection graph.
	E.g. data/diagrams/eph/usu_individual_T318./caes/_persistence_diagram/_hidalgo_disparity_filtration.csv'''
	input:
		"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/diagrams/eph/{eph_file}/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv"
	script:
		"../scripts/utils/compute_persistence_diagram.py"
