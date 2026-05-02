rule compute_projection:
	'''Build projection graph from bipartite graph based on caes or ciuo.'''
	input:
		"data/graphs/{dataset}/bipartite.gexf"
	output:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	log:
		"images/{dataset}/{class_}/compute_projection_{weight_function}.log"
	wildcard_constraints:
		dataset = "|".join(["enes_2019", "enes_2021", "enes_all"])
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


rule _compute_alpha_sensitivity:
	"""Worker for alpha sensitivity: sweep a single projection graph and save metrics to JSON."""
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/processed/{dataset}/{class_}/_alpha_sensitivity/_{weight_function}_{algorithm}.json"
	log:
		"data/processed/{dataset}/{class_}/_alpha_sensitivity/_{weight_function}_{algorithm}.log"
	script:
		"../scripts/utils/compute_alpha_sensitivity.py"


rule _compute_alpha_sensitivity_eph:
	"""Worker for alpha sensitivity: sweep a single EPH projection graph and save metrics to JSON."""
	input:
		"data/graphs/eph/{eph_file}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/processed/eph/{eph_file}/{class_}/_alpha_sensitivity/_{weight_function}_{algorithm}.json"
	log:
		"data/processed/eph/{eph_file}/{class_}/_alpha_sensitivity/_{weight_function}_{algorithm}.log"
	script:
		"../scripts/utils/compute_alpha_sensitivity.py"
