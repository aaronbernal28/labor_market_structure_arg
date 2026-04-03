rule compute_projection:
	'''Build projection graph from bipartite graph based on caes or ciuo.'''
	input:
		"data/graphs/bipartite_{dataset}_{logscale}.gexf"
	wildcard_constraints:
		class_ = "nodelist_caes|nodelist_ciuo",
		weight_function = "dot_product|weighted_hidalgo_weight"
	output:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}.gexf"
	params:
		class_ = lambda wildcards: wildcards.class_,
		weight_function = lambda wildcards: wildcards.weight_function
	script:
		"../scripts/utils/build_projection.py"


rule filter_projection:
	'''Filter projection graph by degree.'''
	input:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}.gexf"
	output:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}.gexf"
	script:
		"../scripts/utils/filter_projection.py"
