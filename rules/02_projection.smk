rule compute_projection:
	'''Build projection graph from bipartite graph based on caes or ciuo.'''
	input:
		"data/graphs/bipartite_{dataset}_{logscale}.gexf"
	output:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}.gexf"
	script:
		"../scripts/utils/build_projection.py"


rule filter_projection:
	'''Filter projection graph by degree.'''
	input:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}.gexf"
	output:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}.gexf"
	params:
		alpha = 1.000
	script:
		"../scripts/utils/filter_projection.py"
