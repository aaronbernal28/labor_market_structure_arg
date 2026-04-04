rule compute_projection:
	'''Build projection graph from bipartite graph based on caes or ciuo.'''
	input:
		"data/graphs/{dataset}/{logscale}/bipartite.gexf"
	output:
		"data/graphs/{dataset}/{logscale}/{class_}/projection_{weight_function}.gexf"
	script:
		"../scripts/utils/build_projection.py"


rule filter_projection:
	'''Filter projection graph by degree.'''
	input:
		"data/graphs/{dataset}/{logscale}/{class_}/projection_{weight_function}.gexf"
	output:
		"data/graphs/{dataset}/{logscale}/{class_}/projection_{weight_function}_{alpha}.gexf",
		"images/{dataset}/{logscale}/{class_}/backbone_weight_histogram_{weight_function}_{alpha}.png"
	script:
		"../scripts/utils/filter_projection.py"
