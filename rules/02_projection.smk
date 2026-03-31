rule compute_projection:
	'''Build projection graph from bipartite graph based on caes or ciuo.'''
	input:
		"data/graphs/bipartite_{dataset}_{logscale}.gexf"
	output:
		"data/graphs/projection_{dataset}_{class}_{logscale}_{weight}_{algorithm}_1.0000.gexf"
	script:
		"scripts/utils/build_projection.py"


rule filter_projection:
	'''Filter projection graph by degree.'''
	input:
		"data/graphs/projection_{dataset}_{class}_{logscale}_{weight}_{algorithm}_1.0000.gexf"
	output:
		"data/graphs/projection_{dataset}_{class}_{logscale}_{weight}_{algorithm}_{alpha}.gexf"
		"images/filter_projection/filter_projection_alpha_sensitivity_{dataset}_{class}_{logscale}_{weight}_{algorithm}.png"
	script:
		"scripts/utils/filter_projection.py"
