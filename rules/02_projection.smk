rule compute_projection:
	input:
		edges="data/graphs/bipartite_{dataset}_{logscale}.gexf"
	output:
		graph="data/processed/projection_{dataset}_{logscale}_{weight}.gexf"
	script:
		"scripts/utils/build_projection.py"
