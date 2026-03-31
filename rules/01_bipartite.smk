rule bipartite_graph:
	'''Build bipartite graph from ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{METADATA[0]}_{dataset}.csv",
		"data/processed/{METADATA[1]}_{dataset}.csv"
	output:
		"data/graphs/bipartite_{dataset}_{logscale}.gexf"
	script:
		"scripts/utils/build_bipartite.py"
