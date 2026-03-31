rule bipartite_graph:
	'''Build bipartite graph from ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{METADATA[0]}.csv",
		"data/processed/{METADATA[1]}.csv"
	output:
		"data/graphs/bipartite_{dataset}.gexf"
	script:
		"scripts/utils/build_bipartite.py"
