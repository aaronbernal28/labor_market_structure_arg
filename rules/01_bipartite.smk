rule bipartite_graph:
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{METADATA[0]}.csv",
		"data/processed/{METADATA[1]}.csv"
	output:
		"data/graphs/{dataset}_bipartite.gexf"
	script:
		"scripts/utils/build_bipartite.py"