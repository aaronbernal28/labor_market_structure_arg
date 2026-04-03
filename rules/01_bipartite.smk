rule bipartite_graph:
	'''Build bipartite graph from ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/nodelist_caes_{dataset}.csv",
		"data/processed/nodelist_ciuo_{dataset}.csv"
	output:
		"data/graphs/bipartite_{dataset}_{logscale}.gexf"
	script:
		"../scripts/utils/build_bipartite.py"
