rule bipartite_graph:
	'''Build bipartite graph from ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{dataset}/nodelist_caes.csv",
		"data/processed/{dataset}/nodelist_ciuo.csv"
	output:
		"data/graphs/{dataset}/{logscale}/bipartite.gexf"
	script:
		"../scripts/utils/build_bipartite.py"
