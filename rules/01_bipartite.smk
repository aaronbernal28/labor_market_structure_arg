rule bipartite_graph:
	'''Build bipartite graph from ENES datasets.'''
	input:
		"data/processed/{dataset}.csv",
		"data/processed/{dataset}/nodelist_caes.csv",
		"data/processed/{dataset}/nodelist_ciuo.csv"
	output:
		"data/graphs/{dataset}/bipartite.gexf"
	log:
		"images/{dataset}/bipartite_graph.log"
	script:
		"../scripts/utils/build_bipartite.py"
