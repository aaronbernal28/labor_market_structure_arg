rule bipartite_graph:
	'''Build bipartite graph from ENES datasets.'''
	input:
		"data/processed/{dataset}.csv"
	output:
		"data/graphs/{dataset}/bipartite.gexf"
	log:
		"images/{dataset}/bipartite_graph.log"
	script:
		"../scripts/utils/build_bipartite.py"
