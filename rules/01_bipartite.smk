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

rule bipartite_graph_eph:
	'''Build bipartite graph from ENES datasets.'''
	input:
		"data/processed/eph/{eph_file}.csv"
	output:
		"data/graphs/eph/{eph_file}/bipartite_eph.gexf"
	log:
		"images/eph/{eph_file}/bipartite_graph.log"
	script:
		"../scripts/utils/build_bipartite_eph.py"
