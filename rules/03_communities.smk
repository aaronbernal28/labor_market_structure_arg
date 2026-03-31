rule compute_communities:
	input:
		"data/graphs/projection_{dataset}_{logscale}_{weight}.gexf"
	output:
		"data/graphs/projection_{dataset}_{logscale}_{weight}_{algorithm}.gexf"
	script:
		"scripts/utils/compute_communities.py"
