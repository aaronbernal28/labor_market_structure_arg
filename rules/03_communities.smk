rule compute_communities:
	input:
		"data/graphs/projection_{dataset}_{logscale}_{class}_{weight}_{algorithm}_{alpha}.gexf"
	output:
		"data/graphs/projection_{dataset}_{logscale}_{class}_{weight}_{algorithm}_{alpha}_community.gexf"
	script:
		"scripts/utils/compute_communities.py"
