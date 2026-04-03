rule compute_positions:
	input:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}.gexf",
		"data/processed/nodelist_{class_}_{dataset}.csv"
	output:
		"data/processed/nodelist_{class_}_{dataset}_{logscale}_{weight_function}_{alpha}_pos.csv"
	script:
		"../scripts/utils/compute_positions.py"


rule compute_communities:
	input:
		"data/graphs/projection_{dataset}_{logscale}_{class_}_{weight_function}_{alpha}.gexf",
		"data/processed/nodelist_{class_}_{dataset}_{logscale}_{weight_function}_{alpha}_pos.csv"
	params:
		algorithm = "louvain",
	output:
		"data/processed/nodelist_{class_}_{dataset}_{logscale}_{weight_function}_{alpha}_pos_{algorithm}.csv"
	script:
		"../scripts/utils/compute_communities.py"
