rule compute_positions:
	input:
		"data/graphs/{dataset}/{logscale}/{class_}/projection_{weight_function}_{alpha}.gexf",
		"data/processed/{dataset}/nodelist_{class_}.csv"
	output:
		"data/processed/{dataset}/{logscale}/nodelist_{class_}_{weight_function}_{alpha}_pos.csv"
	script:
		"../scripts/utils/compute_positions.py"


rule compute_communities:
	input:
		"data/graphs/{dataset}/{logscale}/{class_}/projection_{weight_function}_{alpha}.gexf",
		"data/processed/{dataset}/{logscale}/nodelist_{class_}_{weight_function}_{alpha}_pos.csv"
	output:
		"data/processed/{dataset}/{logscale}/nodelist_{class_}_{weight_function}_{alpha}_pos_{algorithm}.csv",
		"images/{dataset}/{logscale}/{class_}/03_communities/community_distribution__{weight_function}_{alpha}_{algorithm}.png"
	script:
		"../scripts/utils/compute_communities.py"
