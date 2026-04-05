rule compute_positions:
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf",
		"data/processed/{dataset}/nodelist_{class_}.csv"
	output:
		"data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos.csv"
	script:
		"../scripts/utils/compute_positions.py"


rule compute_communities:
	input:
		"data/graphs/{dataset}/{class_}/projection_{weight_function}_{alpha}.gexf",
		"data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos.csv"
	output:
		"data/processed/{dataset}/nodelist_{class_}_{weight_function}_{alpha}_pos_{algorithm}.csv",
		"images/{dataset}/{class_}/03_communities/_distribution_{weight_function}_{alpha}_{algorithm}.png"
	script:
		"../scripts/utils/compute_communities.py"
