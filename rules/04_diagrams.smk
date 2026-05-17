rule _compute_persistence_diagram_distance:
	"""Compute distance between persistence diagrams for real and null graphs on json."""
	input:
		"data/diagrams/{dataset}/{class_}/_persistence_diagram/_{weight_function}_{topo_method}.csv",
		expand(
			"data/diagrams/{dataset}/{class_}/_persistence_diagram/{null_model}/_{weight_function}_{i}_{topo_method}.csv",
			null_model=NULL_GRAPH_MODELS,
			dataset="{dataset}",
			class_="{class_}",
			weight_function="{weight_function}",
			topo_method="{topo_method}",
			i=range(20),
		),
	output:
		"data/diagrams/{dataset}/{class_}/_persistence_diagram_distance/_{weight_function}_{topo_method}.csv"
	script:
		"../scripts/utils/compute_persistence_diagram_distance.py"
