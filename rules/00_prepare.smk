rule prepare_data_enes_2019:
	# TODO: refactor to use the same script for both datasets, with parameters
	'''
	Prepare the data for the ENES 2019 dataset. This includes:
	- Filtering the data to include only the relevant columns and rows
	- Cleaning the data
	'''
	output:
		"data/processed/enes_2019.csv"
	script:
		"scripts/utils/prepare_data_enes_2019.py"


rule prepare_data_enes_2021:
	output:
		"data/processed/enes_2021.csv"
	script:
		"scripts/utils/prepare_data_enes_2021.py"


rule prepare_enes_all:
	'''Merge the ENES 2019 and 2021 datasets into a single dataset.'''
	input:
		"data/processed/enes_2019.csv",
		"data/processed/enes_2021.csv"
	output:
		"data/processed/enes_all.csv"
	script:
		"scripts/utils/prepare_data_enes_all.py"


rule prepare_nodelist_caes:
	'''Add coloring information to the CAES nodelist.'''
	input:
		"data/processed/{dataset}.csv"
	output:
		"data/processed/{METADATA[0]}_{dataset}.csv"
	script:
		"scripts/utils/prepare_nodelist_caes.py"


rule prepare_nodelist_ciuo:
	'''Add coloring information to the CIUO nodelist.'''
	input:
		"data/processed/{dataset}.csv"
	output:
		"data/processed/{METADATA[1]}_{dataset}.csv"
	script:
		"scripts/utils/prepare_nodelist_ciuo.py"

