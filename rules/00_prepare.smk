rule prepare_data_enes:
	'''
	Prepare the data for the ENES {year} dataset. This includes:
	- Filtering the data to include only the relevant columns and rows
	- Cleaning the data
	'''
	output:
		"data/processed/{dataset}.csv"
	script:
		"scripts/utils/prepare_data_enes.py"
	params:
		dataset = lambda wildcards: wildcards.dataset


rule prepare_enes_all:
	'''Merge the ENES 2019 and 2021 datasets into a single dataset.'''
	input:
		"data/processed/enes_2019.csv",
		"data/processed/enes_2021.csv"
	output:
		"data/processed/enes_all.csv"
	script:
		"scripts/utils/prepare_data_enes_all.py"


rule prepare_nodelist:
	'''Add coloring information to the CAES nodelist.'''
	input:
		"data/processed/{dataset}.csv"
	output:
		"data/processed/{nodelist}_{dataset}.csv"
	script:
		"scripts/utils/prepare_nodelist.py"
	params:
		nodelist = lambda wildcards: wildcards.nodelist
	