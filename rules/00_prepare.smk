rule prepare_data_enes:
	'''
	Prepare the data for the ENES {year} dataset. This includes:
	- Filtering the data to include only the relevant columns and rows
	- Cleaning the data
	'''
	wildcard_constraints:
		dataset = "enes_2019|enes_2021"
	output:
		"data/processed/{dataset}.csv"
	params:
		dataset = lambda wildcards: wildcards.dataset
	script:
		"../scripts/utils/prepare_data_enes.py"


rule prepare_enes_all:
	'''Merge the ENES 2019 and 2021 datasets into a single dataset.'''
	input:
		"data/processed/enes_2019.csv",
		"data/processed/enes_2021.csv"
	output:
		"data/processed/enes_all.csv"
	script:
		"../scripts/utils/prepare_data_enes_all.py"


rule prepare_nodelist:
	'''Add coloring and features information to the nodelist infered from the one ENES dataset.'''
	wildcard_constraints:
		nodelist = "nodelist_caes|nodelist_ciuo",
		dataset = "enes_2019|enes_2021|enes_all"
	input:
		"data/processed/{dataset}.csv"
	output:
		"data/processed/{nodelist}_{dataset}.csv"
	params:
		nodelist = lambda wildcards: wildcards.nodelist
	script:
		"../scripts/utils/prepare_nodelist_caes.py"
	