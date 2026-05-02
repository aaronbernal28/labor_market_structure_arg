rule prepare_data_enes:
	'''
	Prepare the data for the ENES {year} dataset. This includes:
	- Filtering the data to include only the relevant columns and rows
	- Cleaning the data
	'''
	wildcard_constraints:
		dataset = "enes_2019|enes_2021"
	input:
		lambda wildcards: "data/raw/base_enespersonas_2021.csv" if wildcards.dataset == "enes_2021" else "data/raw/base_enespersonas.csv"
	output:
		"data/processed/{dataset}.csv"
	log:
		"data/processed/{dataset}/prepare_data_enes.log"
	script:
		"../scripts/utils/prepare_data_enes.py"


rule prepare_enes_all:
	'''Merge the ENES 2019 and 2021 datasets into a single dataset.'''
	input:
		"data/processed/enes_2019.csv",
		"data/processed/enes_2021.csv"
	output:
		"data/processed/enes_all.csv"
	log:
		"data/processed/enes_all/prepare_enes_all.log"
	script:
		"../scripts/utils/prepare_data_enes_all.py"


rule prepare_nodelist:
	'''Add coloring and features information to the nodelist infered from the one ENES dataset.'''
	input:
		"data/processed/{dataset}.csv"
	output:
		"data/processed/{dataset}/nodelist_{class_}.csv"
	log:
		"data/processed/{dataset}/nodelist_{class_}.log"
	script:
		"../scripts/utils/prepare_nodelist_caes.py"


rule import_eph_data:
	'''Scrape INDEC bases page and download EPH resources (side effects), writing a single audit log.'''
	resources:
		limited_slots = 7
	wildcard_constraints:
		year = "|".join(EPH_YEARS)
	output:
		"data/raw/eph/import_eph_data_{year}.log"
	script:
		"../scripts/utils/import_eph_data.py"


rule import_eph_data_all:
	'''Download EPH data for all configured years.'''
	input:
		expand("data/raw/eph/import_eph_data_{year}.log", year=EPH_YEARS)


rule prepare_data_eph:
	'''
	Prepare one raw EPH CSV into a processed per-file CSV.
	'''
	input:
		"data/raw/eph/{eph_file}.csv"
	output:
		"data/processed/eph/{eph_file}.csv"
	script:
		"../scripts/utils/prepare_data_eph.py"


rule prepare_eph_all:
	input:
		expand("data/processed/eph/{eph_file}.csv", eph_file=EPH_FILES)
