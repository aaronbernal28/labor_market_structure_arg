configfile: "config.yaml"

DATASETS = ["enes_all", "enes_2019", "enes_2021"]
METADATA = ["nodelist_caes", "nodelist_ciuo"]
WEIGHTS = ["vecinos_compartidos", "hub_depressed", "dot_product", "hidalgo"]
ALGORITHMS = ["louvain", "leiden", "infomap"]

rule all:
    input:
        expand(
            "images/02_projection_by_groups/02_projection_{dataset}_{layout}.png",
            dataset=DATASETS,
        )


include: "rules/00_prepare.smk"
include: "rules/01_bipartite.smk"
include: "rules/02_projection.smk"
include: "rules/03_communities.smk"
include: "rules/04_backbone.smk"