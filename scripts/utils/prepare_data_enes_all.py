from scripts import *
import pandas as pd


def main() -> None:
	df_2019 = pd.read_csv(snakemake.input[0])
	df_2021 = pd.read_csv(snakemake.input[1])
	df = pd.concat([df_2019, df_2021], ignore_index=True)
	df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
