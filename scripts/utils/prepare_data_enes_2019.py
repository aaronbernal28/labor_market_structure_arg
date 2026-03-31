from scripts import *
import pandas as pd


def main() -> None:
	df = pd.DataFrame()
	df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
