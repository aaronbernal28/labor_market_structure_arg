from snakemake.script import snakemake
import pandas as pd


def main() -> None:
	_ = pd.read_csv(snakemake.input[0])
	_ = pd.read_csv(snakemake.input[1])

	pd.DataFrame().to_csv(snakemake.output[0], index=False)
	pd.DataFrame().to_csv(snakemake.output[1], index=False)
	pd.DataFrame().to_csv(snakemake.output[2], index=False)


if __name__ == "__main__":
	main()
