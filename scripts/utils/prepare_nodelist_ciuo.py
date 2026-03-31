import pandas as pd


def main() -> None:
    _ = pd.read_csv(snakemake.input[0])
    pd.DataFrame().to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
    main()
