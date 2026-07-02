from typing import Any
from pathlib import Path
from scripts import *
import matplotlib.pyplot as plt

import src.topology as topo

snakemake: Any


def _load_diagrams(path: str) -> list:
    loader = topo.DGMS_loader(path)
    diagrams = loader.import_()
    return diagrams[:3] if len(diagrams) > 3 else diagrams


def main() -> None:
    plt.style.use("src/styles/publication.mplstyle")

    white_collar_diagrams = _load_diagrams(snakemake.input["white_collar"])
    blue_collar_diagrams = _load_diagrams(snakemake.input["blue_collar"])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("")

    pl.plot_persistence_diagrams(
        white_collar_diagrams,
        title="Red de trabajadores no manuales",
        ax=axs[0],
        save=False,
    )
    pl.plot_persistence_diagrams(
        blue_collar_diagrams,
        title="Red de trabajadores manuales",
        ax=axs[1],
        save=False,
    )

    plt.tight_layout()
    fig.savefig(Path(snakemake.output[0]), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
