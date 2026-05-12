from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from scripts import *
import src.logging_utils as log

snakemake: Any

HIGHLIGHT_CODES = [81201, 81132, 81202, 81331, 81332]


def main() -> None:
    plt.style.use("src/styles/publication.mplstyle")

    # Load nodelist for labels
    nodelist_df = pd.read_csv(snakemake.input.nodelist)
    cno_labels = dict(zip(nodelist_df["v183cno"], nodelist_df["cnolabel"]))

    # Prepare results list
    centrality_data = []

    projection_paths = [Path(p) for p in snakemake.input.projections]
    eph_files = [utils.extract_eph_file_from_path(p) for p in projection_paths]
    eph_files_sorted = utils.sort_eph_files(eph_files)
    eph_to_projection = {
        utils.extract_eph_file_from_path(p): p for p in projection_paths
    }

    # Iterate over projection files (sorted by EPH wave)
    for eph_file in eph_files_sorted:
        projection_path = eph_to_projection.get(eph_file)
        if projection_path is None:
            continue

        key = utils.parse_eph_file_key(eph_file)
        if key is None:
            time_date = pd.NaT
            time_label = eph_file
        else:
            time_date = pd.Timestamp(key.time_date)
            time_label = key.label

        # Load graph
        graph = nx.read_gexf(projection_path, node_type=int)

        # Calculate betweenness centrality
        bc = nx.betweenness_centrality(graph, weight="weight")

        for code in HIGHLIGHT_CODES:
            if code in bc:
                centrality_data.append(
                    {
                        "eph_file": eph_file,
                        "time_date": time_date,
                        "time_label": time_label,
                        "code": code,
                        "label": cno_labels.get(code, str(code)),
                        "betweenness_centrality": bc[code],
                    }
                )

    df_results = pd.DataFrame(centrality_data)

    if df_results.empty:
        print("No AI occupations found in any of the graphs.")
        # Create empty plot to satisfy snakemake
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No data available", ha="center")
        plt.savefig(snakemake.output[0])
        return

    df_results["time_date"] = pd.to_datetime(df_results["time_date"], errors="coerce")
    df_results = df_results.sort_values(["time_date", "label"], kind="mergesort")

    # Plot time-series with real datetime axis
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_results,
        x="time_date",
        y="betweenness_centrality",
        marker="o",
        hue="label",
    )
    plt.title(
        f"Betweenness Centrality for AI Occupations ({snakemake.wildcards.weight_function})"
    )
    plt.tight_layout()

    # Save plot
    output_path = Path(snakemake.output[0])
    plt.savefig(output_path)

    # Logging
    log_lines = []
    log_lines.append("=" * 60)
    log_lines.append("BETWEENNESS CENTRALITY AI OCCUPATIONS")
    log_lines.append("=" * 60)
    log.add_snakemake_overview(log_lines, snakemake)
    # log.add_dataframe_info(log_lines, "Centrality Results", df_results)
    log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
    log.write_log(log_lines, log_path)


if __name__ == "__main__":
    main()
