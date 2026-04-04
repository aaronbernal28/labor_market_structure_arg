from scripts import *
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import re

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	caes_id = snakemake.config["caes"]["id"]
	ciuo_id = snakemake.config["ciuo"]["id"]
	letra_caes = snakemake.config["caes"]["grupo"]
	letra_ciuo = snakemake.config["ciuo"]["letra"]

	enes_df = pd.read_csv(
		snakemake.input[0],
		dtype={snakemake.config["id_caes"]: int, snakemake.config["id_ciuo"]: int},
	)
	caes_df = pd.read_csv(snakemake.input[1], dtype={caes_id: int})
	ciuo_df = pd.read_csv(snakemake.input[2], dtype={ciuo_id: int})

	enes_df = pd.merge(
		enes_df,
		caes_df[[caes_id, letra_caes]],
		left_on=snakemake.config["id_caes"],
		right_on=caes_id,
		how="left",
	)
	enes_df = pd.merge(
		enes_df,
		ciuo_df[[ciuo_id, letra_ciuo]],
		left_on=snakemake.config["id_ciuo"],
		right_on=ciuo_id,
		how="left",
	)

	biadj = gc.build_biadjacency(
		enes_df,
		caes_col=letra_caes,
		ciuo_col=letra_ciuo,
		logscale=False,
	)
	if biadj.empty:
		raise ValueError("No valid CIUO→CAES pairs found after filtering.")

	flows = biadj.stack().reset_index(name="count")
	flows = flows[flows["count"] > 0]

	def _ciuo_digit(label: str) -> int | None:
		m = re.match(r"^\s*(\d)", str(label))
		return int(m.group(1)) if m else None

	left_labels = sorted(
		biadj.columns.astype(str).unique(),
		key=lambda s: (
			_ciuo_digit(s) is None,
			_ciuo_digit(s) if _ciuo_digit(s) is not None else 999,
			str(s),
		),
	)
	right_labels = sorted(biadj.index.astype(str).unique())

	node_labels = [*left_labels, *right_labels]
	idx_left = {lab: i for i, lab in enumerate(left_labels)}
	offset_right = len(left_labels)
	idx_right = {lab: offset_right + i for i, lab in enumerate(right_labels)}

	sources = flows[letra_ciuo].astype(str).map(idx_left).astype(int).tolist()
	targets = flows[letra_caes].astype(str).map(idx_right).astype(int).tolist()
	values = flows["count"].astype(float).tolist()

	# Colors: color links by CIUO 1-digit.
	from plotly.colors import hex_to_rgb
	from plotly.colors import qualitative

	palette = list(getattr(qualitative, "Plotly", [])) or [
		"#636EFA",
		"#EF553B",
		"#00CC96",
		"#AB63FA",
		"#FFA15A",
		"#19D3F3",
		"#FF6692",
		"#B6E880",
		"#FF97FF",
		"#FECB52",
	]

	def _rgba(hex_color: str, a: float) -> str:
		if hex_to_rgb is None:
			return hex_color
		r, g, b = hex_to_rgb(hex_color)
		return f"rgba({r},{g},{b},{a})"

	left_color_map = {}
	for lab in left_labels:
		d = _ciuo_digit(lab)
		left_color_map[lab] = (
			palette[d % len(palette)] if d is not None else "rgba(160,160,160,0.9)"
		)

	link_colors = [
		_rgba(left_color_map[str(lab)], 0.35)
		for lab in flows[letra_ciuo].astype(str).tolist()
	]
	left_node_colors = [left_color_map[lab] for lab in left_labels]
	right_node_colors = ["rgba(200,200,200,0.7)"] * len(right_labels)
	node_colors = [*left_node_colors, *right_node_colors]

	fig = go.Figure(
		data=[
			go.Sankey(
				arrangement="snap",
				node={
					"pad": 15,
					"thickness": 20,
					"label": node_labels,
					"color": node_colors,
				},
				link={
					"source": sources,
					"target": targets,
					"value": values,
					"color": link_colors,
				},
			)
		]
	)
	fig.update_layout(
		title_text=None,
		width=1200,
		height=800,
	)

	fig.write_image(str(snakemake.output[0]), scale=2)


if __name__ == "__main__":
	main()
