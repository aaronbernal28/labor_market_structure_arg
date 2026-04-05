from scripts import *
import numpy as np
import pandas as pd
from scipy import stats

FONT_SIZE = 15


def wald_test_comparison_proportions(df1, df2, caes_col, ciuo_col, rownames, colnames, alpha=0.05):
	"""
	Performs Wald test on proportions comparing two datasets with Bonferroni correction.

	Args:
		df1: DataFrame for dataset 1
		df2: DataFrame for dataset 2
		caes_col: Column name for CAES (industry)
		ciuo_col: Column name for CIUO (occupation)
		rownames: Row labels (CAES categories)
		colnames: Column labels (CIUO categories)
		alpha: Significance level (default 0.05)
	
	Returns:
		dict containing test results
	"""
	n1 = len(df1)
	n2 = len(df2)
	
	print(f"Sample sizes: n1={n1}, n2={n2}")
	
	# Create contingency tables (counts for each cell)
	# First, create a crosstab for each dataset
	crosstab1 = pd.crosstab(df1[caes_col], df1[ciuo_col])
	crosstab2 = pd.crosstab(df2[caes_col], df2[ciuo_col])
	
	# Ensure both have the same index and columns (fill missing with 0)
	all_caes = sorted(set(crosstab1.index) | set(crosstab2.index))
	all_ciuo = sorted(set(crosstab1.columns) | set(crosstab2.columns))
	
	crosstab1 = crosstab1.reindex(index=all_caes, columns=all_ciuo, fill_value=0)
	crosstab2 = crosstab2.reindex(index=all_caes, columns=all_ciuo, fill_value=0)
	
	# Filter to match the rownames and colnames provided
	crosstab1 = crosstab1.reindex(index=rownames, columns=colnames, fill_value=0)
	crosstab2 = crosstab2.reindex(index=rownames, columns=colnames, fill_value=0)
	
	# Convert to numpy arrays
	counts1 = crosstab1.to_numpy().astype(float)
	counts2 = crosstab2.to_numpy().astype(float)
	
	# Calculate proportions
	p1 = counts1 / n1
	p2 = counts2 / n2
	
	# Estimate delta: delta_hat = p1_hat - p2_hat
	delta_hat = p1 - p2
	
	# Calculate variances (binomial variance for proportions)
	# var(p_hat) = p(1-p) / n
	var1 = p1 * (1 - p1) / n1
	var2 = p2 * (1 - p2) / n2
	
	# Standard error: SE(delta_hat) = sqrt(var1 + var2)
	se = np.sqrt(var1 + var2)
	
	# Avoid division by zero
	se = np.where(se > 0, se, 1e-10)
	
	# Wald statistic: W = delta_hat / SE(delta_hat)
	W = delta_hat / se
	
	# P-values (two-tailed test using normal approximation)
	p_values = 2 * (1 - stats.norm.cdf(np.abs(W)))
	
	# Bonferroni correction
	d = delta_hat.size  # Total number of tests
	bonferroni_threshold = alpha / d
	
	# Reject H_0 if p_i < alpha/d
	rejected = p_values < bonferroni_threshold
	
	print(f"\n=== Wald Test Results (Proportions) ===")
	print(f"Total number of tests (d): {d}")
	print(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
	print(f"Number of rejections: {np.sum(rejected)} ({100*np.sum(rejected)/d:.2f}%)")
	print(f"Mean |delta_hat|: {np.mean(np.abs(delta_hat)):.6f}")
	print(f"Max |delta_hat|: {np.max(np.abs(delta_hat)):.6f}")
	print(f"Mean p-value: {np.mean(p_values):.4f}")
	print(f"Min p-value: {np.min(p_values):.2e}")
	
	return {
		'delta_hat': delta_hat,
		'se': se,
		'W': W,
		'p_values': p_values,
		'bonferroni_threshold': bonferroni_threshold,
		'rejected': rejected,
		'n1': n1,
		'n2': n2,
		'p1': p1,
		'p2': p2,
		'counts1': counts1,
		'counts2': counts2
	}


def bootstrap_se(df1, df2, caes_col, ciuo_col, rownames, colnames, B=1000, seed=28):
	"""Estimate SE of delta_hat = p1 - p2 via bootstrap (B resamples)."""
	deltas = np.zeros((B, len(rownames), len(colnames)))
	for b in range(B):
		s1 = df1.sample(n=len(df1), replace=True, random_state=seed+2*b)
		s2 = df2.sample(n=len(df2), replace=True, random_state=seed+2*b+1)
		ct1 = pd.crosstab(s1[caes_col], s1[ciuo_col]).reindex(index=rownames, columns=colnames, fill_value=0).to_numpy(dtype=float)
		ct2 = pd.crosstab(s2[caes_col], s2[ciuo_col]).reindex(index=rownames, columns=colnames, fill_value=0).to_numpy(dtype=float)
		deltas[b] = ct1 / len(df1) - ct2 / len(df2)
	return deltas.std(axis=0, ddof=1)


def main(enes_df=None, nodelist_caes_df=None, nodelist_ciuo_df=None):
	# Load datasets with metadata
	data_2019 = dl.load_dataset(RAW_ENES_PATH, RAW_CAES_NODELIST_PATH, RAW_CIUO_NODELIST_PATH, CAES_ID, CIUO_ID)
	data_2021 = dl.load_dataset(RAW_ENES_2021_PATH, RAW_CAES_NODELIST_PATH, RAW_CIUO_NODELIST_PATH, CAES_ID, CIUO_ID)

	# Get unique CAES and CIUO labels for heatmap axes
	rownames_caes = data_2019["caes_nodes"][CAES_LETRA].drop_duplicates().sort_values()
	colnames_ciuo = data_2019["ciuo_nodes"][CIUO_LETRA].drop_duplicates().sort_values()

	# Extract base ENES dataframes for comparison
	base_enes_2019_df = data_2019["enes"]
	base_enes_2021_df = data_2021["enes"]
	base_enes_unif_df = pd.concat([base_enes_2019_df, base_enes_2021_df], ignore_index=True, axis=0).drop_duplicates([CAES_ID, CIUO_ID])

	# Find common columns between the two datasets
	common_columns = sorted(list(set(base_enes_2019_df.columns) & set(base_enes_2021_df.columns)))
	base_enes_2019_df = base_enes_2019_df[common_columns]
	base_enes_2021_df = base_enes_2021_df[common_columns]

	print("=" * 60)
	print("COMPARING ENES 2019 AND ESAyPP 2021 DATASETS")
	print("=" * 60)
	
	comparison_df = pd.DataFrame({
		"Dataset": ["ENES 2019", "ESAyPP 2021"],
		"Total Records": [len(base_enes_2019_df), len(base_enes_2021_df)],
		"Unique CAES": [base_enes_2019_df[CAES_ID].nunique(), base_enes_2021_df[CAES_ID].nunique()],
		"Unique CIUO": [base_enes_2019_df[CIUO_ID].nunique(), base_enes_2021_df[CIUO_ID].nunique()],
	})

	print("\n", comparison_df.to_string(index=False))

	# Generate visual comparison of heatmaps
	print("\n" + "=" * 60)
	print("GENERATING HEATMAP COMPARISONS...")
	print("=" * 60)

	N = min(len(base_enes_2019_df), len(base_enes_2021_df))
	base_enes_2019_sampled_df = base_enes_2019_df.sample(n=N, random_state=42, replace=True)
	base_enes_2021_sampled_df = base_enes_2021_df.sample(n=N, random_state=43, replace=True)
	base_enes_unif_sampled_df = base_enes_unif_df.sample(n=N, random_state=44, replace=True)

	biadjacency_2019 = dl.build_biadjacency(
		base_enes_2019_sampled_df,
		CAES_LETRA,
		CIUO_LETRA,
		logscale=True,
		rownames=rownames_caes,
		colnames=colnames_ciuo
	)

	biadjacency_2021 = dl.build_biadjacency(
		base_enes_2021_sampled_df,
		CAES_LETRA,
		CIUO_LETRA,
		logscale=True,
		rownames=rownames_caes,
		colnames=colnames_ciuo
	)

	biadjacency_unif = dl.build_biadjacency(
		base_enes_unif_sampled_df,
		CAES_LETRA,
		CIUO_LETRA,
		logscale=True,
		rownames=rownames_caes,
		colnames=colnames_ciuo
	)

	pl.plot_heatmap(biadjacency_2019, output_path=IMAGE_DIR / "12_heatmap_2019_sampled.png", save=True, font_size=FONT_SIZE)
	pl.plot_heatmap(biadjacency_2021, output_path=IMAGE_DIR / "12_heatmap_2021_sampled.png", save=True, font_size=FONT_SIZE)
	pl.plot_heatmap(biadjacency_unif, output_path=IMAGE_DIR / "12_heatmap_unif_sampled.png", save=True, font_size=FONT_SIZE)
	print(f"Saved heatmaps to:")
	print(f"  - {IMAGE_DIR / '12_heatmap_2019_sampled.png'}")
	print(f"  - {IMAGE_DIR / '12_heatmap_2021_sampled.png'}")
	print(f"  - {IMAGE_DIR / '12_heatmap_unif_sampled.png'}")

	# Perform Wald test on proportions
	print("\n" + "=" * 60)
	print("PERFORMING WALD TEST ON PROPORTIONS")
	print("=" * 60)

	print(f"Dataset sizes: ENES 2019 = {len(base_enes_2019_df)}, ESAyPP 2021 = {len(base_enes_2021_df)}")
	print(f"Matrix dimensions: ({len(rownames_caes)}, {len(colnames_ciuo)})")
	
	# Perform Wald test on proportions
	test_results = wald_test_comparison_proportions(
		base_enes_2019_df,
		base_enes_2021_df,
		CAES_LETRA,
		CIUO_LETRA,
		rownames_caes.tolist(),
		colnames_ciuo.tolist(),
		alpha=0.05
	)

	# Generate visualizations
	print("\n" + "=" * 60)
	print("GENERATING TEST RESULT VISUALIZATIONS")
	print("=" * 60)

	# P-value heatmap: black below Bonferroni threshold, YlOrRd above
	pvalue_output = IMAGE_DIR / "12_wald_test_pvalues_detailed.png"
	pl.plot_rejection_heatmap(
		test_results['p_values'],
		test_results['rejected'],
		rownames_caes.tolist(),
		colnames_ciuo.tolist(),
		test_results['bonferroni_threshold'],
		pvalue_output,
		save=True,
		font_size=FONT_SIZE
	)
	print(f"Saved p-value heatmap to {pvalue_output}")

	# Delta (difference) heatmap
	delta_output = IMAGE_DIR / "12_wald_test_delta.png"
	pl.plot_delta_heatmap(
		test_results['delta_hat'],
		rownames_caes.tolist(),
		colnames_ciuo.tolist(),
		delta_output,
		save=True,
		font_size=FONT_SIZE
	)
	print(f"Saved delta heatmap to {delta_output}")

	# Bootstrap SE heatmap (B=1000)
	print("\nComputing bootstrap SE (B=1000)...")
	se_boot = bootstrap_se(
		base_enes_2019_df, base_enes_2021_df,
		CAES_LETRA, CIUO_LETRA,
		rownames_caes.tolist(), colnames_ciuo.tolist(),
		B=1000
	)
	se_output = IMAGE_DIR / "12_wald_test_bootstrap_se.png"
	pl.plot_bootstrap_se_heatmap(
		se_boot,
		rownames_caes.tolist(),
		colnames_ciuo.tolist(),
		se_output,
		save=True,
		font_size=FONT_SIZE
	)
	print(f"Saved bootstrap SE heatmap to {se_output}")
	print("\n" + "=" * 60)
	print("REJECTED PAIRS (NULL HYPOTHESIS REJECTED)")
	print("=" * 60)
	rejected_indices = np.argwhere(test_results['rejected'])
	if len(rejected_indices) > 0:
		print(f"\nFound {len(rejected_indices)} rejected pairs:\n")
		print(f"{'CAES':<55} {'CIUO':<50} {'delta':<12} {'se_boot':<12} {'p-value':<12} {'Sig.':<6}")
		print("-" * 140)
		
		# Map back to original labels
		rownames_list = rownames_caes.tolist()
		colnames_list = colnames_ciuo.tolist()
		
		for row_idx, col_idx in rejected_indices:
			caes_letra = rownames_list[row_idx]
			ciuo_letra = colnames_list[col_idx]
			
			# Get original labels from the dataframes
			caes_label = data_2019["caes_nodes"][data_2019["caes_nodes"][CAES_LETRA] == caes_letra][CAES_LETRA_OLD].iloc[0]
			ciuo_label = data_2019["ciuo_nodes"][data_2019["ciuo_nodes"][CIUO_LETRA] == ciuo_letra][CIUO_LETRA_OLD].iloc[0]
			
			p_value = test_results['p_values'][row_idx, col_idx]
			delta = test_results['delta_hat'][row_idx, col_idx]
			
			# Determine significance stars
			if p_value < 0.001/len(test_results['delta_hat'].flatten()):
				sig_stars = "***"
			elif p_value < 0.01/len(test_results['delta_hat'].flatten()):
				sig_stars = "**"
			elif p_value < 0.05/len(test_results['delta_hat'].flatten()):
				sig_stars = "*"
			else:
				sig_stars = ""
			
			print(f"{caes_label:<55} {ciuo_label:<50} {delta:<12.4f} {se_boot[row_idx, col_idx]:<12.4f} {p_value:<12.2e} {sig_stars:<6}")
		print(f"\nNote: Significance stars indicate levels of significance after Bonferroni correction.")
	else:
		print("\nNo pairs rejected (all p-values above Bonferroni threshold)")
if __name__ == "__main__":
	main()
