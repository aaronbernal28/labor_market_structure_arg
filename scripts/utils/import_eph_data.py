import io
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

import src.scraping as scraping
import src.logging_utils as log

snakemake: any

URL = "https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos"


def _sleep_backoff(attempt: int) -> None:
	# attempt is 1-based
	time.sleep(float(attempt))


def _download_bytes(url: str, *, timeout_s: float = 30.0, retries: int = 3) -> bytes:
	last_exc: Exception | None = None
	for attempt in range(1, retries + 1):
		try:
			response = requests.get(url, timeout=timeout_s)
			response.raise_for_status()
			return response.content
		except requests.RequestException as exc:
			last_exc = exc
			if attempt < retries:
				_sleep_backoff(attempt)
				continue
			raise
	assert last_exc is not None
	raise last_exc


def _safe_filename_from_url(url: str, *, fallback: str) -> str:
	name = Path(urlparse(url).path).name
	return name or fallback


def main() -> None:
	output_log_path = Path(snakemake.output[0])
	dest_dir = output_log_path.parent
	if not dest_dir.exists():
		raise FileNotFoundError(
			f"Destination directory does not exist: {dest_dir}. "
			"Create it (e.g., data/raw/eph/) before running this rule."
		)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("IMPORT EPH DATA")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(log_lines, "CONFIG", [f"URL: {URL}", f"Dest dir: {dest_dir}"])

	links = scraping.get_indec_links(URL)
	if not links:
		log.add_notes(
			log_lines,
			"ERROR",
			["No links found (get_indec_links returned None/empty)."],
		)
		log.write_log(log_lines, output_log_path)
		raise ValueError("No links found on INDEC page.")

	data_links = [e for e in links if e.get("Type") == "Data (zip)"]
	design_links = [e for e in links if e.get("Type") == "Design (pdf)"]

	# Filter PDFs: only those with 'EPH_registro' (case-insensitive) in name or link.
	filtered_design_links: list[dict[str, str]] = []
	for entry in design_links:
		name = (entry.get("Name") or "").lower()
		link = (entry.get("Link") or "").lower()
		if "eph_registro" in name or "eph_registro" in link:
			filtered_design_links.append(entry)

	log.add_notes(
		log_lines,
		"LINKS SUMMARY",
		[
			f"Total links: {len(links)}",
			f"Data (zip) links: {len(data_links)}",
			f"Design (pdf) links: {len(design_links)}",
			f"Design PDFs matching 'EPH_registro': {len(filtered_design_links)}",
		],
	)

	log.add_notes(
		log_lines,
		"DATA LINKS (ALL FOUND)",
		[
			f"{i + 1:02d}. {e.get('Name', '')} | {e.get('Link', '')}"
			for i, e in enumerate(data_links)
		],
	)
	log.add_notes(
		log_lines,
		"DESIGN PDF LINKS (MATCHING, ALL FOUND)",
		[
			f"{i + 1:02d}. {e.get('Name', '')} | {e.get('Link', '')}"
			for i, e in enumerate(filtered_design_links)
		],
	)

	# ---- Download PDFs (side effects) ----
	pdf_processed = 0
	pdf_errors = 0
	for entry in filtered_design_links[:40]:
		pdf_url = entry.get("Link")
		if not pdf_url:
			continue
		try:
			filename = _safe_filename_from_url(pdf_url, fallback="design.pdf")
			if not filename.lower().endswith(".pdf"):
				filename = f"{filename}.pdf"
			pdf_path = dest_dir / filename
			pdf_bytes = _download_bytes(pdf_url)
			pdf_path.write_bytes(pdf_bytes)
			pdf_processed += 1
			log_lines.append(f"PDF downloaded: {pdf_path} | bytes={len(pdf_bytes)}")
		except Exception as exc:
			pdf_errors += 1
			log_lines.append(f"PDF download error: url={pdf_url} | error={exc!r}")

	# ---- Download ZIPs in-memory, export usu_individual_*.xlsx to CSV (side effects) ----
	zip_processed = 0
	zip_errors = 0
	xlsx_exported = 0
	xlsx_read_errors = 0

	for entry in data_links[:40]:
		zip_url = entry.get("Link")
		if not zip_url:
			continue
		if not zip_url.lower().endswith(".zip"):
			log_lines.append(f"Skipping non-zip data link: {zip_url}")
			continue

		try:
			zip_bytes = _download_bytes(zip_url)
			zip_processed += 1
			log_lines.append(
				f"ZIP downloaded in-memory: url={zip_url} | bytes={len(zip_bytes)}"
			)

			with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
				namelist = zf.namelist()
				preview = namelist[:500]
				log_lines.append(f"ZIP members count: {len(namelist)}")
				log_lines.append(f"ZIP members preview (cap 500): {preview!r}")

				# Match on basename, case-insensitive.
				matched_members: list[str] = []
				for name in namelist:
					base = Path(name).name
					base_lower = base.lower()
					if base_lower.startswith("usu_individual_") and base_lower.endswith(
						".xlsx"
					):
						matched_members.append(name)

				log_lines.append(
					f"Matched usu_individual_*.xlsx members: {matched_members!r}"
				)

				for member in matched_members:
					try:
						with zf.open(member) as f:
							df = pd.read_excel(f)
						csv_name = Path(member).name
						csv_name = csv_name[: -len(".xlsx")] + ".csv"
						csv_path = dest_dir / csv_name
						df.to_csv(csv_path, index=False, encoding="utf-8")
						xlsx_exported += 1
						log_lines.append(
							f"CSV exported: {csv_path} | rows={len(df)} | cols={len(df.columns)}"
						)
					except Exception as exc:
						xlsx_read_errors += 1
						log_lines.append(
							f"Excel read/export error: zip_url={zip_url} | member={member} | error={exc!r}"
						)

		except Exception as exc:
			zip_errors += 1
			log_lines.append(
				f"ZIP download/process error: url={zip_url} | error={exc!r}"
			)

	log.add_notes(
		log_lines,
		"RESULTS",
		[
			f"PDF processed: {pdf_processed} (errors={pdf_errors})",
			f"ZIP processed: {zip_processed} (errors={zip_errors})",
			f"Excel members exported to CSV: {xlsx_exported} (read/export errors={xlsx_read_errors})",
		],
	)

	log.write_log(log_lines, output_log_path)


if __name__ == "__main__":
	main()
