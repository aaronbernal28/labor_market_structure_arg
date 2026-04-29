import io
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

import src.scraping as scraping
import src.logging_utils as log

snakemake: Any

URL = "https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos"

DEFAULT_HEADERS = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def _sleep_backoff(attempt: int) -> None:
	# attempt is 1-based
	time.sleep(float(attempt))


def _download_bytes(
	url: str,
	*,
	headers: dict[str, str] | None = None,
	timeout_s: float = 30.0,
	retries: int = 3,
) -> bytes:
	last_exc: Exception | None = None
	for attempt in range(1, retries + 1):
		try:
			response = requests.get(url, timeout=timeout_s, headers=headers)
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


def _merge_headers(defaults: dict[str, str], override: object) -> dict[str, str]:
	headers = dict(defaults)
	if isinstance(override, dict):
		for key, value in override.items():
			headers[str(key)] = str(value)
	return headers


def _entry_text(entry: dict[str, str]) -> tuple[str, str]:
	name = (entry.get("Name") or "").lower()
	link = (entry.get("Link") or "").lower()
	return name, link


def _filter_by_year(
	entries: list[dict[str, str]], *, year_target: str, year_target_2digit: str
) -> tuple[list[dict[str, str]], str]:
	primary = year_target.lower()
	fallback = year_target_2digit.lower()
	primary_matches = [
		e
		for e in entries
		if primary in (e.get("Name") or "").lower()
		or primary in (e.get("Link") or "").lower()
	]
	if primary_matches:
		return primary_matches, primary
	if fallback and fallback != primary:
		fallback_matches = [
			e
			for e in entries
			if fallback in (e.get("Name") or "").lower()
			or fallback in (e.get("Link") or "").lower()
		]
		return fallback_matches, fallback
	return [], primary


def main() -> None:
	output_log_path = Path(snakemake.output[0])
	year_target = str(snakemake.wildcards.get("year"))
	year_target_2digit = year_target[2:] if len(year_target) == 4 else year_target

	dest_dir = output_log_path.parent
	dest_dir.mkdir(parents=True, exist_ok=True)

	headers = _merge_headers(DEFAULT_HEADERS, snakemake.config.get("http_headers"))

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("IMPORT EPH DATA")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"CONFIG",
		[
			f"URL: {URL}",
			f"Dest dir: {dest_dir}",
			f"year_target: {year_target}",
			f"year_target_2digit: {year_target_2digit}",
			f"User-Agent: {headers.get('User-Agent', '')}",
		],
	)

	links = scraping.get_indec_links(URL, request_kwargs={"headers": headers})
	if not links:
		log.add_notes(
			log_lines,
			"ERROR",
			["No links found (get_indec_links returned None/empty)."],
		)
		log.write_log(log_lines, output_log_path)
		return

	data_links = [e for e in links if e.get("Type") == "Data (zip)"]
	design_links = [e for e in links if e.get("Type") == "Design (pdf)"]

	data_links_year, year_token_used_data = _filter_by_year(
		data_links,
		year_target=year_target,
		year_target_2digit=year_target_2digit,
	)
	design_links_year, year_token_used_design = _filter_by_year(
		design_links,
		year_target=year_target,
		year_target_2digit=year_target_2digit,
	)

	data_links_xls_year: list[dict[str, str]] = []
	for entry in data_links_year:
		name, link = _entry_text(entry)
		is_xls = "formato xls" in name or "_xls.zip" in link or link.endswith("xls.zip")
		is_zip = link.endswith(".zip")
		if is_zip and is_xls:
			data_links_xls_year.append(entry)

	# Filter PDFs: only those with 'EPH_registro' (case-insensitive) in name or link.
	filtered_design_links: list[dict[str, str]] = []
	for entry in design_links_year:
		name, link = _entry_text(entry)
		if "eph_registro" in name or "eph_registro" in link:
			filtered_design_links.append(entry)

	log.add_notes(
		log_lines,
		"LINKS SUMMARY",
		[
			f"Total links: {len(links)}",
			f"Data (zip) links: {len(data_links)}",
			f"Data (zip) links matching year (token={year_token_used_data!r}): {len(data_links_year)}",
			f"Data (zip) links matching year + XLS ZIP: {len(data_links_xls_year)}",
			f"Design (pdf) links: {len(design_links)}",
			f"Design (pdf) links matching year (token={year_token_used_design!r}): {len(design_links_year)}",
			f"Design PDFs matching 'EPH_registro': {len(filtered_design_links)}",
		],
	)

	if not data_links_xls_year and not filtered_design_links:
		log.add_notes(
			log_lines,
			"ERROR",
			[
				"No matching resources for requested year.",
				"Expected at least one XLS ZIP and/or one EPH_registro PDF.",
			],
		)
		log.write_log(log_lines, output_log_path)
		return

	log.add_notes(
		log_lines,
		"DATA LINKS (FILTERED TO YEAR+XLS ZIP, CAP 100)",
		[
			f"{i + 1:02d}. {e.get('Name', '')} | {e.get('Link', '')}"
			for i, e in enumerate(data_links_xls_year[:100])
		],
	)
	log.add_notes(
		log_lines,
		"DESIGN PDF LINKS (MATCHING, FILTERED TO YEAR, CAP 100)",
		[
			f"{i + 1:02d}. {e.get('Name', '')} | {e.get('Link', '')}"
			for i, e in enumerate(filtered_design_links[:100])
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
			pdf_bytes = _download_bytes(pdf_url, headers=headers)
			pdf_path.write_bytes(pdf_bytes)
			pdf_processed += 1
			log_lines.append(f"PDF downloaded: {pdf_path} | bytes={len(pdf_bytes)}")
		except Exception as exc:
			pdf_errors += 1
			log_lines.append(f"PDF download error: url={pdf_url} | error={exc!r}")

	# ---- Download ZIPs in-memory, export usu_individual_*.xls(x) to CSV (side effects) ----
	zip_processed = 0
	zip_errors = 0
	xlsx_exported = 0
	xlsx_read_errors = 0

	for entry in data_links_xls_year[:40]:
		zip_url = entry.get("Link")
		if not zip_url:
			continue
		if not zip_url.lower().endswith(".zip"):
			log_lines.append(f"Skipping non-zip data link: {zip_url}")
			continue

		try:
			zip_bytes = _download_bytes(zip_url, headers=headers)
			zip_processed += 1
			log_lines.append(
				f"ZIP downloaded in-memory: url={zip_url} | bytes={len(zip_bytes)}"
			)

			with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
				namelist = zf.namelist()
				preview = namelist[:500]
				log_lines.append(f"ZIP members count: {len(namelist)}")
				log_lines.append(f"ZIP members preview (cap 500): {preview!r}")

				# Match on basename, case-insensitive. Support both .xls and .xlsx
				matched_members: list[str] = []
				for name in namelist:
					base = Path(name).name
					base_lower = base.lower()
					is_excel = base_lower.endswith(".xls") or base_lower.endswith(
						".xlsx"
					)
					if base_lower.startswith("usu_individual_") and is_excel:
						matched_members.append(name)

				log_lines.append(
					f"Matched usu_individual_*.xls(x) members: {matched_members!r}"
				)

				for member in matched_members:
					try:
						with zf.open(member) as f:
							df = pd.read_excel(f)
						csv_name = Path(member).name
						# Remove .xls or .xlsx extension and add .csv
						if csv_name.lower().endswith(".xlsx"):
							csv_name = csv_name[: -len(".xlsx")] + ".csv"
						elif csv_name.lower().endswith(".xls"):
							csv_name = csv_name[: -len(".xls")] + ".csv"
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
