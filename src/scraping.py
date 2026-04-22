"""src.scraping

Scraping utilities and small helpers for fetching INDEC resources.

The functions added below are intentionally independent and can be called from
scripts, notebooks, or Snakemake rules.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_indec_links(url: str, _depth: int = 0) -> List[Dict[str, str]] | None:
	try:
		response = requests.get(url, timeout=30)
		response.raise_for_status()
		soup = BeautifulSoup(response.text, "html.parser")

		links_found = []

		# INDEC often uses 'a' tags with specific text for documentation and data
		for link in soup.find_all("a", href=True):
			href = str(link["href"])
			text = link.get_text().lower()

			# Filters based on your requirements
			is_xls = (
				".zip" in href
				or ".xlsx" in href
				or ".xls" in href
				or "formato xls" in text
				or "formato zip" in text
			)
			is_design = "diseño de registro" in text or "estructura" in text

			if is_xls or is_design:
				full_url = (
					href
					if href.startswith("http")
					else f"https://www.indec.gob.ar{href}"
				)
				links_found.append(
					{
						"Type": "Data (zip)" if is_xls else "Design (pdf)",
						"Name": link.get_text(strip=True),
						"Link": full_url,
					}
				)

		# INDEC uses an SPA shell that stores the real backend view in a hidden
		# input (id="VistaCarga"). When we don't find any direct download links,
		# follow that view once.
		if not links_found and _depth < 1:
			vista_carga = soup.find(id="VistaCarga")
			view = None
			if vista_carga is not None:
				view = vista_carga.get("value")
			if view:
				view_url = f"https://www.indec.gob.ar/{str(view).lstrip('/')}"
				if view_url != url:
					fallback = get_indec_links(view_url, _depth=_depth + 1)
					if fallback:
						return fallback

		return links_found

	except Exception as e:
		print(f"Error accessing the site: {e}")
		return None


def _infer_filename_from_url(url: str, *, fallback: str = "download") -> str:
	path = urlparse(url).path
	name = Path(path).name
	return name or fallback


def _select_indec_link(
	links: List[Dict[str, str]] | None,
	*,
	link_type: str,
	prefer_suffixes: tuple[str, ...] = (),
) -> Dict[str, str]:
	if not links:
		raise ValueError("No links provided (get_indec_links() returned empty/None).")

	candidates = [entry for entry in links if entry.get("Type") == link_type]
	if not candidates:
		raise ValueError(f"No link found with Type={link_type!r}.")

	if prefer_suffixes:
		for suffix in prefer_suffixes:
			for entry in candidates:
				url = entry.get("Link", "")
				if url.lower().endswith(suffix.lower()):
					return entry

	return candidates[0]


def _download_url(
	url: str,
	*,
	dest_path: Path,
	overwrite: bool = False,
	timeout_s: float = 30.0,
	chunk_size: int = 1024 * 1024,
	request_kwargs: Mapping[str, Any] | None = None,
) -> Path:
	"""Download a URL to a local path using streaming I/O."""
	dest_path = Path(dest_path)
	dest_path.parent.mkdir(parents=True, exist_ok=True)

	if dest_path.exists() and not overwrite:
		return dest_path

	kwargs: dict[str, Any] = dict(request_kwargs or {})
	with requests.get(url, stream=True, timeout=timeout_s, **kwargs) as response:
		response.raise_for_status()
		with dest_path.open("wb") as f:
			for chunk in response.iter_content(chunk_size=chunk_size):
				if chunk:
					f.write(chunk)

	return dest_path


def download_design_pdf_from_links(
	links: List[Dict[str, str]] | None,
	*,
	dest_dir: str | Path = "data/raw/eph",
	overwrite: bool = False,
	timeout_s: float = 30.0,
) -> Path:
	"""Download the first "Design (pdf)" link to disk and return its path."""
	entry = _select_indec_link(
		links,
		link_type="Design (pdf)",
		prefer_suffixes=(".pdf",),
	)
	url = entry["Link"]
	filename = _infer_filename_from_url(url, fallback="design.pdf")
	if not filename.lower().endswith(".pdf"):
		filename = f"{filename}.pdf"

	dest_path = Path(dest_dir) / filename
	return _download_url(
		url, dest_path=dest_path, overwrite=overwrite, timeout_s=timeout_s
	)


def load_design_pdf_from_links(
	links: List[Dict[str, str]] | None,
	*,
	dest_dir: str | Path = "data/raw/eph",
) -> bytes:
	"""Load previously-downloaded design PDF bytes (does not download)."""
	entry = _select_indec_link(
		links,
		link_type="Design (pdf)",
		prefer_suffixes=(".pdf",),
	)
	url = entry["Link"]
	filename = _infer_filename_from_url(url, fallback="design.pdf")
	if not filename.lower().endswith(".pdf"):
		filename = f"{filename}.pdf"

	path = Path(dest_dir) / filename
	if not path.exists():
		raise FileNotFoundError(
			f"Design PDF not found at {path}. "
			"Run download_design_pdf_from_links(...) first."
		)
	return path.read_bytes()


def load_usu_individual_t325_from_links(
	links: List[Dict[str, str]] | None,
	*,
	dest_dir: str | Path = "data/raw/eph",
	member_name: str = "usu_individual_T325.xlsx",
	overwrite_download: bool = False,
	overwrite_extract: bool = False,
	timeout_s: float = 30.0,
	read_excel_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
	"""Download the "Data (zip)" resource, unzip, and load the Excel as a DataFrame."""
	entry = _select_indec_link(
		links,
		link_type="Data (zip)",
		prefer_suffixes=(".zip", ".xlsx", ".xls"),
	)
	url = entry["Link"]
	filename = _infer_filename_from_url(url, fallback="data.zip")
	download_path = Path(dest_dir) / filename
	download_path = _download_url(
		url,
		dest_path=download_path,
		overwrite=overwrite_download,
		timeout_s=timeout_s,
	)

	member_dest_path = Path(dest_dir) / member_name

	if download_path.suffix.lower() == ".zip":
		if member_dest_path.exists() and not overwrite_extract:
			excel_path = member_dest_path
		else:
			with zipfile.ZipFile(download_path) as zf:
				namelist = zf.namelist()
				member = None
				if member_name in namelist:
					member = member_name
				else:
					# Some INDEC zips include a directory prefix.
					member_lower = member_name.lower()
					for name in namelist:
						if name.lower().endswith(member_lower):
							member = name
							break
				if member is None:
					raise ValueError(
						f"ZIP file does not contain {member_name!r}. "
						f"Members preview={namelist[:20]!r}"
					)
				member_dest_path.parent.mkdir(parents=True, exist_ok=True)
				with zf.open(member) as src, member_dest_path.open("wb") as dst:
					shutil.copyfileobj(src, dst)
			excel_path = member_dest_path
	elif download_path.suffix.lower() in {".xlsx", ".xls"}:
		excel_path = download_path
	else:
		raise ValueError(
			"Unsupported downloaded file type for 'Data (zip)' link: "
			f"{download_path.name}"
		)

	kwargs = dict(read_excel_kwargs or {})
	return pd.read_excel(excel_path, **kwargs)
