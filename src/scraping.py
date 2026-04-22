"""
This module contains functions for scraping data from the INDEC website.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict

def get_indec_links(url: str) -> List[Dict[str, str]] | None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        links_found = []

        # INDEC often uses 'a' tags with specific text for documentation and data
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().lower()

            # Filters based on your requirements
            is_xls = ".xls" in href or "formato xls" in text
            is_design = "diseño de registro" in text or "estructura" in text

            if is_xls or is_design:
                full_url = href if href.startswith('http') else f"https://www.indec.gob.ar{href}"
                links_found.append({
                    "Type": "Data (zip)" if is_xls else "Design (pdf)",
                    "Name": link.get_text(strip=True),
                    "Link": full_url
                })

        return links_found

    except Exception as e:
        print(f"Error accessing the site: {e}")
        return None
