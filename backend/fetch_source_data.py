"""Download the demo corpus from Wikipedia's public API."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://en.wikipedia.org/w/api.php"
ARTICLE_TITLES = (
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "BERT (language model)",
)
ATTRIBUTION = (
    'Source: Wikipedia articles "Transformer (deep learning architecture)",\n'
    '"Attention (machine learning)", and "BERT (language model)", used under\n'
    "CC BY-SA 4.0."
)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "source_text.txt"


def fetch_article(title: str) -> str:
    """Return the plain-text extract for a Wikipedia article."""
    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        }
    )
    request = Request(
        f"{API_URL}?{query}",
        headers={
            "User-Agent": "RAGEvalSidekick/0.1 (educational hackathon project)"
        },
    )

    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    page = payload["query"]["pages"][0]
    if page.get("missing"):
        raise RuntimeError(f"Wikipedia article not found: {title}")

    extract = page.get("extract", "").strip()
    if not extract:
        raise RuntimeError(f"Wikipedia returned no text for: {title}")
    return extract


def build_source_text() -> str:
    """Fetch and concatenate all configured articles."""
    sections = [
        f"===== {title} =====\n\n{fetch_article(title)}"
        for title in ARTICLE_TITLES
    ]
    return "\n\n".join(sections) + f"\n\n{ATTRIBUTION}\n"


def main() -> None:
    """Write the concatenated corpus to the sample-data directory."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_source_text(), encoding="utf-8")
    print(f"Saved Wikipedia source text to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
