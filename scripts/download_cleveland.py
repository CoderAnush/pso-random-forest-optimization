"""Download the UCI Cleveland Heart Disease file once and record its provenance (Phase 2, ADR-009).

Usage:
    python scripts/download_cleveland.py [--force] [--data-dir DATA_DIR]

Writes ``<data-dir>/raw/processed.cleveland.data`` (the bytes exactly as served) and
``<data-dir>/raw/MANIFEST.json`` (URL used, UTC download time, SHA-256, size, citation, licence).
The direct UCI archive URL is tried first, then the ``heart+disease.zip`` bundle; no other source
is ever used. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRIMARY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
FALLBACK_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
FILE_NAME = "processed.cleveland.data"
CITATION = (
    "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X"
)
LICENSE = "CC BY 4.0"
UCI_ID = 45
N_COLUMNS = 14
USER_AGENT = "pso-rf-download/0.1 (+https://github.com/CoderAnush/pso-random-forest-optimization)"
TIMEOUT_S = 60
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Also follow HTTP 308; Python 3.10's urllib follows only 301, 302, 303 and 307."""

    def http_error_308(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any) -> Any:
        return self.http_error_302(req, fp, 307, msg, headers)


def fetch(url: str) -> bytes:
    """GET ``url`` (following redirects) and return the response body."""
    opener = urllib.request.build_opener(_RedirectHandler)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _is_number(field: str) -> bool:
    try:
        float(field)
    except ValueError:
        return False
    return True


def check_cleveland(data: bytes) -> int:
    """Return the row count if ``data`` looks like processed.cleveland.data; raise ValueError otherwise."""
    lines = [line for line in data.decode("ascii").splitlines() if line.strip()]
    if not lines:
        raise ValueError("the file is empty")
    for number, line in enumerate(lines, 1):
        fields = line.split(",")
        if len(fields) != N_COLUMNS or not all(f.strip() == "?" or _is_number(f) for f in fields):
            raise ValueError(f"line {number} is not {N_COLUMNS} numeric or '?' fields: {line[:80]!r}")
    return len(lines)


def download() -> tuple[bytes, str]:
    """Fetch the file from the primary URL, else from the zip bundle; return (bytes, url_used)."""
    failures = []
    try:
        data = fetch(PRIMARY_URL)
        check_cleveland(data)
        return data, PRIMARY_URL
    except (OSError, ValueError) as exc:  # URLError/HTTPError are OSErrors; decode errors are ValueErrors
        failures.append(f"{PRIMARY_URL}: {exc}")
    try:
        with zipfile.ZipFile(io.BytesIO(fetch(FALLBACK_URL))) as archive:
            members = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == FILE_NAME]
            if not members:
                raise ValueError(f"{FILE_NAME} is not in the archive")
            data = archive.read(members[0])
        check_cleveland(data)
        return data, FALLBACK_URL
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        failures.append(f"{FALLBACK_URL}: {exc}")
    raise RuntimeError("both UCI sources failed (no substitute dataset is used):\n  " + "\n  ".join(failures))


def main(argv: list[str] | None = None) -> int:
    """Download (unless a verified copy exists), write the file and its manifest, and print a summary."""
    parser = argparse.ArgumentParser(description="Download the UCI Cleveland Heart Disease file once.")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="data directory (default: data/)"
    )
    parser.add_argument("--force", action="store_true", help="download again even if a verified copy exists")
    args = parser.parse_args(argv)

    raw_dir = args.data_dir / "raw"
    target = raw_dir / FILE_NAME
    manifest_path = raw_dir / "MANIFEST.json"
    if target.exists() and manifest_path.exists() and not args.force:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("sha256")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual == recorded:
            print(f"{target} matches MANIFEST.json (sha256 {actual}); use --force to download again.")
            return 0
        print(f"{target} does not match MANIFEST.json; downloading again.")

    data, url = download()
    raw_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest = {
        "file": FILE_NAME,
        "source_url_used": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "citation": CITATION,
        "license": LICENSE,
        "uci_id": UCI_ID,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2) + "\n")
    print(f"downloaded {check_cleveland(data)} rows ({len(data)} bytes) from {url}")
    print(f"sha256 {manifest['sha256']}")
    print(f"wrote {target} and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
