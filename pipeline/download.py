"""Download the EU MRV (THETIS-MRV) yearly emission files from EMSA.

Usage:
    python -m pipeline.download              # all years
    python -m pipeline.download --year 2024  # one year only

Files go to data/raw/{year}.xlsx and every download is recorded in data/raw/manifest.json.
A year is skipped when the manifest already has EMSA's current version and the file is on disk.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "https://mrv.emsa.europa.eu/api/public-emission-report"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
USER_AGENT = "ShipCarbon/0.1 (educational data project; python-httpx)"
TIMEOUT = httpx.Timeout(60.0, connect=15.0)
RETRIES = 3
RETRY_DELAY_S = 3.0      # wait 3 s, then 6 s, then 9 s between attempts
PAUSE_BETWEEN_FILES_S = 1.5


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """One shared client: same headers and timeout for every request."""
    return httpx.Client(
        base_url=BASE_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        follow_redirects=True,
        transport=transport,
    )


def get_with_retries(client: httpx.Client, url: str, sleep=time.sleep) -> httpx.Response:
    """GET a URL, retrying network errors and server errors (5xx / 429) a few times."""
    for attempt in range(1, RETRIES + 1):
        try:
            response = client.get(url)
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()  # 4xx (e.g. 404) is a real error: don't retry
                return response
            error: Exception = httpx.HTTPStatusError(
                f"server answered {response.status_code}", request=response.request, response=response
            )
        except httpx.TransportError as exc:  # timeouts, dropped connections, DNS problems
            error = exc
        if attempt < RETRIES:
            sleep(RETRY_DELAY_S * attempt)
    raise error


def fetch_available(client: httpx.Client, sleep=time.sleep) -> list[dict]:
    """Ask EMSA which years exist and the current file version of each one."""
    periods = get_with_retries(client, "/reporting-periods", sleep).json()
    files = get_with_retries(client, "/downloadable-files", sleep).json()["results"]
    by_year = {f["reportingPeriod"]: f for f in files}

    available = []
    for year in sorted(p["value"] for p in periods):
        f = by_year.get(year)
        if f is None:
            print(f"  {year}: listed as a reporting period but EMSA has no downloadable file yet")
            continue
        available.append({
            "year": year,
            "version": f["version"],
            "generation_date": f["generationDate"],
            "file_name": f["fileName"],
        })
    return available


def load_manifest(raw_dir: Path) -> dict:
    path = raw_dir / "manifest.json"
    if not path.exists():
        return {"source": BASE_URL, "years": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(raw_dir: Path, manifest: dict) -> None:
    path = raw_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_up_to_date(entry: dict, manifest: dict, raw_dir: Path) -> bool:
    """True when we already have this exact version on disk."""
    known = manifest["years"].get(str(entry["year"]))
    return (
        known is not None
        and known["version"] == entry["version"]
        and (raw_dir / f"{entry['year']}.xlsx").exists()
    )


def download_year(client: httpx.Client, entry: dict, raw_dir: Path, sleep=time.sleep) -> dict:
    """Download one year's xlsx and return its manifest record."""
    year, version = entry["year"], entry["version"]
    response = get_with_retries(client, f"/reporting-period-document/binary/{year}/{version}", sleep)
    content = response.content
    if not content.startswith(b"PK"):  # every .xlsx is a zip file, and zip files start with "PK"
        raise ValueError(f"{year}: response is not an xlsx file ({response.headers.get('content-type')})")

    final = raw_dir / f"{year}.xlsx"
    partial = raw_dir / f"{year}.xlsx.part"
    partial.write_bytes(content)
    partial.replace(final)  # only now does {year}.xlsx appear, so it is never half-written

    return {
        **entry,
        "file_size_bytes": len(content),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run(client: httpx.Client, raw_dir: Path, year: int | None = None, sleep=time.sleep) -> dict[str, list[int]]:
    """Download every missing or outdated year. Returns which years were downloaded/skipped/failed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)
    available = fetch_available(client, sleep)
    if year is not None:
        available = [e for e in available if e["year"] == year]
        if not available:
            raise SystemExit(f"Year {year} is not available from EMSA.")

    result = {"downloaded": [], "skipped": [], "failed": []}
    for i, entry in enumerate(available):
        label = f"{entry['year']} (v{entry['version']})"
        if is_up_to_date(entry, manifest, raw_dir):
            print(f"  {label}: skipped, already up to date")
            result["skipped"].append(entry["year"])
            continue
        old = manifest["years"].get(str(entry["year"]))
        try:
            record = download_year(client, entry, raw_dir, sleep)
        except Exception as exc:
            print(f"  {label}: FAILED - {exc}")
            result["failed"].append(entry["year"])
            continue
        manifest["years"][str(entry["year"])] = record
        save_manifest(raw_dir, manifest)
        note = f", replaced v{old['version']}" if old else ""
        print(f"  {label}: downloaded {record['file_size_bytes'] / 1e6:.1f} MB{note}")
        result["downloaded"].append(entry["year"])
        if i < len(available) - 1:
            sleep(PAUSE_BETWEEN_FILES_S)  # be polite to EMSA's server
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download EU MRV emission files from EMSA.")
    parser.add_argument("--year", type=int, help="download only this reporting year")
    args = parser.parse_args(argv)

    print(f"Checking EMSA for available files -> {RAW_DIR}")
    with make_client() as client:
        result = run(client, RAW_DIR, args.year)
    print(
        f"Done: {len(result['downloaded'])} downloaded, "
        f"{len(result['skipped'])} skipped, {len(result['failed'])} failed"
    )
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
