"""Tests for pipeline/download.py. A fake EMSA server stands in for the network."""

import json

import httpx
import pytest

from pipeline import download

FAKE_XLSX = b"PK\x03\x04 fake xlsx bytes"


class FakeEmsa:
    """Answers the three EMSA endpoints and records which files were downloaded."""

    def __init__(self, versions: dict[int, int]):
        self.versions = versions          # {year: current version}
        self.downloads: list[str] = []    # binary URLs that were requested
        self.fail_next: list[int] = []    # status codes to answer before the next success

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0))
        path = request.url.path.removeprefix("/api/public-emission-report")
        if path == "/reporting-periods":
            return httpx.Response(200, json=[{"value": y} for y in sorted(self.versions, reverse=True)])
        if path == "/downloadable-files":
            return httpx.Response(200, json={"results": [
                {"reportingPeriod": y, "version": v, "generationDate": "01-01-2026 05:00:00",
                 "fileName": f"{y}-v{v}-EU MRV Publication of information"}
                for y, v in self.versions.items()
            ]})
        if path.startswith("/reporting-period-document/binary/"):
            self.downloads.append(path)
            return httpx.Response(200, content=FAKE_XLSX)
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return download.make_client(transport=httpx.MockTransport(self.handler))


def no_sleep(_seconds):
    pass


def run(emsa: FakeEmsa, raw_dir, year=None):
    with emsa.client() as client:
        return download.run(client, raw_dir, year=year, sleep=no_sleep)


def read_manifest(raw_dir):
    return json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))


def test_first_run_downloads_every_year_and_writes_manifest(tmp_path):
    emsa = FakeEmsa({2018: 5, 2019: 7})

    result = run(emsa, tmp_path)

    assert result == {"downloaded": [2018, 2019], "skipped": [], "failed": []}
    assert (tmp_path / "2018.xlsx").read_bytes() == FAKE_XLSX
    record = read_manifest(tmp_path)["years"]["2019"]
    assert record["version"] == 7
    assert record["generation_date"] == "01-01-2026 05:00:00"
    assert record["file_name"] == "2019-v7-EU MRV Publication of information"
    assert record["file_size_bytes"] == len(FAKE_XLSX)
    assert record["downloaded_at_utc"].endswith("+00:00")


def test_second_run_with_same_versions_downloads_nothing(tmp_path):
    emsa = FakeEmsa({2018: 5, 2019: 7})
    run(emsa, tmp_path)
    emsa.downloads.clear()

    result = run(emsa, tmp_path)

    assert result["skipped"] == [2018, 2019]
    assert result["downloaded"] == []
    assert emsa.downloads == []


def test_newer_version_is_downloaded_again_and_manifest_updated(tmp_path):
    emsa = FakeEmsa({2018: 5, 2019: 7})
    run(emsa, tmp_path)
    emsa.versions[2019] = 8
    emsa.downloads.clear()

    result = run(emsa, tmp_path)

    assert result["downloaded"] == [2019]
    assert result["skipped"] == [2018]
    assert emsa.downloads == ["/reporting-period-document/binary/2019/8"]
    assert read_manifest(tmp_path)["years"]["2019"]["version"] == 8


def test_missing_file_is_downloaded_even_if_manifest_has_it(tmp_path):
    emsa = FakeEmsa({2018: 5})
    run(emsa, tmp_path)
    (tmp_path / "2018.xlsx").unlink()

    result = run(emsa, tmp_path)

    assert result["downloaded"] == [2018]


def test_year_option_downloads_only_that_year(tmp_path):
    emsa = FakeEmsa({2018: 5, 2019: 7, 2020: 9})

    result = run(emsa, tmp_path, year=2019)

    assert result["downloaded"] == [2019]
    assert list(read_manifest(tmp_path)["years"]) == ["2019"]


def test_unknown_year_stops_with_a_message(tmp_path):
    with pytest.raises(SystemExit):
        run(FakeEmsa({2018: 5}), tmp_path, year=1999)


def test_server_error_is_retried_then_succeeds(tmp_path):
    emsa = FakeEmsa({2018: 5})
    emsa.fail_next = [503, 503]  # first two requests fail, the third works

    result = run(emsa, tmp_path)

    assert result["downloaded"] == [2018]


def test_not_an_xlsx_counts_as_failed_and_leaves_no_file(tmp_path):
    emsa = FakeEmsa({2018: 5})
    original = emsa.handler
    emsa.handler = lambda req: (
        httpx.Response(200, text="<html>maintenance</html>")
        if "/binary/" in req.url.path else original(req)
    )

    result = run(emsa, tmp_path)

    assert result["failed"] == [2018]
    assert not (tmp_path / "2018.xlsx").exists()
    assert not (tmp_path / "manifest.json").exists()
