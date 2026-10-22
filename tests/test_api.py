"""Tests for the FastAPI app. MongoDB is replaced by mongomock (an in-memory fake), so real data is never touched."""

import mongomock
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.db import get_db
from backend.app.routes import stats

CRUISE = "Passenger ship (Cruise Passenger ship)"  # EMSA's 2025 label, merged into "Passenger ship"


def ship_year(imo, year, ship_type, flag, co2, co2eq=0.0, name=None):
    return {"_id": f"{imo}-{year}", "imo": imo, "year": year, "name": name or f"SHIP {imo}", "ship_type": ship_type,
            "flag": flag, "co2_t": co2, "co2eq_t": co2eq, "fuel_t": co2 / 3.0}


SHIP_YEARS = [
    ship_year("1", 2023, "Container ship", "Liberia", 1000),
    ship_year("1", 2024, "Container ship", "Liberia", 1200, co2eq=1210),
    ship_year("2", 2024, "Bulk carrier", "Malta", 300, co2eq=301),
    ship_year("3", 2024, "Passenger ship", "Malta", 500, co2eq=505),
    ship_year("3", 2025, CRUISE, "Malta", 600, co2eq=606),       # same ship, relabelled in 2025
    ship_year("4", 2025, CRUISE, "Malta", 700, co2eq=707),
    ship_year("5", 2025, "LNG carrier", "Unknown", 800, co2eq=900),
]
TOTAL_CO2 = 5100


@pytest.fixture
def fake_db():
    db = mongomock.MongoClient()["shipcarbon_test"]
    db["ship_years"].insert_many([dict(d) for d in SHIP_YEARS])
    return db


@pytest.fixture
def client(fake_db, monkeypatch):
    monkeypatch.setenv("WARM_CACHE", "0")
    stats.clear_cache()
    main.app.dependency_overrides[get_db] = lambda: fake_db
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()
    stats.clear_cache()


# ---------- health and meta ----------

def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["database"] is True and body["model"] is True
    assert body["data_versions"]["2025"] > 0


def test_health_when_database_is_down(client):
    class DownDb:
        def command(self, *args):
            raise ConnectionError("no server")

    main.app.dependency_overrides[get_db] = lambda: DownDb()
    body = client.get("/api/health").json()
    assert body["database"] is False and body["status"] == "degraded"


def test_meta(client):
    body = client.get("/api/meta").json()
    assert body["years"] == [2023, 2024, 2025]
    assert body["ship_types"] == ["Bulk carrier", "Container ship", "LNG carrier", "Passenger ship"]
    assert body["flags"] == ["Liberia", "Malta", "Unknown"]  # Unknown always last
    groups = {g["group"]: g["ship_types"] for g in body["ship_groups"]}
    assert list(groups) == ["Container", "Tanker", "Bulk", "LNG", "Ro-pax", "Other"]
    assert groups["Container"] == ["Container ship"] and groups["Other"] == ["Passenger ship"]


# ---------- statistics ----------

def test_kpis_all_years(client):
    body = client.get("/api/kpis").json()
    assert body["co2_t"] == TOTAL_CO2
    assert body["ships"] == 5 and body["ship_years"] == 7
    assert body["co2eq_t"] is None  # includes 2023, before CO2eq was published
    assert body["co2_per_ship_t"] == pytest.approx(TOTAL_CO2 / 7, abs=0.1)


@pytest.mark.parametrize("query, co2, ships, ship_years", [
    ("year=2025", 2100, 3, 3),
    ("ship_type=Passenger ship", 1800, 2, 3),            # both labels counted
    (f"ship_type={CRUISE}", 1800, 2, 3),                 # the 2025 label means the same type
    ("flag=Malta", 2100, 3, 4),
    ("year=2024&flag=Malta&ship_type=Bulk carrier", 300, 1, 1),
    ("flag=Nowhere", 0, 0, 0),
])
def test_kpis_filters(client, query, co2, ships, ship_years):
    body = client.get(f"/api/kpis?{query}").json()
    assert (body["co2_t"], body["ships"], body["ship_years"]) == (co2, ships, ship_years)


def test_kpis_co2eq_from_2024(client):
    assert client.get("/api/kpis?year=2025").json()["co2eq_t"] == 2213


def test_co2_by_type(client):
    rows = client.get("/api/co2-by-type").json()
    assert [r["ship_type"] for r in rows] == ["Container ship", "Passenger ship", "LNG carrier", "Bulk carrier"]
    passenger = rows[1]
    assert passenger["co2_t"] == 1800 and passenger["ships"] == 2 and passenger["ship_years"] == 3
    assert sum(r["share_pct"] for r in rows) == pytest.approx(100, abs=0.2)
    assert [r["ship_type"] for r in client.get("/api/co2-by-type?year=2025").json()] == ["Passenger ship",
                                                                                       "LNG carrier"]


def test_co2_by_year(client):
    rows = client.get("/api/co2-by-year").json()
    assert [(r["year"], r["co2_t"], r["ships"]) for r in rows] == [(2023, 1000, 1), (2024, 2000, 3), (2025, 2100, 3)]
    assert rows[0]["co2eq_t"] is None and rows[1]["co2eq_t"] == 2016
    assert [r["year"] for r in client.get("/api/co2-by-year?flag=Liberia").json()] == [2023, 2024]


def test_top_emitters(client):
    rows = client.get("/api/top-emitters").json()
    assert [r["co2_t"] for r in rows] == [1200, 1000, 800, 700, 600, 500, 300]
    assert rows[0] == {"rank": 1, "imo": "1", "name": "SHIP 1", "year": 2024, "ship_type": "Container ship",
                       "flag": "Liberia", "co2_t": 1200, "fuel_t": 400}
    rows = client.get("/api/top-emitters?year=2025&limit=2").json()
    assert [(r["imo"], r["ship_type"]) for r in rows] == [("5", "LNG carrier"), ("4", "Passenger ship")]


@pytest.mark.parametrize("url", [
    "/api/top-emitters?limit=0", "/api/top-emitters?limit=51", "/api/kpis?year=abc", "/api/co2-by-type?year=1990",
])
def test_bad_filters_are_rejected(client, url):
    assert client.get(url).status_code == 422


def test_warm_cache_fills_unfiltered_year_and_type_views(client, fake_db):
    stats.warm_cache(fake_db)
    assert len(stats._cache) == 4 * (1 + 3 + 4)  # 4 endpoints x (no filter + 3 years + 4 ship types)
    assert client.get("/api/kpis").json()["co2_t"] == TOTAL_CO2  # answered from the cache, same result


# ---------- prediction ----------

def test_predict(client):
    r = client.post("/api/predict", json={"ship_type": "Bulk carrier", "time_at_sea_h": 5000})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fuel_t", "co2_t", "smoke"}
    assert 500 < body["fuel_t"] < 20_000 and body["co2_t"] > body["fuel_t"] and 0 <= body["smoke"] <= 1


def test_predict_full_input_and_cruise_label(client):
    full = {"ship_type": CRUISE, "time_at_sea_h": 5000, "tech_eff_type": "EIV", "tech_eff_value": 12.5,
            "flag": "Malta", "ice_class": True, "year": 2025}
    assert client.post("/api/predict", json=full).status_code == 200


@pytest.mark.parametrize("payload, field", [
    ({"time_at_sea_h": 5000}, "ship_type"),                                              # missing
    ({"ship_type": "Submarine", "time_at_sea_h": 5000}, "ship_type"),                    # unknown type
    ({"ship_type": "Bulk carrier", "time_at_sea_h": -1}, "time_at_sea_h"),               # negative
    ({"ship_type": "Bulk carrier", "time_at_sea_h": 9000}, "time_at_sea_h"),             # more than a year
    ({"ship_type": "Bulk carrier", "time_at_sea_h": "lots"}, "time_at_sea_h"),           # not a number
    ({"ship_type": "Bulk carrier", "time_at_sea_h": 5000, "tech_eff_type": "XYZ"}, "tech_eff_type"),
    ({"ship_type": "Bulk carrier", "time_at_sea_h": 5000, "tech_eff_value": 0}, "tech_eff_value"),
    ({"ship_type": "Bulk carrier", "time_at_sea_h": 5000, "year": 1999}, "year"),
    ({"ship_type": "Bulk carrier", "time_at_sea_h": 5000, "co2_t": 10}, "co2_t"),        # leakage field
])
def test_predict_bad_input_gives_clear_422(client, payload, field):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422
    assert field in [err["loc"][-1] for err in r.json()["detail"]]


# ---------- CORS ----------

def preflight(client, origin):
    return client.options("/api/kpis", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})


def test_cors_allows_local_frontend_only(client):
    assert preflight(client, "http://localhost:5173").headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in preflight(client, "https://evil.example").headers


def test_frontend_origins_from_env(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGINS", " https://shipcarbon.vercel.app/ , https://www.example.org")
    assert main.allowed_origins() == ["http://localhost:5173", "https://shipcarbon.vercel.app",
                                      "https://www.example.org"]


# ---------- website ----------

def test_home_page_is_the_website(client):
    res = client.get("/")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
    assert "<title>ShipCarbon</title>" in res.text and 'src="app.js"' in res.text
    assert client.get("/styles.css").status_code == 200 and client.get("/app.js").status_code == 200


def test_docs_and_api_still_work_next_to_the_website(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/api/health").json()["database"] is True
    assert client.get("/api/does-not-exist").status_code == 404
