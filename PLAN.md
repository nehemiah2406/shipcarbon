# ShipCarbon — Project Plan

An interactive website about CO₂ emissions from large ships calling at EU/EEA ports, built on the public
EU MRV (THETIS-MRV) data published by EMSA. It includes a 3D ocean hero, exploratory insights, and an ML model
that predicts a ship's annual fuel consumption.

---

## 1. Research: where the data comes from

**Source:** EMSA THETIS-MRV public portal — <https://mrv.emsa.europa.eu/#public/emission-report>.
The portal is a JavaScript (ExtJS) app. There is no HTML table to scrape: the page calls a JSON API, and we
call the same API directly.

| What | Endpoint (base `https://mrv.emsa.europa.eu/api/public-emission-report`) |
|---|---|
| Available years | `GET /reporting-periods` → `[{"value":2025}, … {"value":2018}]` |
| Latest file per year (+ version) | `GET /downloadable-files` → `{reportingPeriod, version, generationDate, fileName}` |
| Download one year (.xlsx) | `GET /reporting-period-document/binary/{year}/{version}` |
| Ship type list | `GET /ship-types` |

- **Years available (checked 2026-10-15):** 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.
- **Versions change often.** EMSA republishes a year when reports are corrected (2024 was at v252). The download
  script must always ask `/downloadable-files` for the current version, and must save that version number with the
  file so results can be reproduced.
- **File size:** 4 MB (2018) to 12 MB (2025). About 12k–17k ships per year.
- **Who is covered:** ships > 5,000 GT calling at EU/EEA ports. 2025 has noticeably more rows (17.4k vs 14.2k).
  The likely reason is the scope extension in the revised MRV Regulation (offshore ships and smaller general cargo
  ships). **Verify this in EDA** before comparing 2025 with earlier years.

### File layout (verified by opening 2018, 2021, 2024 and 2025)

- Rows 1–2 are grouped header bands ("Ship", "DoC", "Verifier", …). **Row 3 holds the real column names.** Data
  starts on row 4.
- **2018–2023:** one sheet named after the year, **62 columns**.
- **2024 onward:** two sheets, `"{year} Full ERs"` and `"{year} Partial ERs"`, **113 columns**. A *partial* report
  covers part of a year, for example when a ship changed owner. The same IMO number can appear in both sheets.
- No duplicate IMO numbers within a sheet.

### Key columns we will use

| Clean name | Source column(s) | Notes |
|---|---|---|
| `imo` | IMO Number | String, unique ship ID |
| `name` | Name | |
| `ship_type` | Ship type | 17 values, e.g. "Bulk carrier", "Container ship", "LNG carrier", "Ro-pax ship" |
| `year` | Reporting Period | |
| `tech_eff_type`, `tech_eff_value` | Technical efficiency | Text like `"EEDI (4.12 gCO₂/t·nm)"` → split into `EEDI` + `4.12`. Types: EEXI, EEDI, EIV, Not Applicable |
| `port_of_registry` | Port of Registry | Mixed case ("Panama" / "PANAMA"); ~2.7% blank |
| `flag` | *derived* | See "Flag state" below |
| `ice_class` | Ice Class | Mostly blank |
| `fuel_t` | Total fuel consumption [m tonnes] | **ML target** |
| `co2_t` | Total CO₂ emissions [m tonnes] | Main metric for charts |
| `co2_between_ms_t`, `co2_from_ms_t`, `co2_to_ms_t`, `co2_at_berth_t` | CO₂ … under a MS jurisdiction | Split by voyage type |
| `time_at_sea_h` | **renamed across years:** "Annual Total time spent at sea [hours]" (2018), "Annual Time spent at sea [hours]" (2021), "Time spent at sea [hours]" (2024+) | Needs a per-era column map |
| `co2_per_nm_kg` | "Annual average CO₂ emissions per distance" (≤2023) / "CO₂ emissions per distance" (2024+) | Renamed too |
| `ch4_t`, `n2o_t`, `co2eq_t` | Total CH₄ / N₂O / CO₂eq emissions | **Only from 2024** |
| `report_kind` | *derived* | `full` or `partial` (from sheet name; all pre-2024 rows = `full`) |

### Data facts and gotchas found during research

1. **No flag-state column.** EMSA has switched flag search off: `/flags` returns *"search with flag is not enabled"*.
   → We derive `flag` from `Port of Registry` using a small hand-checked lookup file
   (`data/reference/port_to_flag.csv`), e.g. Monrovia → Liberia, Majuro → Marshall Islands, Valletta → Malta,
   Madeira → Portugal. About 500 distinct ports after upper-casing; **the top 200 cover 96% of ships.** Unmapped
   or blank → `Unknown`. The website will label this "flag (from port of registry)".
2. **No ship size (DWT/GT) column.** The model has to work without size. The technical-efficiency value and time at
   sea act as partial stand-ins. This limitation goes in the README.
3. **CO₂ ≈ fuel × 3.14.** The CO₂/fuel ratio is 3.01–3.21 (5th–95th percentile) in every year. So:
   - CO₂ (and anything derived from fuel, such as per-distance metrics) must **never** be a model feature
     (target leakage).
   - Predicted CO₂ for the 3D smoke = predicted fuel × the ship type's median CO₂/fuel ratio.
4. **Zero-fuel rows:** 4–12% of ships report 0 fuel (2025: 2,061 rows). Keep them in the totals, but exclude them
   from model training.
5. **Numbers are already floats** in the xlsx. Some ratio columns can contain text such as "Division by zero!",
   so convert them with `errors="coerce"`.
6. **Totals sanity check** (Full ERs): 2018 ≈ 145 Mt CO₂, 2021 ≈ 127 Mt, 2024 ≈ 147 Mt, 2025 ≈ 152 Mt.

### Licence / attribution
Data © EMSA / European Commission, published under the MRV Regulation (EU) 2015/757. Before publishing, check the
THETIS-MRV terms of use. In the README and the site footer, credit "Source: EMSA THETIS-MRV" with the
download date and file versions.

---

## 2. Tech stack and libraries

**Python (backend + data)**

| Library | Why | New for you? |
|---|---|---|
| `httpx` | Download files from the EMSA API | known |
| ~~beautifulsoup4~~ | **Not needed.** The portal has no HTML to parse: the data comes from a JSON API and xlsx files. Left out to keep the install small. | — |
| `pandas` | Read, clean and aggregate tables | new, needed: core tool for tabular cleaning |
| `openpyxl` | The engine pandas uses to read `.xlsx` files | new, required by `pandas.read_excel` |
| `pymongo` | Talk to MongoDB (local + Atlas) | known |
| `python-dotenv` | Load the MongoDB connection string from `.env` so secrets never go into git | new, tiny |
| `scikit-learn` | ML model (HistGradientBoostingRegressor), pipelines, cross-validation; also brings `joblib` to save the model | new, needed for Goal 3 |
| `matplotlib` | Static charts for the EDA notebook and README images | new, needed for Goal 2 |
| `ipykernel` | Lets VS Code run Jupyter notebooks inside `.venv` | new, needed for notebooks |
| `fastapi` + `uvicorn[standard]` | API server (requested) | requested |
| `pytest` | Small automated tests for cleaning and the API | new, small |

**Frontend (Phase 5, installed with npm later)**

| Package | Why |
|---|---|
| `react`, `vite` | requested |
| `three`, `@react-three/fiber` | requested: 3D hero |
| `@react-three/drei` | Ready-made `OrbitControls` (drag to orbit), `useTexture`, `PerformanceMonitor`. Saves writing a lot of camera code |
| `recharts` | Bar and line charts as React components, styled with our colours |
| Fonts | Google Fonts `<link>` (Archivo, IBM Plex Sans, IBM Plex Mono). No package needed |

The ocean (moving Gerstner-wave shader with sunset sun glint), the sky gradient and the smoke particles will be
**custom GLSL/Three code**, with no extra library. The five ships (container, tanker, bulk, LNG, ro-pax) will be
**low-poly models built from Three.js primitives**. That avoids licensing issues and keeps them light. They can be
swapped for CC0 glTF models later.

**Hosting (all free tiers)**
- Database: **MongoDB Atlas M0** (512 MB). We store only about 30 clean columns, which comes to roughly
  110k documents and well under the limit.
- API: **Render** free web service (FastAPI). It sleeps after about 15 minutes idle, so the first request can take
  30–60 s. The frontend shows a "waking up the server…" state.
- Frontend: **Vercel** (or Netlify): static Vite build.

---

## 3. Folder structure

```
shipcarbon/
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                 # MONGODB_URI=... (template, no secrets)
├── data/
│   ├── raw/                     # downloaded xlsx + manifest.json   (git-ignored)
│   ├── processed/               # cleaned CSV                        (git-ignored)
│   └── reference/
│       └── port_to_flag.csv     # hand-checked lookup (committed)
├── pipeline/
│   ├── download.py              # httpx → data/raw/{year}.xlsx
│   ├── columns.py               # per-era column-name map → clean names
│   ├── clean.py                 # raw xlsx → data/processed/emissions.csv
│   └── load_mongo.py            # CSV → MongoDB (local or Atlas)
├── notebooks/
│   └── 01_eda.ipynb
├── ml/
│   ├── features.py
│   └── train.py                 # → backend/app/model/fuel_model.joblib + metrics.json
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + CORS
│   │   ├── db.py
│   │   ├── routes/              # stats.py, predict.py
│   │   └── model/
│   └── requirements.txt         # slim list for Render (no notebook libs)
├── frontend/                    # Vite + React + R3F
│   └── src/
│       ├── three/               # Ocean, Sky, Ship, Smoke, Hero
│       ├── components/          # Filters, KpiCards, charts, PredictForm, TopEmitters
│       └── styles/tokens.css    # colour + font tokens
├── tests/
└── docs/
    ├── insights.md
    └── img/
```

---

## 4. Phases

### Phase 0 — Setup ✅
git repo, `.venv`, `requirements.txt`, `.gitignore`, this plan.
**Done when:** `git status` is clean after the first commit; `.venv` activates in VS Code;
`pip install -r requirements.txt` finishes without errors; `.env`, `.venv/` and `data/raw/` are ignored.

### Phase 1 — Download all years ✅ (2026-10-15)
`pipeline/download.py`: call `/downloadable-files`, then download every year with httpx into
`data/raw/{year}.xlsx`. Write `data/raw/manifest.json` (year, version, generation date, file size, download time).
Skip a year if the same version is already on disk. Retry with a polite delay; set a clear User-Agent.
**Done when:** all 8 years (2018–2025) are in `data/raw/` and each opens with pandas; the manifest lists their
versions; a second run downloads nothing.
**Result:** 8 files (46 MB) downloaded. All 10 sheets open with pandas: 62 columns for 2018–2023, 113 for 2024–2025.
A second run skipped all 8 years. 8/8 tests in `tests/test_download.py` pass.

### Phase 2 — Clean and store ✅ (2026-10-15)
`pipeline/columns.py` maps each era's column names to the clean names in section 1. `pipeline/clean.py`:
- reads both sheets for 2024+ and adds `report_kind`
- parses technical efficiency into type + value
- normalises ports (strip, upper case) and joins `port_to_flag.csv` → `flag`
- coerces numbers, flags zero-fuel rows, and adds `ship_group` (Container / Tanker / Bulk / LNG / Ro-pax / Other)
  to drive the 3D ship buttons

It writes `data/processed/emissions.csv`. `pipeline/load_mongo.py` then loads it into a `reports` collection, with
indexes on `year`, `ship_type`, `flag` and `imo`. The same script works for local MongoDB and Atlas; only
`MONGODB_URI` in `.env` changes.
**Done when:** row counts per year/sheet equal the raw files; total CO₂ per year matches the raw totals within
0.01%; ≥ 95% of ships get a real flag (not `Unknown`); `pytest tests/test_clean.py` passes; MongoDB has the
documents and indexes.

**Result:**
- 109,711 reports (= raw) and 107,894 ship-years, in both CSV and MongoDB (`reports`, `ship_years`).
- CO₂ per year equals the raw files (0.0000% difference).
- Flag known for 96.0–98.2% of ship-years per year (97.6% overall). A blank port borrows the same ship's nearest
  other year (`flag_source = port_other_year`, 4,385 ship-years).
- Loading twice changes 0 documents.
- 36 tests in `tests/test_clean.py` pass.

**Decision — partial reports overlap full reports.** For a ship that has both in one year, the full report already
covers the partial periods. Evidence: in 2024, 190 of 726 such ships have identical numbers, 255 have more sea hours
than the non-partial days allow, and only 2 have full < partial. So:
- every report is kept with `report_kind`
- `counts_in_total` is true for full reports and for partials of ships without a full report
- yearly totals and `ship_years` use only `counts_in_total` rows

Adding all partials would double-count 4.4 Mt (2024) and 5.1 Mt (2025).

### Phase 3 — Exploratory analysis ✅ (2026-10-15)
`notebooks/01_eda.ipynb`. Questions: total CO₂ per year (and whether 2025 is comparable); CO₂ by ship type (total
vs per ship); which flags carry the most emissions; how concentrated emissions are (share from the top 10% of
ships); CO₂ split between intra-EU voyages, voyages into/out of the EU, and at berth; efficiency (CO₂ per n mile) by
type; CH₄/N₂O for LNG carriers since 2024.
**Done when:** the notebook runs top to bottom on a fresh kernel; `docs/insights.md` has 5–7 short insights, each
with a number and a chart in `docs/img/`; every number can be traced back to a notebook cell.

**Result:**
- The notebook runs in a fresh kernel via nbclient: 9 code cells, 0 errors, about 10 s.
- 7 charts are in `docs/img/` and 7 insights in `docs/insights.md`.

Findings that matter for later phases:
- **2025 is not like-for-like.** About 2,200 newly covered general cargo and offshore ships add 6.6 Mt. The ships
  reported in both 2024 and 2025 emitted 6.3% less.
- **EMSA relabelled passenger ships in 2025** ("Passenger ship (Cruise Passenger ship)"). The labels are merged in
  the analysis; Phase 4 and the API must merge them too.
- **CO₂eq uses the IPCC factors** CH₄ × 28 and N₂O × 265.

### Phase 4 — ML: predict annual fuel consumption ✅ (2026-10-15)
- **Target:** `log(fuel_t)`, on full reports with fuel > 0.
- **Features:** ship type, technical-efficiency type + value, time at sea, ice class (yes/no), flag (top 15, else
  "Other"), year.
- **Excluded on purpose:** CO₂ / CH₄ / CO₂eq, per-distance and per-transport-work metrics, laden fuel. These are
  all computed from fuel, so using them would be leakage.
- **Validation:** GroupKFold by IMO, so the same ship never appears in both train and test, plus a time check
  (train ≤ 2023, test 2024–2025).
- **Baseline:** median fuel per hour at sea × hours, per ship type. **Model:** scikit-learn
  `HistGradientBoostingRegressor` in a Pipeline.
- **Output:** `backend/app/model/fuel_model.joblib`, plus `metrics.json` (MAE in tonnes, MAPE, R²) and
  `co2_factor_by_type.json`.

**Done when:** the model beats the baseline MAE by a clear margin on held-out ships; metrics are saved and quoted
in the README; a feature-importance chart is in `docs/img/`; predicting a single ship from a Python dict works.

**Result:** `python -m ml.train`; `ml/predict.py` → `predict_one(dict)` returns fuel, CO₂ and smoke (0–1).
- 101,428 ship-years from 25,249 ships. Best settings: learning rate 0.05, 127 leaves, min 50 per leaf.

| Validation | Baseline MAE | Model MAE | Model median % error | Model R² |
|---|---|---|---|---|
| GroupKFold (5) by IMO | 1,405 t | **755 t** (−46%) | 14.9% | 0.86 |
| Train 2018–23, test 2024–25 | 1,458 t | **736 t** (−50%) | 15.7% | 0.88 |

- Drivers (permutation importance): time at sea and ship type, then the efficiency index value; year adds nothing.
- Weakest types: offshore (53% median error), Ro-pax and passenger (24–27%) — no ship size in the data.
- CO₂ = fuel × median CO₂/fuel of the ship type (3.13–3.21; LNG carrier 2.79).
- `predict_one` rejects unknown ship types, unknown fields and > 8,784 h; unseen flags become "Other".
- 14 tests in `tests/test_ml.py` pass (58 in total). The README is written in Phase 7 and quotes the table above.

### Phase 5a — API (FastAPI)
Endpoints: `GET /api/health`, `GET /api/meta` (years, ship types, flags), `GET /api/kpis`,
`GET /api/co2-by-type`, `GET /api/co2-by-year`, `GET /api/top-emitters`. The data endpoints take the filters
`year`, `ship_type` and `flag`. `POST /api/predict` returns fuel, CO₂ and a smoke-scale value. CORS is limited to the
frontend's URL.
**Done when:** `/docs` shows every endpoint; `pytest tests/test_api.py` passes; each endpoint answers in < 300 ms
locally.

### Phase 5b — Website (React + Vite + react-three-fiber)
- Design tokens are copied exactly from the mockup: `#0a1220` background, `#0f1a2b` panels, `#20324a` lines,
  `#e9eef4` text, `#9aabbf` muted, amber `#ffb547` (CO₂ data and buttons only), sea-glass `#8fd0c8`.
  Fonts: Archivo, IBM Plex Sans and IBM Plex Mono.
- Hero: full-width canvas with the ocean shader, sky gradient, a ship that bobs on the waves, a smoke plume, the
  5 ship-type buttons, and OrbitControls with limited angles.
- Below the hero: filters sidebar, which becomes a drawer on phones; 4 KPI cards; bar chart; line chart; prediction
  form (it updates the hero ship type and smoke size); top-emitters table, which scrolls sideways on phones.
- Performance: cap the device pixel ratio at 1.5; use fewer wave segments and particles on mobile; with
  `prefers-reduced-motion`, show calm water and no orbit animation.

**Done when:** it matches the mockup side by side; it works at 375 px width with no horizontal page scroll; the hero
holds ~60 fps on a laptop and stays smooth on a mid-range phone; the Lighthouse mobile score is ≥ 80 for
performance and ≥ 90 for accessibility; filters update all panels.

### Phase 6 — Deploy (free)
Atlas M0 cluster, then `load_mongo.py` against Atlas; FastAPI on Render with `MONGODB_URI` set as an environment
variable; frontend on Vercel with `VITE_API_URL`.
**Done when:** the public URL loads the full site from a fresh browser; a prediction works end to end; no secrets
appear in the git history; the cold-start state shows correctly.

### Phase 7 — README and polish
README: short pitch, hero GIF, live-demo link, architecture diagram, data source and attribution, how flag is
derived, key insights, model metrics and limitations (no ship size, derived flag, 2025 scope change), how to run
locally on Windows, project structure, licence.
**Done when:** a stranger can clone the repo and run it locally from the README alone; every link works; the repo
has a description, topics and a screenshot.

---

## 5. Decisions
- **Phase 2:** use the local MongoDB (`mongodb://localhost:27017`, already installed) for development.
  Atlas is used only from Phase 6 (deploy).
- **Phase 5b:** start with procedural low-poly ships. CC0 glTF models can come later.
