# ShipCarbon

How much CO₂ do ships emit on voyages to and from Europe, and which ships emit the most?
ShipCarbon answers that with 8 years of official EU data, and predicts a ship's yearly fuel use with machine learning.

**Status:** website done, deploy next.

## What it does

1. **Downloads** the EU MRV emission reports for 2018–2025 from EMSA (THETIS-MRV) and records each file's version.
2. **Cleans** them into one row per report and one row per ship per year (109,711 reports), without double counting partial reports, and loads them into MongoDB.
3. **Analyses** the data in a Jupyter notebook. Key findings are in [docs/insights.md](docs/insights.md).
4. **Predicts** a ship's annual fuel (and CO₂) from its type, efficiency index, hours at sea, ice class, flag and year.
5. **Serves** statistics and predictions through a FastAPI backend.
6. **Shows** it all on a simple website: headline numbers, filters, charts, the top emitters, key findings and a prediction form.

## Key findings

- **Container ships** are 14% of the fleet but **34%** of the CO₂ (2025).
- **One ship in ten** produces **48%** of the CO₂.
- **COVID** cut emissions by 12% in 2020; they only passed the 2019 level again in 2024.
- The **2025 increase** comes from newly covered ship types. Ships reported in both 2024 and 2025 emitted **6.3% less**.
- **LNG carriers** are 2% of ships but about **60% of the methane**.

## Machine learning model

Model: scikit-learn `HistGradientBoostingRegressor` trained on 101,428 ship-years from 25,249 ships.
No input is computed from fuel or CO₂, so the model cannot copy the answer.

| Test | Simple baseline error | Model error | R² |
|---|---|---|---|
| Ships the model never saw | 1,405 t | **755 t** (−46%) | 0.86 |
| Train 2018–2023, test 2024–2025 | 1,458 t | **736 t** (−50%) | 0.88 |

The main limitation is that the public data has no ship size, so a small and a very large ship of the same type look alike.

## Tech stack

Python · httpx · pandas · MongoDB · Jupyter · Matplotlib · scikit-learn · FastAPI · pytest · HTML/CSS/JavaScript · Chart.js

## Project structure

```text
pipeline/    download, clean and load the EMSA data into MongoDB
notebooks/   exploratory data analysis (01_eda.ipynb)
ml/          features, training and single-ship prediction
backend/     FastAPI app and the trained model
frontend/    the website (plain HTML, CSS and JavaScript, no build step)
docs/        insights and charts
tests/       automated tests (pytest)
```

## How to run

Needs Python 3.11+ and MongoDB running locally (`mongodb://localhost:27017`).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env

python -m pipeline.download     # download EMSA files to data/raw
python -m pipeline.clean        # build data/processed
python -m pipeline.load_mongo   # load into MongoDB (db: shipcarbon)
python -m ml.train              # train the model

uvicorn backend.app.main:app --reload
```

Then open the website at **http://127.0.0.1:8000**. The API documentation is at http://127.0.0.1:8000/docs.

## API endpoints

| Method | Endpoint | What it returns |
|---|---|---|
| GET | `/api/health` | API, database and model status |
| GET | `/api/meta` | Years, ship types and flags for filters |
| GET | `/api/kpis` | Headline totals for the selected filters |
| GET | `/api/co2-by-type` | CO₂ per ship type |
| GET | `/api/co2-by-year` | CO₂ per year |
| GET | `/api/top-emitters` | Ship-years with the most CO₂ |
| POST | `/api/predict` | Predicted annual fuel and CO₂ for one ship |

## Tests

```bash
pytest
```

## Data source

EU MRV data published by the European Maritime Safety Agency (EMSA), THETIS-MRV public emission reports.
Raw files are not included in this repository; `python -m pipeline.download` fetches them.
This is a personal learning project and is not affiliated with EMSA.

## Author

Hari P · [GitHub](https://github.com/nehemiah2406) · [LinkedIn](https://www.linkedin.com/in/nehemiah-hari)
