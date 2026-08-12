# EcoHome Energy Advisor

EcoHome is a LangGraph energy-advisor project for households with rooftop solar,
an electric vehicle, controllable HVAC, flexible appliances, and optional energy
storage. It combines forecasts, device-specific time-of-use prices, household
history, and retrieved energy guidance to recommend operating windows.

## Environment

- Python 3.13.5
- LangChain 0.3.x
- LangGraph 0.5–0.6
- ChromaDB
- SQLAlchemy with SQLite
- Pydantic 2
- Open-Meteo geocoding and forecast APIs
- OpenAI-compatible chat and embedding endpoints

Install the declared dependencies from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the environment template and add the course API key:

```bash
cp .env.example .env
```

Required variables are `VOCAREUM_API_KEY`, `OPENAI_BASE_URL`,
`GEOCODING_URL`, and `FORECAST_URL`. `EVALUATOR_MODEL` defaults to `gpt-4o`.
The Energy Advisor itself defaults to `gpt-4o-mini`.

## Project structure

```text
ecohome_solution/
├── models/
│   ├── energy.py              # SQLAlchemy models and database access
│   ├── evaluate.py            # Structured LLM-judge response schema
│   └── weather.py             # Validated geocoding and forecast contracts
├── data/documents/            # Energy-saving knowledge articles
├── tests/test_weather_tool.py # Weather transformation regressions
├── agent.py                   # Explicit LangGraph schema, nodes, and edges
├── tools.py                   # Weather, pricing, database, RAG, and savings tools
├── 01_db_setup.ipynb          # Database creation and sample data
├── 02_rag_setup.ipynb         # Document loading and Chroma vector-store setup
├── 03_run_and_evaluate.ipynb  # Agent scenarios, metrics, and report
├── .env.example               # Reproducible non-secret configuration
└── requirements.txt
```

## Run order

Run the notebooks from the `ecohome_solution` working directory:

1. `01_db_setup.ipynb`
2. `02_rag_setup.ipynb`
3. `03_run_and_evaluate.ipynb`

The first notebook generates approximately 30 days of energy-usage and solar
records. The second discovers every `.txt` article in `data/documents`, splits
the articles, and creates `data/vectorstore`. The third runs at least ten agent
scenarios and evaluates their saved message traces.

The database and vector store are generated artifacts and are ignored by Git.
Rebuild them by rerunning notebooks 1 and 2 from a clean generated-data state.

## LangGraph workflow

`agent.py` declares an `AgentState` message schema and two nodes:

```text
START → agent → tools → agent
          └────────────→ END
```

The agent node invokes the tool-enabled chat model. Conditional routing sends a
tool request to `ToolNode`; a final model response ends the graph. `invoke()`
accepts the required `question` and optional trusted `context`, and returns the
complete message trace used by the evaluation notebook.

## Tools

- `get_weather_forecast` resolves a human-readable place and retrieves validated,
  timezone-aware Open-Meteo data including solar irradiance.
- `get_electricity_prices` returns the project's static, device-specific
  time-of-use tariff.
- `query_energy_usage` and `query_solar_generation` retrieve historical records.
- `get_recent_energy_summary` returns recent totals and device breakdowns.
- `search_energy_tips` searches the persisted Chroma knowledge base.
- `calculate_energy_savings` calculates savings for an explicit measurement
  period and annualizes from that period.

## Evaluation

The notebook records each question, full response trace, expected tools, and
expected response requirements in `test_results`. Response quality is evaluated
by a configurable stronger LLM using a Pydantic schema for accuracy, relevance,
completeness, usefulness, and detailed feedback. Tool appropriateness and tool
completeness are computed deterministically from executed `ToolMessage` objects.

`generate_evaluation_report()` aggregates the results; the separate
`display_evaluation_report()` function renders summary scores, strengths,
weaknesses, recommendations, and per-test measurements.

## Validation

Run the weather regression suite with the project environment:

```bash
python -m unittest tests.test_weather_tool -q
```

All notebooks should be restarted and run from top to bottom before submission,
then saved with their final outputs.
