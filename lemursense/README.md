# 🐒 LemurSense

**A biomimetic, explainable AI system for IT anomaly monitoring**, modeled on the vigilance and foraging behavior of ring-tailed lemurs.

LemurSense doesn't watch every metric on every host with equal intensity all the time. Like a lemur troop, it allocates limited "attention" adaptively — watching more closely where risk is rising, relaxing where things are calm, and reacting as a group when one member sounds an alarm.

---

## Why lemurs?

Ring-tailed lemurs live in troops with a few behaviors that map surprisingly well onto system monitoring:

| Lemur behavior | IT monitoring analog | Where it lives in the code |
|---|---|---|
| **Sentinel rotation** — troop members take turns watching for threats | Not all metrics are polled at max frequency; attention rotates to whichever metric looks riskiest | `lemur_sensor.py` → `SentinelState`, `is_sentinel` |
| **Alarm calls** — one lemur spotting danger alerts the whole troop | An anomaly on one metric raises attention on sibling metrics on the same host | `lemur_sensor.py` → alarm propagation step in `LemurSensor.update()` |
| **Adaptive foraging** — more energy spent searching when conditions are volatile, less when calm | Polling interval shrinks when attention is high, grows when calm (energy conservation) | `SentinelState.polling_interval()` |
| **Territory / scent-marking** — each lemur troop learns its home range | Each host gets its own learned baseline ("territory"); deviations from *that host's* normal are what matters, not a global threshold | `anomaly_detector.py` → one `HostAnomalyModel` per host |
| **Circadian rhythm** — lemurs are most active at dawn/dusk | Attention baseline is modulated by time-of-day, since "normal" volatility differs between business hours and overnight | `lemur_sensor.py` → `_circadian_factor()` |

---

## Architecture

```
                 ┌────────────────────┐
                 │  data_generator.py │   synthetic VM/host metrics
                 │  (circadian + )    │   (cpu, mem, disk I/O, net I/O)
                 │  injected anomalies│
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ anomaly_detector.py│   per-host Isolation Forest
                 │  (per-host models) │   + rolling volatility features
                 └─────────┬──────────┘
                           │ anomaly scores
                 ┌─────────┴──────────┐
                 ▼                    ▼
       ┌──────────────────┐   ┌──────────────────┐
       │  lemur_sensor.py │   │  explainer.py     │
       │  adaptive         │   │  SHAP explanations│
       │  attention/alarm  │   │  per anomaly       │
       └─────────┬─────────┘   └─────────┬─────────┘
                 └───────────┬───────────┘
                             ▼
                     ┌───────────────┐
                     │   app.py      │  FastAPI: /simulate/tick,
                     │  (FastAPI)    │  /lemur/state, /explain, ...
                     └───────┬───────┘
                             ▼
                   ┌───────────────────┐
                   │ streamlit_app.py  │  dashboard: metric charts,
                   │  (Streamlit)      │  troop attention panel, SHAP
                   └───────────────────┘
```

## Project structure

```
lemursense/
├── backend/
│   ├── data_generator.py    # synthetic metric generation
│   ├── anomaly_detector.py  # Isolation Forest per host + feature engineering
│   ├── lemur_sensor.py      # biomimicry core: attention, alarm calls, foraging
│   ├── explainer.py         # SHAP explainability wrapper
│   └── app.py                # FastAPI application
├── frontend/
│   └── streamlit_app.py      # dashboard UI
├── data/                      # generated synthetic dataset (csv)
├── requirements.txt
├── run.sh                     # launches backend + frontend together
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Option 1 — one command:

```bash
./run.sh
```

Option 2 — run each service manually (two terminals):

```bash
# terminal 1
cd backend
uvicorn app:app --reload --port 8000

# terminal 2
cd frontend
streamlit run streamlit_app.py
```

Then open the Streamlit URL (usually `http://localhost:8501`).

## Using the dashboard

1. Pick a host in the sidebar.
2. Click **Advance Time** to move the simulated clock forward — this reveals new data points, scores them, and updates the lemur troop's attention.
3. Watch the **Lemur Troop Attention** panel: the 👁️ icon marks the current *sentinel* metric (the one getting the most attention), progress bars show attention level, and captions show the current polling interval — shorter when the troop is vigilant, longer when calm.
4. Click **Explain most recent reading** to get a SHAP breakdown of which features are pushing the anomaly score up or down for the selected host's latest data point.
5. Use **Reset** to restart the simulation clock from the beginning.

## API reference (FastAPI, `backend/app.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service + host/metric list |
| `/hosts` | GET | List of monitored hosts and metrics |
| `/metrics/{host}` | GET | Recent revealed metric history for a host |
| `/simulate/tick?steps=N` | POST | Advance the simulated clock by N steps, score new data, update the lemur sensor |
| `/lemur/state` | GET | Current attention/sentinel/troop-alert snapshot |
| `/events?limit=N` | GET | Recent tick history with flagged anomalies |
| `/explain/{host}?index=I` | GET | SHAP explanation for a specific (or latest) data point |
| `/reset` | POST | Reset the simulation clock and lemur sensor state |

Interactive API docs are available at `http://localhost:8000/docs` once the backend is running.

## Notes on the ML approach

- **Model**: Isolation Forest, one per host, trained on the full synthetic history at startup. Isolation Forest was chosen because it's unsupervised (no labeled anomalies required, matching real-world monitoring), fast, and works well on the kind of tabular, multivariate sensor data used here.
- **Features**: raw metric values, short-window rolling standard deviation (volatility signal), and sine/cosine encodings of hour-of-day (circadian signal).
- **Explainability**: SHAP's permutation-based `Explainer` is used against the model's `decision_function`, which keeps it model-agnostic and robust even though `IsolationForest` isn't always covered by SHAP's fast tree-specific path.
- **This is a simulation**: metrics are synthetically generated (with injected spikes, drops, and sustained drifts) rather than pulled from a live VM/hypervisor. Swapping in a real metrics source (Prometheus, vCenter/VMware performance counters, cloud monitoring APIs, etc.) means replacing `data_generator.py` with a real data connector — the detection, explainability, and biomimicry layers are otherwise unchanged.

## Possible extensions

- Replace the synthetic feed with a real metrics source (Prometheus/vCenter/cloud monitoring API).
- Add cross-host alarm propagation (e.g., a db host alarm raises attention on dependent web hosts).
- Persist trained models and historical events to disk/DB instead of in-memory state.
- Add authentication and a proper task scheduler for the "polling interval" the LemurSensor computes (currently advisory/visualized, not yet driving real poll scheduling).
