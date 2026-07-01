"""
app.py
-------
FastAPI backend for LemurSense.

Simulates a live monitoring feed over the pre-generated synthetic
dataset: each call to /simulate/tick advances a virtual clock by one
step, scores the newest data point per host with the anomaly engine,
feeds those scores into the LemurSensor (adaptive attention / alarm
propagation), and returns everything the dashboard needs to render.
"""

from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_generator import generate_dataset, HOSTS, METRICS
from anomaly_detector import AnomalyEngine
from explainer import HostExplainer
from lemur_sensor import LemurSensor

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    df = generate_dataset(days=14)
    engine = AnomalyEngine().fit_all(df)

    explainers = {
        host: HostExplainer(engine.models[host], df[df.host == host])
        for host in HOSTS
    }

    STATE["df"] = df
    STATE["engine"] = engine
    STATE["explainers"] = explainers
    STATE["sensor"] = LemurSensor(HOSTS, METRICS)
    # cursor = how many rows of "history" have been revealed per host, starts
    # after a warmup window so rolling features have context immediately
    warmup = 40
    STATE["cursor"] = {host: warmup for host in HOSTS}
    STATE["host_frames"] = {
        host: df[df.host == host].sort_values("timestamp").reset_index(drop=True)
        for host in HOSTS
    }
    STATE["latest_scores"] = {}
    STATE["events"] = []  # rolling log of ticks with anomalies
    yield
    STATE.clear()


app = FastAPI(title="LemurSense API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TickResult(BaseModel):
    timestamp: str
    scores: dict
    flagged: list
    troop_alert_level: float


@app.get("/health")
def health():
    return {"status": "ok", "hosts": HOSTS, "metrics": METRICS}


@app.get("/hosts")
def get_hosts():
    return {"hosts": HOSTS, "metrics": METRICS}


@app.get("/metrics/{host}")
def get_metrics(host: str, window: int = 200):
    if host not in HOSTS:
        raise HTTPException(404, f"Unknown host {host}")
    frame = STATE["host_frames"][host]
    cursor = STATE["cursor"][host]
    start = max(0, cursor - window)
    sub = frame.iloc[start:cursor].copy()
    sub["timestamp"] = sub["timestamp"].astype(str)
    return sub.to_dict(orient="records")


@app.post("/simulate/tick")
def simulate_tick(steps: int = 1):
    engine: AnomalyEngine = STATE["engine"]
    sensor: LemurSensor = STATE["sensor"]
    last_result = None

    for _ in range(steps):
        scores_this_tick = {}
        flagged = []
        timestamp = None

        for host in HOSTS:
            frame = STATE["host_frames"][host]
            cursor = STATE["cursor"][host]
            if cursor >= len(frame):
                continue

            window_df = frame.iloc[: cursor + 1]
            score_arr = engine.score_host(host, window_df)
            score = float(score_arr[-1])
            timestamp = frame.iloc[cursor]["timestamp"]

            for m in METRICS:
                # attribute the host-level score to whichever metric is
                # most volatile right now as a lightweight per-metric proxy;
                # the SHAP explainer gives the real per-feature breakdown.
                scores_this_tick[(host, m)] = score

            if score >= 0.55:
                flagged.append({
                    "host": host,
                    "timestamp": str(timestamp),
                    "score": round(score, 3),
                    "true_label": bool(frame.iloc[cursor]["is_anomaly"]),
                    "true_type": frame.iloc[cursor]["anomaly_type"],
                })

            STATE["latest_scores"][host] = score
            STATE["cursor"][host] = cursor + 1

        if timestamp is not None:
            sensor.update(timestamp, scores_this_tick)

        last_result = {
            "timestamp": str(timestamp) if timestamp is not None else None,
            "scores": {h: round(STATE["latest_scores"].get(h, 0.0), 3) for h in HOSTS},
            "flagged": flagged,
            "troop_alert_level": round(sensor.troop_alert_level, 3),
        }
        STATE["events"].append(last_result)
        if len(STATE["events"]) > 500:
            STATE["events"].pop(0)

    return last_result


@app.get("/lemur/state")
def lemur_state():
    sensor: LemurSensor = STATE["sensor"]
    return sensor.snapshot()


@app.get("/events")
def get_events(limit: int = 50):
    return STATE["events"][-limit:]


@app.get("/explain/{host}")
def explain(host: str, index: Optional[int] = None):
    if host not in HOSTS:
        raise HTTPException(404, f"Unknown host {host}")
    frame = STATE["host_frames"][host]
    cursor = STATE["cursor"][host]
    if cursor == 0:
        raise HTTPException(400, "No data revealed yet for this host, call /simulate/tick first")

    row_index = index if index is not None else cursor - 1
    if row_index < 0 or row_index >= cursor:
        raise HTTPException(400, f"index must be within [0, {cursor - 1}]")

    window_df = frame.iloc[: cursor]
    explainer: HostExplainer = STATE["explainers"][host]
    result = explainer.explain_row(window_df, row_index, max_evals=150)
    result["host"] = host
    result["timestamp"] = str(frame.iloc[row_index]["timestamp"])
    result["true_label"] = bool(frame.iloc[row_index]["is_anomaly"])
    result["true_type"] = frame.iloc[row_index]["anomaly_type"]
    return result


@app.post("/reset")
def reset():
    warmup = 40
    STATE["cursor"] = {host: warmup for host in HOSTS}
    STATE["latest_scores"] = {}
    STATE["events"] = []
    STATE["sensor"] = LemurSensor(HOSTS, METRICS)
    return {"status": "reset"}
