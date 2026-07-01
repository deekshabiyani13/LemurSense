"""
data_generator.py
------------------
Generates synthetic IT/VM system metrics used to simulate a monitored
environment for LemurSense. Metrics follow a daily (circadian) rhythm,
similar to how a lemur's environment has predictable day/night cycles,
with occasional injected anomalies (spikes, drops, sustained drift).
"""

import numpy as np
import pandas as pd

HOSTS = ["vm-web-01", "vm-db-01", "vm-cache-01", "vm-worker-01"]
METRICS = ["cpu_pct", "mem_pct", "disk_io_mbps", "net_io_mbps"]

BASELINES = {
    "vm-web-01":    {"cpu_pct": 35, "mem_pct": 55, "disk_io_mbps": 20, "net_io_mbps": 40},
    "vm-db-01":     {"cpu_pct": 45, "mem_pct": 70, "disk_io_mbps": 60, "net_io_mbps": 25},
    "vm-cache-01":  {"cpu_pct": 25, "mem_pct": 60, "disk_io_mbps": 10, "net_io_mbps": 55},
    "vm-worker-01": {"cpu_pct": 50, "mem_pct": 45, "disk_io_mbps": 30, "net_io_mbps": 20},
}

NOISE_STD = {"cpu_pct": 4, "mem_pct": 3, "disk_io_mbps": 5, "net_io_mbps": 6}


def _circadian(hour, amplitude):
    """Daily rhythm: busier during 'daylight' working hours, quieter at night."""
    return amplitude * np.sin((hour - 6) / 24 * 2 * np.pi) * 0.5 + amplitude * 0.5


def generate_host_series(host, start, periods, freq_minutes=5, seed=None, inject_anomalies=True):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min")
    base = BASELINES[host]

    df = pd.DataFrame({"timestamp": timestamps, "host": host})
    for metric in METRICS:
        hours = timestamps.hour + timestamps.minute / 60
        rhythm = _circadian(hours, amplitude=base[metric] * 0.25)
        noise = rng.normal(0, NOISE_STD[metric], size=periods)
        values = base[metric] + rhythm + noise
        df[metric] = np.clip(values, 0, None)

    df["is_anomaly"] = False
    df["anomaly_type"] = ""

    if inject_anomalies:
        n_anomalies = max(3, periods // 400)
        anomaly_idxs = rng.choice(periods, size=n_anomalies, replace=False)
        metric_col = {m: df.columns.get_loc(m) for m in METRICS}
        anomaly_col = df.columns.get_loc("is_anomaly")
        type_col = df.columns.get_loc("anomaly_type")

        for idx in anomaly_idxs:
            kind = rng.choice(["spike", "drop", "sustained_drift"])
            metric = rng.choice(METRICS)
            col = metric_col[metric]

            if kind == "spike":
                end = min(idx + 3, periods)
                rows = slice(idx, end)
                df.iloc[rows, col] = df.iloc[rows, col].to_numpy() * rng.uniform(2.2, 3.5)
                df.iloc[rows, anomaly_col] = True
                df.iloc[rows, type_col] = f"spike:{metric}"
            elif kind == "drop":
                end = min(idx + 3, periods)
                rows = slice(idx, end)
                df.iloc[rows, col] = df.iloc[rows, col].to_numpy() * rng.uniform(0.05, 0.3)
                df.iloc[rows, anomaly_col] = True
                df.iloc[rows, type_col] = f"drop:{metric}"
            else:  # sustained drift (e.g., memory leak)
                end = min(idx + 30, periods)
                rows = slice(idx, end)
                span_len = end - idx
                drift = np.linspace(0, base[metric] * 1.2, span_len)
                df.iloc[rows, col] = df.iloc[rows, col].to_numpy() + drift
                df.iloc[rows, anomaly_col] = True
                df.iloc[rows, type_col] = f"drift:{metric}"

    return df


def generate_dataset(start="2026-06-15", days=14, freq_minutes=5, seed=42):
    periods = int((days * 24 * 60) / freq_minutes)
    frames = []
    for i, host in enumerate(HOSTS):
        frames.append(
            generate_host_series(host, start, periods, freq_minutes, seed=seed + i)
        )
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["host", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "/home/claude/lemursense/data/synthetic_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
    print("Anomaly rate:", df["is_anomaly"].mean())
