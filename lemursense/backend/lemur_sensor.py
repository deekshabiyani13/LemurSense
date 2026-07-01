"""
lemur_sensor.py
----------------
The biomimicry core of LemurSense.

Real ring-tailed lemurs live in troops that rely on shared vigilance:
one or more "sentinels" watch for threats while others forage, and when
a sentinel spots something, an alarm call redirects the whole troop's
attention toward the source of danger. Lemurs also forage adaptively —
spending more energy searching when food is scarce or conditions are
volatile, and conserving energy when things are calm. Their behavior
also follows circadian rhythm (most active at dawn/dusk).

LemurSense maps these behaviors onto IT monitoring:

  1. VIGILANCE / SENTINEL ROTATION
     Not every metric is watched with equal intensity all the time.
     Each (host, metric) pair has an "attention weight". Attention is
     a scarce resource (like a troop's limited number of sentinels) that
     gets reallocated toward metrics showing volatility or risk.

  2. ALARM CALLS / TROOP COMMUNICATION
     When a metric on a host is flagged anomalous, an "alarm call" is
     broadcast to related metrics on the same host (and optionally
     related hosts), temporarily raising their attention/sampling rate
     -- mirroring how a lemur troop reacts as a group to a threat, not
     just the individual that spotted it.

  3. ADAPTIVE FORAGING (sampling rate)
     Higher attention weight -> shorter polling interval (watched more
     closely). Lower attention -> longer interval (energy conserved).

  4. TERRITORY / SCENT-MARKING (baselining)
     Each metric has a learned "normal territory" (baseline range).
     Deviation from territory is what triggers vigilance in the first
     place -- handled by the anomaly detector, consumed here.

  5. CIRCADIAN AWARENESS
     Alarm decay and attention baselines are modulated by time-of-day,
     since "normal" volatility differs between business hours and
     overnight.
"""

from dataclasses import dataclass, field
from datetime import datetime
import math


MIN_INTERVAL_SEC = 15        # fastest possible polling (fully alarmed)
MAX_INTERVAL_SEC = 300       # slowest polling (fully calm)
ALARM_DECAY = 0.85           # how quickly alarm-boosted attention fades per tick
ALARM_PROPAGATION = 0.6      # how much of an alarm call spreads to sibling metrics
BASE_ATTENTION = 0.2         # resting attention weight, all metrics start here


@dataclass
class SentinelState:
    host: str
    metric: str
    attention: float = BASE_ATTENTION
    last_score: float = 0.0
    is_sentinel: bool = False
    history: list = field(default_factory=list)

    def polling_interval(self) -> float:
        """Higher attention -> shorter interval (watched more closely)."""
        a = max(0.0, min(1.0, self.attention))
        return MAX_INTERVAL_SEC - a * (MAX_INTERVAL_SEC - MIN_INTERVAL_SEC)


class LemurSensor:
    """
    Troop-level coordinator. Tracks a SentinelState per (host, metric)
    pair and updates attention allocation each tick based on incoming
    anomaly scores, alarm propagation, and circadian rhythm.
    """

    def __init__(self, hosts, metrics):
        self.hosts = hosts
        self.metrics = metrics
        self.states = {
            (h, m): SentinelState(host=h, metric=m)
            for h in hosts for m in metrics
        }
        self.troop_alert_level = 0.0  # 0 = calm troop, 1 = full alarm

    def _circadian_factor(self, timestamp: datetime) -> float:
        """
        Lemurs are most vigilant at dawn/dusk (crepuscular). We use a
        gentle multiplier so attention baselines rise slightly during
        typical high-traffic business hours (proxy for real risk windows)
        and ease off overnight.
        """
        hour = timestamp.hour + timestamp.minute / 60
        return 0.85 + 0.3 * math.sin((hour - 8) / 24 * 2 * math.pi)

    def update(self, timestamp: datetime, scores: dict):
        """
        scores: dict of {(host, metric): anomaly_score} where higher
        score = more anomalous (already normalized roughly to [0, 1]).
        """
        circadian = self._circadian_factor(timestamp)

        # Step 1: decay all attention toward baseline (troop relaxing)
        for state in self.states.values():
            state.attention = BASE_ATTENTION + (state.attention - BASE_ATTENTION) * ALARM_DECAY
            state.is_sentinel = False

        # Step 2: raise attention directly from this tick's anomaly scores
        for key, score in scores.items():
            state = self.states.get(key)
            if state is None:
                continue
            state.last_score = score
            state.history.append(score)
            if len(state.history) > 200:
                state.history.pop(0)
            boosted = BASE_ATTENTION + score * circadian
            state.attention = max(state.attention, min(1.0, boosted))

        # Step 3: alarm calls -- propagate to sibling metrics on same host
        alarm_threshold = 0.55
        alarmed_hosts = {
            key[0] for key, score in scores.items() if score >= alarm_threshold
        }
        for host in alarmed_hosts:
            for m in self.metrics:
                sibling = self.states[(host, m)]
                sibling.attention = min(1.0, max(sibling.attention, sibling.attention + ALARM_PROPAGATION * 0.4))

        # Step 4: designate sentinels = top attention metrics per host
        for host in self.hosts:
            host_states = [self.states[(host, m)] for m in self.metrics]
            top = max(host_states, key=lambda s: s.attention)
            top.is_sentinel = True

        # Step 5: troop-wide alert level = mean of top score per host
        per_host_max = [
            max(scores.get((h, m), 0.0) for m in self.metrics) for h in self.hosts
        ]
        self.troop_alert_level = sum(per_host_max) / max(1, len(per_host_max))

    def snapshot(self):
        """Return a serializable view of current sensor/troop state."""
        return {
            "troop_alert_level": round(self.troop_alert_level, 3),
            "sentinels": [
                {
                    "host": s.host,
                    "metric": s.metric,
                    "attention": round(s.attention, 3),
                    "polling_interval_sec": round(s.polling_interval(), 1),
                    "last_score": round(s.last_score, 3),
                    "is_sentinel": s.is_sentinel,
                }
                for s in self.states.values()
            ],
        }
