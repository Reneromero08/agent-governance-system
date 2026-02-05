# CODEBOOK_SYNC_PROTOCOL: Session Management

**Parent Document:** [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)
**Section:** 8

---

## 8. Session Management

### 8.1 Session Lifecycle

```
SessionInit --> Handshake --> Synced --> [Operations] --> SessionEnd
                   |                          |
                   +---- Mismatch ------------+
                           |
                        Resync
```

### 8.2 Session Token

Successful handshake returns a session token for subsequent operations:

```json
{
  "session_token": "sess-abc123",
  "ttl_seconds": 3600,
  "sync_tuple_hash": "sha256:abc123..."
}
```

**Token Properties:**
- Bound to specific sync_tuple
- Expires after TTL
- Invalidated on any sync_tuple change
- Must be included in all pointer operations

### 8.3 Heartbeat

Long-lived sessions should send periodic heartbeats:

```json
{
  "message_type": "SYNC_HEARTBEAT",
  "session_token": "sess-abc123",
  "timestamp_utc": "2026-01-11T12:30:00Z",
  "local_codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015..."
}
```

Response:
```json
{
  "message_type": "HEARTBEAT_ACK",
  "session_token": "sess-abc123",
  "blanket_status": "ALIGNED",
  "ttl_remaining_seconds": 1800
}
```

### 8.4 Blanket Health Tracking

Beyond binary status, continuous blanket health enables predictive maintenance and early warning of alignment drift.

**Health Metrics:**
```json
{
  "blanket_health": 0.95,
  "drift_velocity": 0.001,
  "predicted_dissolution": "2026-01-12T00:00:00Z",
  "health_factors": {
    "r_value": 1.0,
    "ttl_fraction": 0.85,
    "heartbeat_streak": 47,
    "last_resync_distance": 3600
  }
}
```

| Metric | Type | Description |
|--------|------|-------------|
| `blanket_health` | float [0,1] | Composite health score |
| `drift_velocity` | float | Rate of health decline per second |
| `predicted_dissolution` | ISO timestamp | Extrapolated time when health < threshold |
| `health_factors` | object | Component scores |

**Health Computation:**
```python
def compute_blanket_health(session: SyncSession) -> dict:
    """Compute blanket health with predictive dissolution."""
    now = utc_now()

    # Factor 1: R-value from continuous formula (Section 7.5)
    r_value = session.continuous_r

    # Factor 2: TTL fraction remaining
    ttl_elapsed = (now - session.last_sync).total_seconds()
    ttl_fraction = max(0, 1 - ttl_elapsed / session.ttl_seconds)

    # Factor 3: Heartbeat reliability (streak of successful heartbeats)
    heartbeat_factor = min(1.0, session.heartbeat_streak / 10)

    # Factor 4: Time since last resync (recency)
    resync_age = (now - session.last_resync).total_seconds()
    resync_factor = 1 / (1 + resync_age / 86400)  # Decay over 24h

    # Composite health (weighted geometric mean)
    weights = [0.4, 0.3, 0.15, 0.15]
    factors = [r_value, ttl_fraction, heartbeat_factor, resync_factor]
    health = prod(f ** w for f, w in zip(factors, weights))

    # Drift velocity (health change rate)
    if session.prev_health is not None:
        dt = (now - session.prev_health_time).total_seconds()
        drift_velocity = (session.prev_health - health) / dt if dt > 0 else 0
    else:
        drift_velocity = 0

    # Predict dissolution (linear extrapolation)
    dissolution_threshold = 0.5
    if drift_velocity > 0 and health > dissolution_threshold:
        time_to_dissolution = (health - dissolution_threshold) / drift_velocity
        predicted_dissolution = now + timedelta(seconds=time_to_dissolution)
    else:
        predicted_dissolution = None

    return {
        "blanket_health": health,
        "drift_velocity": drift_velocity,
        "predicted_dissolution": predicted_dissolution.isoformat() if predicted_dissolution else None,
        "health_factors": {
            "r_value": r_value,
            "ttl_fraction": ttl_fraction,
            "heartbeat_streak": session.heartbeat_streak,
            "resync_factor": resync_factor
        }
    }
```

**Extended Heartbeat Response:**
```json
{
  "message_type": "HEARTBEAT_ACK",
  "session_token": "sess-abc123",
  "blanket_status": "ALIGNED",
  "ttl_remaining_seconds": 1800,
  "health": {
    "blanket_health": 0.95,
    "drift_velocity": 0.0001,
    "predicted_dissolution": null,
    "warning": null
  }
}
```

**Health Warnings:**

| Condition | Warning | Recommended Action |
|-----------|---------|-------------------|
| health < 0.8 | `HEALTH_DEGRADED` | Increase heartbeat frequency |
| drift_velocity > 0.01 | `DRIFT_DETECTED` | Investigate cause |
| predicted_dissolution within 1h | `DISSOLUTION_IMMINENT` | Proactive resync |

---

*Back to [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)*
