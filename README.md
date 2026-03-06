# Flight Telemetry Analytics Pipeline

A production-style data pipeline that ingests raw aircraft telemetry logs,
cleans and indexes them, detects anomalies, and uses an LLM to generate
human-readable fault summaries — built for the LAT Aerospace technical interview.

---

## Architecture

```
Raw CSV Logs
     │
     ▼
[Stage 1] Ingest & Validate   → schema check, null reporting
     │
     ▼
[Stage 2] Clean & Resample    → uniform 10 Hz grid, interpolation
     │
     ▼
[Stage 3] Feature Engineering → phase labels, rolling stats, delta rates
     │
     ▼
[Stage 4] Anomaly Detection   → threshold rules + rolling z-score
     │
     ▼
[Stage 5] Metrics & Report    → per-flight KPIs, fleet JSON report
     │
     ▼
[LLM Layer] Claude API        → fault signatures, root cause, recommendations
```

---

## Files

| File | Purpose |
|------|---------|
| `generate_logs.py` | Simulates raw flight telemetry CSVs (5 flights, 2 with injected faults) |
| `pipeline.py` | Full 5-stage data pipeline + visualisation |
| `llm_summarizer.py` | Claude API integration for LLM log interpretation |
| `data/` | Raw telemetry CSVs (generated) |
| `outputs/` | Processed CSVs, dashboards (PNG), flight_report.json, llm_summaries.json |

---

## Quickstart

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib scipy anthropic

# 2. Generate simulated flight logs
python generate_logs.py

# 3. Run the full analytics pipeline
python pipeline.py

# 4. (Optional) Run LLM analysis — requires Anthropic API key
export ANTHROPIC_API_KEY=your_key_here
python llm_summarizer.py

# Analyse a single flight
python llm_summarizer.py --flight FLT_003
```

---

## Telemetry Schema

Each flight log is a CSV with 10 Hz sensor data:

| Column | Description |
|--------|-------------|
| `timestamp_s` | Elapsed time in seconds |
| `flight_id` | Unique flight identifier |
| `altitude_m` | GPS altitude in metres |
| `airspeed_ms` | Indicated airspeed in m/s |
| `pitch_deg` / `roll_deg` / `yaw_deg` | Attitude angles |
| `motor1..4_temp_c` | Per-motor temperature in °C |
| `battery_v` | Pack voltage |
| `battery_pct` | State of charge |
| `vibration_g` | Airframe vibration in g |

---

## Anomaly Detection Logic

### Threshold Rules
- Motor temp **> 80°C** → WARNING flag
- Motor temp **> 95°C** → CRITICAL flag
- Motor temp **spread > 25°C** → one motor diverging (bearing/winding fault)
- Battery **< 38V** → LOW_BATTERY_VOLTAGE
- Vibration **> 0.6g** → HIGH_VIBRATION
- Pitch **> 30°** or Roll **> 45°** → EXCESSIVE_ATTITUDE

### Statistical (Z-score)
- Rolling 50-sample window z-score on motor_max_temp and vibration_g
- Flag if |z| > 3.5σ — catches anomalies that stay below absolute thresholds
  but deviate significantly from local baseline

---

## LLM Integration

`llm_summarizer.py` sends structured JSON metrics to the Claude API with a
system prompt that constrains output to a structured JSON schema:

```json
{
  "flight_id": "FLT_003",
  "status": "CRITICAL",
  "summary": "...",
  "fault_signatures": ["Motor 2 overheat: 112°C peak, spread 47°C above fleet baseline"],
  "root_cause_hypotheses": ["Bearing seizure", "Winding short", "Coolant blockage"],
  "recommendations": ["Ground motor 2 for inspection", "Borescope rotor windings"],
  "performance_score": 34
}
```

This mirrors the production requirement of training LLMs to interpret logs
and generate performance metrics, summaries, and fault signatures.

---

## Interview Talking Points

1. **Pipeline design** — why each stage is separated (testability, debuggability, re-runnability)
2. **Resampling** — why uniform time grids matter for time-series ML models
3. **Z-score vs threshold** — thresholds miss slow drifts; z-score catches them
4. **Motor temp spread** — a single high reading vs divergence tells very different stories
5. **LLM prompt design** — structured JSON output, schema enforcement, fault signature vocabulary
6. **Scalability** — how you'd extend this: Parquet instead of CSV, streaming ingestion, vector DB for log search
