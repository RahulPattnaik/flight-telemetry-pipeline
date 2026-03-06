"""
pipeline.py
Flight Telemetry Analytics Pipeline
------------------------------------
Stage 1 — Ingest & Parse    : Load raw CSV logs, validate schema, handle missing values.
Stage 2 — Clean & Index     : Resample to uniform 10 Hz, forward-fill gaps, add derived features.
Stage 3 — Feature Engineering: Phase labels, rolling stats, delta rates.
Stage 4 — Anomaly Detection  : Z-score + threshold rules → fault flags.
Stage 5 — Metrics & Reporting: Per-flight KPIs, fault summary, exportable report.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for scripted runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS / THRESHOLDS
# ─────────────────────────────────────────────
MOTOR_TEMP_WARN   = 80   # °C  — warning threshold
MOTOR_TEMP_CRIT   = 95   # °C  — critical / fault threshold
BATTERY_LOW_V     = 38   # V   — low battery warning
VIBRATION_HIGH    = 0.6  # g   — excessive vibration
ZSCORE_WINDOW     = 50   # rolling window (samples) for z-score
ZSCORE_THRESHOLD  = 3.5  # sigma

PHASE_RULES = {
    "ground"  : lambda df: df["altitude_m"] < 2,
    "takeoff" : lambda df: (df["altitude_m"] >= 2) & (df["altitude_m"] < 30) & (df["airspeed_ms"] < 15),
    "climb"   : lambda df: (df["altitude_m"] >= 30) & (df["altitude_ms_rate"] > 0.5),
    "cruise"  : lambda df: (df["altitude_m"] >= 30) & (df["altitude_ms_rate"].abs() <= 0.5),
    "descent" : lambda df: (df["altitude_m"] >= 2)  & (df["altitude_ms_rate"] < -0.5),
    "landing" : lambda df: (df["altitude_m"] < 30)  & (df["airspeed_ms"] < 10),
}


# ─────────────────────────────────────────────
# STAGE 1 — INGEST & PARSE
# ─────────────────────────────────────────────
REQUIRED_COLS = [
    "timestamp_s", "flight_id", "altitude_m", "airspeed_ms",
    "pitch_deg", "roll_deg", "yaw_deg",
    "motor1_temp_c", "motor2_temp_c", "motor3_temp_c", "motor4_temp_c",
    "battery_v", "battery_pct", "vibration_g",
]

def ingest(path: str) -> pd.DataFrame:
    """Load a single telemetry CSV, validate schema, report data quality."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    null_pct = df.isnull().mean() * 100
    noisy    = null_pct[null_pct > 0]
    if not noisy.empty:
        print(f"  [WARN] Null values in {Path(path).name}: {noisy.to_dict()}")

    df = df[REQUIRED_COLS].copy()
    df.sort_values("timestamp_s", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# STAGE 2 — CLEAN & INDEX
# ─────────────────────────────────────────────
def clean(df: pd.DataFrame, freq_hz: int = 10) -> pd.DataFrame:
    """
    - Resample to uniform time grid at freq_hz
    - Forward-fill gaps up to 1 second; NaN beyond that stays NaN
    - Drop duplicated timestamps
    Uses integer sample index internally to avoid floating-point reindex mismatches.
    """
    df = df.copy()
    # Convert to integer sample index (avoids float accumulation in np.arange)
    df["sample_idx"] = (df["timestamp_s"] * freq_hz).round().astype(int)
    df = df.drop_duplicates(subset="sample_idx")

    min_idx = df["sample_idx"].min()
    max_idx = df["sample_idx"].max()
    uniform_idx = np.arange(min_idx, max_idx + 1)

    df_idx = df.set_index("sample_idx")
    df_resampled = df_idx.reindex(uniform_idx)

    # Restore float timestamp
    df_resampled["timestamp_s"] = uniform_idx / freq_hz

    # Interpolate numeric sensor columns (linear, limit gap to 1 s = freq_hz samples)
    numeric_cols = [c for c in df_resampled.select_dtypes(include=np.number).columns
                    if c != "sample_idx"]
    df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(
        method="linear", limit=freq_hz, limit_direction="forward"
    )

    # Fill flight_id (string) forward
    df_resampled["flight_id"] = df_resampled["flight_id"].ffill()

    df_resampled.index.name = "sample_idx"
    df_resampled.reset_index(drop=True, inplace=True)
    return df_resampled


# ─────────────────────────────────────────────
# STAGE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived signals useful for anomaly detection and performance metrics."""

    # Rate of change (finite difference, per second)
    df["altitude_ms_rate"]  = df["altitude_m"].diff() * 10   # 10 Hz → per second
    df["airspeed_ms_rate"]  = df["airspeed_ms"].diff() * 10

    # Max motor temperature across all 4 motors
    motor_cols = ["motor1_temp_c", "motor2_temp_c", "motor3_temp_c", "motor4_temp_c"]
    df["motor_max_temp"]   = df[motor_cols].max(axis=1)
    df["motor_temp_spread"] = df[motor_cols].max(axis=1) - df[motor_cols].min(axis=1)

    # Rolling z-score on motor_max_temp (anomaly signal)
    roll_mean = df["motor_max_temp"].rolling(ZSCORE_WINDOW, min_periods=5).mean()
    roll_std  = df["motor_max_temp"].rolling(ZSCORE_WINDOW, min_periods=5).std()
    df["motor_temp_zscore"] = (df["motor_max_temp"] - roll_mean) / (roll_std + 1e-6)

    # Rolling vibration z-score
    v_mean = df["vibration_g"].rolling(ZSCORE_WINDOW, min_periods=5).mean()
    v_std  = df["vibration_g"].rolling(ZSCORE_WINDOW, min_periods=5).std()
    df["vibration_zscore"] = (df["vibration_g"] - v_mean) / (v_std + 1e-6)

    # Flight phase labels
    df["phase"] = "unknown"
    for phase, rule_fn in PHASE_RULES.items():
        try:
            mask = rule_fn(df)
            df.loc[mask, "phase"] = phase
        except Exception:
            pass

    return df


# ─────────────────────────────────────────────
# STAGE 4 — ANOMALY DETECTION
# ─────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based + statistical anomaly detection.
    Returns df with 'fault_flag' and 'fault_reason' columns.
    """
    df["fault_flag"]   = False
    df["fault_reason"] = ""

    def flag(mask, reason):
        df.loc[mask, "fault_flag"]    = True
        df.loc[mask, "fault_reason"] += reason + "; "

    # Motor temperature thresholds
    motor_cols = ["motor1_temp_c", "motor2_temp_c", "motor3_temp_c", "motor4_temp_c"]
    for col in motor_cols:
        mid = col.replace("_temp_c", "")
        flag(df[col] > MOTOR_TEMP_CRIT, f"{mid}_CRITICAL_OVERHEAT")
        flag((df[col] > MOTOR_TEMP_WARN) & (df[col] <= MOTOR_TEMP_CRIT), f"{mid}_TEMP_WARN")

    # Z-score spike on motor temperature
    flag(df["motor_temp_zscore"].abs() > ZSCORE_THRESHOLD, "MOTOR_TEMP_ANOMALY_ZSCORE")

    # Abnormal motor temperature spread (one motor diverging from others)
    flag(df["motor_temp_spread"] > 25, "MOTOR_TEMP_SPREAD_HIGH")

    # Battery
    flag(df["battery_v"] < BATTERY_LOW_V, "LOW_BATTERY_VOLTAGE")

    # Vibration
    flag(df["vibration_g"] > VIBRATION_HIGH, "HIGH_VIBRATION")
    flag(df["vibration_zscore"].abs() > ZSCORE_THRESHOLD, "VIBRATION_ANOMALY_ZSCORE")

    # Attitude limits
    flag(df["pitch_deg"].abs() > 30, "EXCESSIVE_PITCH")
    flag(df["roll_deg"].abs()  > 45, "EXCESSIVE_ROLL")

    return df


# ─────────────────────────────────────────────
# STAGE 5 — METRICS & REPORTING
# ─────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute per-flight KPIs from processed telemetry."""
    flight_id  = df["flight_id"].iloc[0]
    duration_s = df["timestamp_s"].max() - df["timestamp_s"].min()

    motor_cols = ["motor1_temp_c", "motor2_temp_c", "motor3_temp_c", "motor4_temp_c"]
    faults = df[df["fault_flag"]]
    fault_reasons = []
    if not faults.empty:
        all_reasons = "; ".join(faults["fault_reason"].tolist())
        from collections import Counter
        parts = [r.strip() for r in all_reasons.split(";") if r.strip()]
        fault_counts = Counter(parts)
        fault_reasons = [{"reason": k, "count": v} for k, v in fault_counts.most_common()]

    cruise = df[df["phase"] == "cruise"]

    metrics = {
        "flight_id"              : flight_id,
        "duration_s"             : round(duration_s, 1),
        "total_samples"          : len(df),
        "fault_samples"          : int(df["fault_flag"].sum()),
        "fault_rate_pct"         : round(df["fault_flag"].mean() * 100, 2),
        "fault_types"            : fault_reasons,
        "max_altitude_m"         : round(df["altitude_m"].max(), 1),
        "mean_cruise_airspeed_ms": round(cruise["airspeed_ms"].mean(), 2) if not cruise.empty else None,
        "max_motor_temp_c"       : round(df["motor_max_temp"].max(), 1),
        "max_motor_temp_spread"  : round(df["motor_temp_spread"].max(), 1),
        "min_battery_v"          : round(df["battery_v"].min(), 2),
        "min_battery_pct"        : round(df["battery_pct"].min(), 1),
        "mean_vibration_g"       : round(df["vibration_g"].mean(), 4),
        "phase_distribution"     : df["phase"].value_counts().to_dict(),
    }
    return metrics


def generate_report(all_metrics: list[dict], out_dir: str):
    """Save a JSON summary report of all flights."""
    report = {
        "pipeline_version": "1.0.0",
        "total_flights"   : len(all_metrics),
        "flagged_flights" : sum(1 for m in all_metrics if m["fault_rate_pct"] > 0),
        "flights"         : all_metrics,
    }
    path = os.path.join(out_dir, "flight_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {path}")
    return report


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def plot_flight(df: pd.DataFrame, metrics: dict, out_dir: str):
    """4-panel diagnostic plot for a single flight."""
    fid = metrics["flight_id"]
    t   = df["timestamp_s"]

    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    fig.suptitle(f"Flight Telemetry Dashboard — {fid}", fontsize=14,
                 color="white", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {"alt": "#4fc3f7", "speed": "#81c784", "fault": "#ef5350",
              "m1": "#ff8a65", "m2": "#ffd54f", "m3": "#ce93d8", "m4": "#80deea",
              "batt": "#a5d6a7", "vib": "#ffe082", "bg": "#161b22", "ax": "#e6edf3"}

    def styled_ax(ax):
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["ax"], labelsize=7)
        ax.xaxis.label.set_color(COLORS["ax"])
        ax.yaxis.label.set_color(COLORS["ax"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        return ax

    # Panel 1 — Altitude & Airspeed
    ax1 = styled_ax(fig.add_subplot(gs[0, :]))
    ax1.plot(t, df["altitude_m"], color=COLORS["alt"], lw=1, label="Altitude (m)")
    ax1b = ax1.twinx()
    ax1b.plot(t, df["airspeed_ms"], color=COLORS["speed"], lw=1, alpha=0.8, label="Airspeed (m/s)")
    ax1b.tick_params(colors=COLORS["ax"], labelsize=7)
    fault_mask = df["fault_flag"]
    ax1.scatter(t[fault_mask], df["altitude_m"][fault_mask], color=COLORS["fault"], s=4, zorder=5, label="Fault")
    ax1.set_ylabel("Altitude (m)", fontsize=8)
    ax1b.set_ylabel("Airspeed (m/s)", fontsize=8)
    ax1.set_title("Altitude & Airspeed with Fault Markers", color=COLORS["ax"], fontsize=9)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right",
               facecolor="#21262d", labelcolor="white")

    # Panel 2 — Motor Temperatures
    ax2 = styled_ax(fig.add_subplot(gs[1, 0]))
    for i, (col, color) in enumerate(zip(
        ["motor1_temp_c","motor2_temp_c","motor3_temp_c","motor4_temp_c"],
        [COLORS["m1"], COLORS["m2"], COLORS["m3"], COLORS["m4"]]
    )):
        ax2.plot(t, df[col], color=color, lw=0.9, label=f"Motor {i+1}")
    ax2.axhline(MOTOR_TEMP_WARN, color="#ffd54f", lw=0.8, ls="--", label=f"Warn {MOTOR_TEMP_WARN}°C")
    ax2.axhline(MOTOR_TEMP_CRIT, color=COLORS["fault"], lw=0.8, ls="--", label=f"Crit {MOTOR_TEMP_CRIT}°C")
    ax2.set_ylabel("Temp (°C)", fontsize=8)
    ax2.set_title("Motor Temperatures", color=COLORS["ax"], fontsize=9)
    ax2.legend(fontsize=6, facecolor="#21262d", labelcolor="white", ncol=2)

    # Panel 3 — Battery
    ax3 = styled_ax(fig.add_subplot(gs[1, 1]))
    ax3.plot(t, df["battery_v"], color=COLORS["batt"], lw=1, label="Voltage (V)")
    ax3b = ax3.twinx()
    ax3b.plot(t, df["battery_pct"], color="#80cbc4", lw=0.8, alpha=0.7, label="SOC (%)")
    ax3b.tick_params(colors=COLORS["ax"], labelsize=7)
    ax3.axhline(BATTERY_LOW_V, color=COLORS["fault"], lw=0.8, ls="--")
    ax3.set_ylabel("Voltage (V)", fontsize=8)
    ax3b.set_ylabel("SOC (%)", fontsize=8)
    ax3.set_title("Battery State", color=COLORS["ax"], fontsize=9)

    # Panel 4 — Vibration & Z-score anomaly
    ax4 = styled_ax(fig.add_subplot(gs[2, 0]))
    ax4.plot(t, df["vibration_g"], color=COLORS["vib"], lw=0.8, label="Vibration (g)")
    ax4.axhline(VIBRATION_HIGH, color=COLORS["fault"], lw=0.8, ls="--", label=f"Thresh {VIBRATION_HIGH}g")
    ax4.set_ylabel("Vibration (g)", fontsize=8)
    ax4.set_title("Vibration", color=COLORS["ax"], fontsize=9)
    ax4.legend(fontsize=7, facecolor="#21262d", labelcolor="white")

    # Panel 5 — Motor Temp Z-score
    ax5 = styled_ax(fig.add_subplot(gs[2, 1]))
    ax5.plot(t, df["motor_temp_zscore"], color="#f48fb1", lw=0.8, label="Motor Temp Z-score")
    ax5.axhline( ZSCORE_THRESHOLD, color=COLORS["fault"], lw=0.8, ls="--")
    ax5.axhline(-ZSCORE_THRESHOLD, color=COLORS["fault"], lw=0.8, ls="--")
    ax5.set_ylabel("Z-score (σ)", fontsize=8)
    ax5.set_title("Motor Temp Anomaly Signal", color=COLORS["ax"], fontsize=9)
    ax5.legend(fontsize=7, facecolor="#21262d", labelcolor="white")

    fig.text(0.5, 0.01,
             f"Duration: {metrics['duration_s']}s  |  Max Alt: {metrics['max_altitude_m']}m  "
             f"|  Fault Rate: {metrics['fault_rate_pct']}%  |  Min Battery: {metrics['min_battery_v']}V",
             ha="center", fontsize=8, color="#8b949e")

    path = os.path.join(out_dir, f"{fid}_dashboard.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Plot saved → {path}")


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run_pipeline(data_dir: str = "data", out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    log_files = sorted(glob.glob(os.path.join(data_dir, "*_telemetry.csv")))

    if not log_files:
        print(f"No telemetry files found in '{data_dir}/'. Run generate_logs.py first.")
        return

    print(f"Found {len(log_files)} flight log(s). Starting pipeline...\n")
    all_metrics = []

    for path in log_files:
        fname = Path(path).name
        print(f"[{fname}]")

        # Pipeline stages
        df = ingest(path)
        print(f"  Stage 1 — Ingested {len(df):,} rows")

        df = clean(df)
        print(f"  Stage 2 — Cleaned & resampled → {len(df):,} rows")

        df = engineer_features(df)
        print(f"  Stage 3 — Features engineered")

        df = detect_anomalies(df)
        n_faults = df["fault_flag"].sum()
        print(f"  Stage 4 — Anomaly detection: {n_faults:,} fault samples flagged")

        metrics = compute_metrics(df)
        all_metrics.append(metrics)
        print(f"  Stage 5 — Metrics: fault_rate={metrics['fault_rate_pct']}%, "
              f"max_motor_temp={metrics['max_motor_temp_c']}°C")

        # Save processed log
        processed_path = os.path.join(out_dir, fname.replace("_telemetry.csv", "_processed.csv"))
        df.to_csv(processed_path, index=False)

        # Plot
        plot_flight(df, metrics, out_dir)
        print()

    report = generate_report(all_metrics, out_dir)

    print("\n══════════════════ FLEET SUMMARY ══════════════════")
    print(f"  Total flights     : {report['total_flights']}")
    print(f"  Flagged flights   : {report['flagged_flights']}")
    for m in all_metrics:
        status = "⚠ FAULT" if m["fault_rate_pct"] > 0 else "✓ OK"
        print(f"  {m['flight_id']}  {status}  fault_rate={m['fault_rate_pct']}%  "
              f"max_temp={m['max_motor_temp_c']}°C")


if __name__ == "__main__":
    run_pipeline()
