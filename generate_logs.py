"""
generate_logs.py
Simulates raw flight telemetry log files (CSV format) as would come
from a UAV / hybrid-electric STOL aircraft's onboard sensors.
Run this first to create sample data before running the pipeline.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_flight_log(flight_id: str, duration_s: int = 600, fault: bool = False) -> pd.DataFrame:
    """
    Generate a single flight log with 10 Hz sensor data.
    Columns mirror typical MAVLink / PX4 telemetry fields.
    If fault=True, inject a motor-overheat anomaly mid-flight.
    """
    freq = 10  # 10 Hz
    n = duration_s * freq
    t = np.linspace(0, duration_s, n)

    # --- Altitude profile: takeoff → cruise → landing ---
    alt = np.where(t < 60,  t * 2,                          # climb
          np.where(t < 480, 120 + 5 * np.sin(t / 30),      # cruise
                            120 - (t - 480) * 2.5))          # descent
    alt = np.clip(alt + np.random.normal(0, 0.5, n), 0, None)

    airspeed  = np.where(t < 60, t * 0.4, np.where(t < 480, 24 + np.random.normal(0, 0.8, n), 24 - (t-480)*0.1))
    pitch     = np.where(t < 60, 8.0, np.where(t < 480, np.random.normal(0, 1.5, n), -5.0))
    roll      = np.random.normal(0, 2, n)
    yaw       = np.cumsum(np.random.normal(0, 0.1, n)) % 360

    # Motor temperatures (4 motors)
    motor_temp_base = 55 + np.random.normal(0, 1, (n, 4))
    if fault:
        # Motor 2 overheats between t=200 and t=350 — peaks at ~115°C
        fault_idx = np.where((t >= 200) & (t <= 350))[0]
        spike = np.linspace(0, 60, len(fault_idx))
        motor_temp_base[fault_idx, 1] += spike

    battery_v  = np.clip(52 - (t / duration_s) * 12 + np.random.normal(0, 0.1, n), 30, 54)
    battery_pct = np.clip(100 - (t / duration_s) * 85 + np.random.normal(0, 0.3, n), 0, 100)
    vibration  = np.abs(np.random.normal(0.3, 0.05, n))

    df = pd.DataFrame({
        "timestamp_s"  : np.round(t, 2),
        "flight_id"    : flight_id,
        "altitude_m"   : np.round(alt, 2),
        "airspeed_ms"  : np.round(np.clip(airspeed, 0, None), 2),
        "pitch_deg"    : np.round(pitch, 3),
        "roll_deg"     : np.round(roll, 3),
        "yaw_deg"      : np.round(yaw % 360, 3),
        "motor1_temp_c": np.round(motor_temp_base[:, 0], 2),
        "motor2_temp_c": np.round(motor_temp_base[:, 1], 2),
        "motor3_temp_c": np.round(motor_temp_base[:, 2], 2),
        "motor4_temp_c": np.round(motor_temp_base[:, 3], 2),
        "battery_v"    : np.round(battery_v, 3),
        "battery_pct"  : np.round(battery_pct, 1),
        "vibration_g"  : np.round(vibration, 4),
    })
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    flights = [
        ("FLT_001", 600, False),
        ("FLT_002", 480, False),
        ("FLT_003", 720, True),   # has motor fault
        ("FLT_004", 540, False),
        ("FLT_005", 600, True),   # has motor fault
    ]
    for fid, dur, fault in flights:
        df = generate_flight_log(fid, dur, fault)
        path = f"data/{fid}_telemetry.csv"
        df.to_csv(path, index=False)
        print(f"Generated {path}  ({len(df):,} rows, fault={fault})")

    print("\nDone. Run: python pipeline.py")
