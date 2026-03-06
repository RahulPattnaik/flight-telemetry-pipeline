"""
llm_summarizer.py
──────────────────
Uses the Anthropic Claude API to interpret processed flight telemetry data
and generate human-readable summaries, fault signatures, and recommendations.

This simulates the "Train LLM-based models to interpret logs and generate
performance metrics, summaries, and fault signatures" requirement from LAT.

Usage:
    python llm_summarizer.py                   # summarises all flights in outputs/
    python llm_summarizer.py --flight FLT_003  # single flight
    python llm_summarizer.py --batch           # batch mode, one API call per flight
"""

import os
import json
import argparse
import anthropic


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an aerospace telemetry analyst AI embedded in a
flight-data analytics platform. Your job is to interpret structured flight
metrics and produce concise, technically precise reports for GNC/autonomy
engineers.

Always respond in the following JSON format (no prose outside the JSON):
{
  "flight_id": "...",
  "status": "NOMINAL" | "WARNING" | "CRITICAL",
  "summary": "2-3 sentence plain-English summary of the flight",
  "fault_signatures": ["list of identified fault patterns with technical detail"],
  "root_cause_hypotheses": ["ordered list of likely causes, most probable first"],
  "recommendations": ["actionable steps for the maintenance/flight-test team"],
  "performance_score": 0-100
}"""


def build_user_prompt(metrics: dict) -> str:
    return f"""Analyse the following flight telemetry metrics and return a structured report.

FLIGHT METRICS (JSON):
{json.dumps(metrics, indent=2)}

Key thresholds for your analysis:
- Motor temperature WARNING: 80°C, CRITICAL: 95°C
- Battery low voltage: 38 V
- High vibration: > 0.6 g
- Normal motor temp spread: < 10°C (spread > 25°C suggests one motor diverging)
- Fault rate > 5% = significant concern; > 20% = flight safety event

Identify fault signatures, hypothesise root causes, and give actionable recommendations."""


# ── API call ──────────────────────────────────────────────────────────────────

def call_llm(metrics: dict, client: anthropic.Anthropic) -> dict:
    """Send one flight's metrics to Claude and parse the structured response."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(metrics)}],
    )
    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    return json.loads(raw)


# ── Batch runner ──────────────────────────────────────────────────────────────

def summarise_all(report_path: str = "outputs/flight_report.json",
                  out_path: str = "outputs/llm_summaries.json",
                  flight_filter: str = None):

    if not os.path.exists(report_path):
        print(f"Report not found at '{report_path}'. Run pipeline.py first.")
        return

    with open(report_path) as f:
        report = json.load(f)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERROR] Set the ANTHROPIC_API_KEY environment variable.")
        return

    client    = anthropic.Anthropic(api_key=api_key)
    summaries = []

    flights = report["flights"]
    if flight_filter:
        flights = [m for m in flights if m["flight_id"] == flight_filter]
        if not flights:
            print(f"Flight '{flight_filter}' not found in report.")
            return

    print(f"Sending {len(flights)} flight(s) to Claude for analysis...\n")

    for metrics in flights:
        fid = metrics["flight_id"]
        print(f"  Analysing {fid} ...", end=" ", flush=True)
        try:
            result = call_llm(metrics, client)
            summaries.append(result)
            status = result.get("status", "?")
            score  = result.get("performance_score", "?")
            print(f"done  [{status}  score={score}]")
        except Exception as e:
            print(f"ERROR — {e}")
            summaries.append({"flight_id": fid, "error": str(e)})

    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nLLM summaries saved → {out_path}")

    # Pretty-print to console
    print("\n══════════ LLM ANALYSIS RESULTS ══════════")
    for s in summaries:
        if "error" in s:
            print(f"\n[{s['flight_id']}] ERROR: {s['error']}")
            continue
        print(f"\n[{s['flight_id']}]  Status: {s['status']}  Score: {s.get('performance_score')}/100")
        print(f"  Summary: {s.get('summary','')}")
        if s.get("fault_signatures"):
            print("  Faults:")
            for sig in s["fault_signatures"]:
                print(f"    • {sig}")
        if s.get("recommendations"):
            print("  Recommendations:")
            for rec in s["recommendations"]:
                print(f"    → {rec}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Flight Log Summariser")
    parser.add_argument("--report",  default="outputs/flight_report.json",  help="Path to pipeline report JSON")
    parser.add_argument("--out",     default="outputs/llm_summaries.json",  help="Output path for LLM summaries")
    parser.add_argument("--flight",  default=None, help="Analyse a single flight ID (e.g. FLT_003)")
    args = parser.parse_args()

    summarise_all(report_path=args.report, out_path=args.out, flight_filter=args.flight)
