import os, pandas as pd, requests
from pathlib import Path

WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")
if not WEBHOOK:
    raise SystemExit("Missing SLACK_WEBHOOK_URL environment variable.")

out = Path("out")
summary_path = out / "summary_ranked.csv"
if not summary_path.exists():
    raise SystemExit("summary_ranked.csv not found. Run Linkedin_Report_Task1.py first.")

df = pd.read_csv(summary_path).head(3)
lines = ["*Top LinkedIn Post(s) Today*"]
for _, r in df.iterrows():
    er = r.get("engagement_rate", 0)
    ctr = r.get("ctr", 0)
    score = r.get("predicted_performance_score", 0)
    url = r.get("url", "")
    pid = r.get("post_id", "")
    lines.append(f"• *{pid}* — ER: {er:.4f}, CTR: {ctr:.4f}, Score: {score:.2f} {('— ' + url) if url else ''}")

payload = {"text": "\n".join(lines)}
resp = requests.post(WEBHOOK, json=payload, timeout=10)
resp.raise_for_status()
print("Slack message posted.")