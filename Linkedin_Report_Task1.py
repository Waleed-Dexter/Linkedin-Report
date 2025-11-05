import os, re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


POST_ID   = "klarna-7346877091532959746-748v"
POST_URL  = "https://www.linkedin.com/posts/klarna_klarnas-climate-resilience-program-activity-7346877091532959746-748v/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAADnFyQBBmDIWAAFjnAInMjU44QmY2tSbC8"
AUTHOR    = "Klarna"
CREATED_AT = ""

TEXT = (
"🌍 We’re proud to announce the launch of our AI for Climate Resilience Program.\n\n"
"AI already powers everything we do at Klarna — and now we’re turning that same expertise toward the front lines of climate change. "
"We take pride in our legacy as a climate leader, and we’re committed to driving positive change for the future. "
"The AI for Climate Resilience Program will support pioneering projects that harness artificial intelligence to help climate-vulnerable communities adapt and thrive.\n\n"
"This is technology in service of both people and the planet.\n\n"
"This program will support local, practical, and community-owned solutions. From strengthening food security and improving health systems to building coastal resilience in the face of climate change.\n\n"
"What's on offer:\n"
"🧬 Grants of up to $300,000\n"
"👩‍🏫 Mentorship, training, and a supportive community of practice\n\n"
"We encourage applications from organizations working to reduce vulnerability of local communities to climate-related risks in low- and middle-income countries. "
"We welcome early stage applications as well, from teams that need support in developing technical details further. "
"Whether you’re using AI to support smallholder farmers, build early warning systems, or translate complex risk data into community action plans, we want to hear from you!\n\n"
"Find out more about the program and apply here 👉 https://lnkd.in/d3tFWFHJ"
)

HASHTAGS   = "" 
IMPRESSIONS = 0
LIKES       = 97
COMMENTS    = 7
SHARES      = 15
CLICKS      = 0 

AUDIENCE = {
    "role_Founder": 0, "role_CEO": 0, "role_CFO": 0, "role_Engineer": 0, "role_Head of Sustainability": 0,
    "seniority_C-Level": 0, "seniority_VP": 0, "seniority_Director": 0, "seniority_Manager": 0,
    "industry_Fintech": 0, "industry_E-commerce": 0, "industry_Retail": 0, "industry_Technology": 0,
    "company_size_11-50": 0, "company_size_51-200": 0, "company_size_201-1000": 0, "company_size_1001-5000": 0,
}

ICP = {
    "roles":        {"Founder": 0.25, "CFO": 0.25, "Head of Sustainability": 0.30, "CEO": 0.20},
    "seniority":    {"Director": 0.25, "VP": 0.30, "C-Level": 0.45},
    "industry":     {"Fintech": 0.5, "E-commerce": 0.5},
    "company_size": {"51-200": 0.3, "201-1000": 0.5, "1001-5000": 0.2}
}

OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

EMOJI_PATTERN   = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+")
URL_PATTERN     = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")

POS_WORDS = set("""amazing awesome great good excellent positive sustainable sustainability growth efficient love
impact win best benefit success thrilled excited happy thanks welcome""".split())
NEG_WORDS = set("""bad worse worst negative problem issue risk slow sad sorry decline drop hate angry disappointed""".split())

def simple_sentiment(text: str) -> float:
    tokens = re.findall(r"[A-Za-z']+", str(text).lower())
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, (pos + neg))

def predicted_performance_score_base(row) -> float:
    """Base heuristic score (before learned lifts)."""
    score = 0.0
    score += 2.5 * float(row.get("sentiment", 0))
    # Hashtag zone
    hc = int(row.get("hashtag_count", 0) or 0)
    if 2 <= hc <= 5:   score += 1.0
    elif hc == 1:      score += 0.3
    elif hc > 6:       score -= 0.3
    # Word count
    wc = int(row.get("word_count", 0) or 0)
    if 30 <= wc <= 80: score += 1.2
    elif 15 <= wc < 30:score += 0.5
    elif wc > 120:     score -= 0.5
    # Triggers
    score += 0.8 * int(row.get("has_question", 0) or 0)
    score += 0.2 * int(row.get("has_emoji", 0) or 0)
    score += 0.2 * int(row.get("has_exclaim", 0) or 0)
    score += 0.1 * int(row.get("has_mentions", 0) or 0)
    score += 0.2 * int(row.get("has_link", 0) or 0)
    # Timing
    hour = row.get("hour"); weekday = row.get("weekday")
    if pd.notna(hour) and pd.notna(weekday) and weekday <= 4 and 8 <= hour <= 11:
        score += 0.5
    return round(score, 4)

def compute_icp_score(aud_row: pd.Series, icp: dict) -> float:
    score, total_w = 0.0, 0.0
    for role, w in icp.get("roles", {}).items():
        total_w += w; score += w * (float(aud_row.get(f"role_{role}", 0)) / 100.0)
    for s, w in icp.get("seniority", {}).items():
        total_w += w; score += w * (float(aud_row.get(f"seniority_{s}", 0)) / 100.0)
    for ind, w in icp.get("industry", {}).items():
        total_w += w; score += w * (float(aud_row.get(f"industry_{ind}", 0)) / 100.0)
    for cs, w in icp.get("company_size", {}).items():
        total_w += w; score += w * (float(aud_row.get(f"company_size_{cs}", 0)) / 100.0)
    return round(score / max(total_w, 1e-6), 4)

def save_chart(x, y, title, xlab, ylab, filename):
    plt.figure()
    plt.bar(x, y)
    plt.title(title); plt.xlabel(xlab); plt.ylabel(ylab)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path); plt.close()
    return path

created_at = pd.to_datetime(CREATED_AT, errors="coerce") if CREATED_AT else pd.NaT
posts = pd.DataFrame([{
    "post_id": POST_ID,
    "created_at": created_at,
    "author": AUTHOR,
    "url": POST_URL,
    "text": TEXT,
    "hashtags": HASHTAGS,
    "impressions": int(IMPRESSIONS),
    "likes": int(LIKES),
    "comments": int(COMMENTS),
    "shares": int(SHARES),
    "clicks": int(CLICKS)
}])

df = posts.copy()
df["text_len"] = df["text"].astype(str).str.len()
df["word_count"] = df["text"].astype(str).str.split().str.len()
df["has_link"] = df["text"].astype(str).str.contains(URL_PATTERN).astype(int)
df["has_emoji"] = df["text"].astype(str).str.contains(EMOJI_PATTERN).astype(int)
df["has_question"] = df["text"].astype(str).str.contains(r"\?").astype(int)
df["has_exclaim"] = df["text"].astype(str).str.contains(r"!").astype(int)
df["has_mentions"] = df["text"].astype(str).str.contains(MENTION_PATTERN).astype(int)
df["hashtag_count"] = df["text"].astype(str).apply(lambda t: len(re.findall(HASHTAG_PATTERN, t)))
df["sentiment"] = df["text"].astype(str).apply(simple_sentiment)

# Engagement metrics
df["engagements"] = df[["likes","comments","shares"]].sum(axis=1)
df["impressions"] = df["impressions"].clip(lower=1)
df["engagement_rate"] = (df["engagements"] / df["impressions"]).round(6)
df["ctr"] = (df["clicks"] / df["impressions"]).round(6)

# Time features
df["hour"] = pd.to_datetime(df["created_at"], errors="coerce").dt.hour
df["weekday"] = pd.to_datetime(df["created_at"], errors="coerce").dt.dayofweek

# Base score
df["predicted_performance_score"] = df.apply(predicted_performance_score_base, axis=1)


def learn_hashtag_lift(feat: pd.DataFrame):
    if "hashtag_count" not in feat.columns or "engagement_rate" not in feat.columns or len(feat) < 5:
        return {}
    tmp = feat.copy()
    tmp["bucket"] = pd.cut(tmp["hashtag_count"],
                           bins=[-1,0,1,3,5,10,999],
                           labels=["0","1","2-3","4-5","6-10",">10"])
    bucket_means = tmp.groupby("bucket", dropna=False)["engagement_rate"].mean()
    overall = tmp["engagement_rate"].mean()
    lift = {}
    for b, m in bucket_means.items():
        key = str(b)
        if pd.notna(m) and overall and overall != 0:
            lift[key] = float(m / overall)
        else:
            lift[key] = 1.0
    return lift

def learn_time_lift(feat: pd.DataFrame):
    if "engagement_rate" not in feat.columns or len(feat) < 5:
        return {"hour": {}, "weekday": {}}
    overall = feat["engagement_rate"].mean()
    hour_lift, weekday_lift = {}, {}

    if "hour" in feat.columns:
        hg = feat.groupby("hour", dropna=False)["engagement_rate"].mean()
        for h, m in hg.items():
            key = int(h) if pd.notna(h) else -1
            hour_lift[key] = float(m / overall) if pd.notna(m) and overall else 1.0

    if "weekday" in feat.columns:
        wg = feat.groupby("weekday", dropna=False)["engagement_rate"].mean()
        for w, m in wg.items():
            key = int(w) if pd.notna(w) else -1
            weekday_lift[key] = float(m / overall) if pd.notna(m) and overall else 1.0

    return {"hour": hour_lift, "weekday": weekday_lift}

hashtag_lifts = learn_hashtag_lift(df)
time_lifts = learn_time_lift(df)

def apply_learned_lifts(row):
    # Hashtag bucket lift
    hc = int(row.get("hashtag_count", 0) or 0)
    if hc <= 0: bucket = "0"
    elif hc == 1: bucket = "1"
    elif 2 <= hc <= 3: bucket = "2-3"
    elif 4 <= hc <= 5: bucket = "4-5"
    elif 6 <= hc <= 10: bucket = "6-10"
    else: bucket = ">10"
    h_lift = hashtag_lifts.get(bucket, 1.0)

    # Time lifts
    h = row.get("hour"); w = row.get("weekday")
    hour_lift = time_lifts.get("hour", {}).get(int(h) if pd.notna(h) else -1, 1.0)
    wd_lift   = time_lifts.get("weekday", {}).get(int(w) if pd.notna(w) else -1, 1.0)

    # Combine conservatively (dampen to avoid overfitting on small data)
    combined = h_lift * hour_lift * wd_lift
    damp = 0.5
    return 1.0 + (combined - 1.0) * damp

df["predicted_performance_score"] = (
    df["predicted_performance_score"] * df.apply(apply_learned_lifts, axis=1)
).round(4)

aud_row = {"post_id": POST_ID}
aud_row.update(AUDIENCE)
aud_score_df = pd.DataFrame([{
    "post_id": POST_ID,
    "icp_relevance_score": (
        (lambda r: (
            (ICP["roles"].get("Founder",0)   * (r.get("role_Founder",0)/100)) +
            (ICP["roles"].get("CFO",0)       * (r.get("role_CFO",0)/100)) +
            (ICP["roles"].get("Head of Sustainability",0) * (r.get("role_Head of Sustainability",0)/100)) +
            (ICP["roles"].get("CEO",0)       * (r.get("role_CEO",0)/100)) +
            (ICP["seniority"].get("Director",0) * (r.get("seniority_Director",0)/100)) +
            (ICP["seniority"].get("VP",0)       * (r.get("seniority_VP",0)/100)) +
            (ICP["seniority"].get("C-Level",0)  * (r.get("seniority_C-Level",0)/100)) +
            (ICP["industry"].get("Fintech",0)   * (r.get("industry_Fintech",0)/100)) +
            (ICP["industry"].get("E-commerce",0)* (r.get("industry_E-commerce",0)/100)) +
            (ICP["company_size"].get("51-200",0)    * (r.get("company_size_51-200",0)/100)) +
            (ICP["company_size"].get("201-1000",0)  * (r.get("company_size_201-1000",0)/100)) +
            (ICP["company_size"].get("1001-5000",0) * (r.get("company_size_1001-5000",0)/100))
        ))(AUDIENCE)
    )
}])

feat_cols = [
    "post_id","created_at","author","url","text","hashtags",
    "impressions","likes","comments","shares","clicks",
    "engagements","engagement_rate","ctr","word_count","text_len",
    "hashtag_count","sentiment","has_link","has_emoji","has_question",
    "has_exclaim","has_mentions","hour","weekday","predicted_performance_score"
]

features_out = os.path.join(OUT_DIR, "post_features_and_predictions.csv")
audience_out = os.path.join(OUT_DIR, "post_icp_relevance.csv")
summary_out  = os.path.join(OUT_DIR, "summary_ranked.csv")

df[feat_cols].to_csv(features_out, index=False)
aud_score_df.to_csv(audience_out, index=False)

summary = df.merge(aud_score_df, on="post_id", how="left") if "post_id" in aud_score_df.columns else df.copy()
if "icp_relevance_score" not in summary.columns:
    summary["icp_relevance_score"] = 0.0
summary.to_csv(summary_out, index=False)

eng_plot = save_chart(
    summary["post_id"], summary["engagement_rate"],
    "Engagement Rate", "Post ID", "Engagement Rate", "engagement_rate_by_post.png"
)
pred_plot = save_chart(
    summary["post_id"], summary["predicted_performance_score"],
    "Predicted Performance Score", "Post ID", "Score", "predicted_score_by_post.png"
)
icp_plot = save_chart(
    summary["post_id"], summary["icp_relevance_score"].fillna(0),
    "ICP Relevance Score", "Post ID", "ICP Relevance (0–1)", "icp_relevance_by_post.png"
)

html_path = os.path.join(OUT_DIR, "linkedin_growth_report.html")
def render_table(d, n=50): return d.head(n).to_html(index=False, escape=False)
with open(html_path, "w", encoding="utf-8") as f:
    f.write(f"""<!doctype html><html><head><meta charset='utf-8'><title>LinkedIn Single Post Report</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px}}.card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:20px}}img{{max-width:100%}}</style>
</head><body>
<h1>LinkedIn Single Post Report</h1>
<p><b>URL:</b> <a href="{POST_URL}">{POST_URL}</a></p>
<div class="card"><h2>Summary</h2>{render_table(summary[['post_id','url','engagement_rate','ctr','predicted_performance_score','icp_relevance_score']])}</div>
<div class="card"><h2>Feature Details</h2>{render_table(df[feat_cols])}</div>
<div class="card"><h2>ICP Relevance Details</h2>{render_table(aud_score_df)}</div>
</body></html>""")

print("Done. Outputs written to ./out/")
print(f"- {features_out}")
print(f"- {audience_out}")
print(f"- {summary_out}")
print(f"- {html_path}")