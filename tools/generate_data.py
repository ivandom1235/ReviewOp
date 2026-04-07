import json
import random
from pathlib import Path

domains_config = {
    "telecom": ["connectivity", "service quality", "value", "performance", "timeliness"],
    "healthcare": ["service quality", "timeliness", "value", "cleanliness", "accessibility"],
    "automotive": ["reliability", "build quality", "performance", "display quality", "comfort"],
    "finance": ["performance", "service quality", "value", "reliability", "accessibility"],
    "education": ["accessibility", "service quality", "comfort", "value", "cleanliness"],
    "travel": ["timeliness", "cleanliness", "comfort", "service quality", "connectivity", "value"],
    "gaming": ["performance", "display quality", "reliability", "value"],
    "home_appliances": ["sound quality", "cleanliness", "reliability", "accessibility", "value"],
    "ecommerce": ["reliability", "accessibility", "timeliness", "build quality", "value"],
    "delivery": ["food quality", "build quality", "service quality", "timeliness", "value", "accessibility"],
    "entertainment": ["performance", "sound quality", "accessibility", "value", "timeliness"],
    "real_estate": ["comfort", "service quality", "cleanliness", "accessibility", "value"],
    "fashion": ["build quality", "comfort", "reliability", "display quality", "value"],
    "fitness": ["reliability", "service quality", "cleanliness", "value", "accessibility"],
    "news_media": ["performance", "reliability", "accessibility", "value", "timeliness"],
    "legal": ["timeliness", "service quality", "cleanliness", "value", "reliability"]
}

def generate_row(domain, i):
    aspects = domains_config[domain]
    asp = random.choice(aspects)
    sent = random.choice(["positive", "negative"])
    text = f"Generic {sent} review about {asp} in {domain} domain sample {i}."
    
    return {
        "text": text,
        "domain": domain,
        "gold_interpretations": [{
            "aspect": asp,
            "sentiment": sent,
            "evidence": text
        }],
        "group_id": f"{domain}_gen_{i // 5:03d}"
    }

all_rows = []

# 500 for Telecom
for i in range(500):
    all_rows.append(generate_row("telecom", i))

# 20 for each of the other 15 domains
other_domains = [d for d in domains_config.keys() if d != "telecom"]
for domain in other_domains:
    for i in range(20):
        all_rows.append(generate_row(domain, i))

# Hard cases
hard_cases = [
    {"text": "I had to walk to the window just to finish the call.", "domain": "telecom", "aspect": "connectivity", "sentiment": "negative", "hardness": 2, "evidence": "walk to the window"},
    {"text": "I keep my phone plugged in whenever I'm home just in case.", "domain": "electronics", "aspect": "power", "sentiment": "negative", "hardness": 3, "evidence": "keep my phone plugged in"},
    {"text": "The service was exactly what I expected.", "domain": "telecom", "aspect": "general", "sentiment": "neutral", "abstain": True}
]

for i, case in enumerate(hard_cases):
    all_rows.append({
        "text": case["text"],
        "domain": case["domain"],
        "gold_interpretations": [{
            "aspect": case["aspect"],
            "sentiment": case["sentiment"],
            "evidence": case.get("evidence", ""),
            "hardness_tier": f"H{case.get('hardness', 0)}"
        }],
        "abstain_acceptable": case.get("abstain", False),
        "group_id": f"hard_case_{i:03d}"
    })

with open("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/input/v6_topup.jsonl", "w") as f:
    for row in all_rows:
        f.write(json.dumps(row) + "\n")

print(f"Generated {len(all_rows)} rows to v6_topup.jsonl")
