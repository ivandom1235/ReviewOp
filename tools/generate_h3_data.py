import json
from pathlib import Path
import random

input_dir = Path("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/input")
output_file = input_dir / "h3_targeting.jsonl"

domains = [
    "healthcare", "finance", "education", "entertainment", 
    "gaming", "real_estate", "legal", "automotive", 
    "travel", "fitness", "news_media", "ecommerce", 
    "home_appliances", "fashion", "delivery", "telecom"
]

h3_templates = {
    "connectivity": [
        "I had to walk to the window just to maintain the connection.",
        "The signal bars disappeared whenever I entered the basement.",
        "Resetting the router became a daily ritual for me.",
        "Searching for a hotspot was my main activity during the trip."
    ],
    "performance": [
        "The interface stuttered every time I scrolled down.",
        "I was staring at the loading icon for what felt like an eternity.",
        "Commands took several seconds to actually register on the screen.",
        "The stuttering made it almost impossible to finish the task."
    ],
    "thermal": [
        "The device felt uncomfortably hot to the touch after just ten minutes.",
        "The fan noise was so loud it drowned out the video.",
        "It started throttling down as soon as the temperature peaked.",
        "I had to put it on a cooling pad just to keep it from shutting off."
    ],
    "service quality": [
        "The staff seemed completely indifferent to our long wait.",
        "I was passed around between three different departments without an answer.",
        "Every question was met with a blank stare and no actual help.",
        "They made us feel like we were an inconvenience rather than customers."
    ],
    "value": [
        "I've seen similar features in products half this price.",
        "My wallet definitely felt the pinch for very little return.",
        "The premium price tag doesn't match the mediocre experience.",
        "I could have spent the same amount elsewhere for a much better result."
    ],
    "cleanliness": [
        "I noticed dust build-up in places that should have been pristine.",
        "The sticky residue on the table was a clear sign of poor upkeep.",
        "The environment didn't exactly instill confidence in their hygiene standards.",
        "I spent the first few minutes just wiping down the surfaces myself."
    ],
    "accessibility": [
        "The lack of a ramp made entering the building a struggle.",
        "The font was so small even with my glasses on.",
        "Navigation was a nightmare for anyone not familiar with the layout.",
        "It was impossible to find the elevator without asking several people."
    ]
}

rows = []
for domain in domains:
    for aspect, templates in h3_templates.items():
        for i in range(10): # 10 rows per aspect-domain combo
            template = random.choice(templates)
            evidence = template.split(" ")[-3:] # Pick some words as evidence
            evidence_str = " ".join(evidence).strip(".")
            
            row = {
                "text": template,
                "domain": domain,
                "gold_interpretations": [{
                    "aspect": aspect,
                    "sentiment": random.choice(["negative", "positive"]),
                    "evidence": evidence_str,
                    "hardness_tier": "H3"
                }],
                "group_id": f"h3_{domain}_{aspect}_{i:03d}"
            }
            rows.append(row)

with open(output_file, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print(f"Generated {len(rows)} H3 rows in {output_file.name}")
