import json
from pathlib import Path
from collections import defaultdict

input_dir = Path("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/input")
topup_file = input_dir / "v6_topup.jsonl"

if not topup_file.exists():
    print(f"{topup_file} not found.")
    exit(1)

rows_by_domain = defaultdict(list)
with open(topup_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        rows_by_domain[row.get("domain", "unknown")].append(row)

for domain, rows in rows_by_domain.items():
    domain_file = input_dir / f"{domain}.jsonl"
    with open(domain_file, "a") as f: # Append if exists
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {domain_file.name}")

# Delete the temporary file
topup_file.unlink()
print(f"Deleted {topup_file.name}")
