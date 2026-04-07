import sys
from pathlib import Path

target = Path("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/code/build_dataset.py")
content = target.read_text(encoding="utf-8")

# SITE 1: source_domain_family_counts (Line 3121)
content = content.replace(
    'Counter(str(_benchmark_domain_family(str(row.get("domain", "unknown")))) for row in rows)',
    'Counter(str(_benchmark_domain_family(str(_get_row_domain(row)))) for row in rows)'
)

# SITE 2: benchmark_domain_family_counts (Line 3122)
content = content.replace(
    'Counter(str(_benchmark_domain_family(str(row.get("domain", "unknown")))) for row in benchmark_rows)',
    'Counter(str(_benchmark_domain_family(str(_get_row_domain(row)))) for row in benchmark_rows)'
)

# SITE 3: _select_working_rows priority sorting (Line 188)
content = content.replace(
    '_benchmark_domain_family(row.get("domain"))',
    '_benchmark_domain_family(str(_get_row_domain(row)))'
)

# SITE 4: _grouped_split and other sites if they exist
content = content.replace(
    'domain = str(row.get("domain", "unknown"))',
    'domain = str(_get_row_domain(row))'
)

target.write_text(content, encoding="utf-8")
print("Done fixing all domain inference sites in build_dataset.py")
