import sys
from pathlib import Path

target = Path("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/code/build_dataset.py")
content = target.read_text(encoding="utf-8")

old_def = "def _canonical_domain"
new_def = """def _get_row_domain(row: dict[str, Any]) -> str:
    # V6 Research Spec: Prefer explicit 'domain' key over filename inference.
    explicit_domain = row.get("domain")
    source_file = row.get("source_file", "unknown")
    if isinstance(explicit_domain, str) and explicit_domain and explicit_domain != "unknown":
        return explicit_domain.strip().lower()
    return _canonical_domain(str(source_file))


def _canonical_domain"""

if old_def in content:
    content = content.replace(old_def, new_def, 1)
    
# Also fix the call sites
old_call1 = 'Counter(str(_canonical_domain(str(row.get("source_file", "unknown")))) for row in train_rows)'
new_call1 = 'Counter(str(_get_row_domain(row)) for row in train_rows)'
content = content.replace(old_call1, new_call1)

old_call2 = 'sorted({str(_canonical_domain(str(row.get("source_file", "unknown")))) for row in train_rows})'
new_call2 = 'sorted({str(_get_row_domain(row)) for row in train_rows})'
content = content.replace(old_call2, new_call2)

old_call3 = 'domain = _canonical_domain(str(row.get("source_file", "unknown")))'
new_call3 = 'domain = _get_row_domain(row)'
content = content.replace(old_call3, new_call3)

target.write_text(content, encoding="utf-8")
print("Done fixing build_dataset.py")
