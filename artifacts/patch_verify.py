
import os

path = r"c:\Users\MONISH\Desktop\GitHub Repo\ReviewOp\dataset_builder\scripts\verify_artifact.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    new_lines.append(line)
    if 'cf_stats = cf_payload.get("stats", {})' in line and not found:
        new_lines.append('    leakage = metrics.get("leakage", {}) if isinstance(metrics.get("leakage", {}), dict) else {}\n')
        new_lines.append('    near_duplicate_leakage = int(leakage.get("near_duplicate_leakage", 0) or 0)\n')
        new_lines.append('\n')
        new_lines.append('    if near_duplicate_leakage > 0:\n')
        new_lines.append('        msg = f"near_duplicate_leakage is {near_duplicate_leakage}, expected 0"\n')
        new_lines.append('        if profile in {"stability", "journal", "diagnostic_strict"}:\n')
        new_lines.append('            failures.append(msg)\n')
        new_lines.append('        else:\n')
        new_lines.append('            warnings.append(msg)\n')
        found = True

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Done")
