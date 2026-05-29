
import os

path = r"c:\Users\MONISH\Desktop\GitHub Repo\ReviewOp\dataset_builder\orchestrator\release_gate.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    new_lines.append(line)
    if 'raise QualityGateError(gate_results, "Critical Failure: exact text leakage detected")' in line and not found:
        new_lines.append('        \n')
        new_lines.append('    # Near-duplicate leakage enforcement (EC-P1)\n')
        new_lines.append('    near_dup = int(leakage.get("near_duplicate_leakage", 0))\n')
        new_lines.append('    if near_dup > 0:\n')
        new_lines.append('        msg = f"near-duplicate split leakage detected ({near_dup})"\n')
        new_lines.append('        if profile in {"stability", "journal", "research_default", "diagnostic_strict"}:\n')
        new_lines.append('            gate_results = {\n')
        new_lines.append('                "status": "FAIL",\n')
        new_lines.append('                "profile": profile,\n')
        new_lines.append('                "failures": [msg],\n')
        new_lines.append('                "warnings": [],\n')
        new_lines.append('                "metrics": {"near_duplicate_leakage": near_dup},\n')
        new_lines.append('            }\n')
        new_lines.append('            raise QualityGateError(gate_results, msg)\n')
        new_lines.append('        warnings.append(msg)\n')
        found = True

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Done")
