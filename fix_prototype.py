import sys
import os
from pathlib import Path

def fix_file(path_str: str, search: str, replace: str):
    path = Path(path_str)
    if not path.exists():
        print(f"Skipping {path_str} (not found)")
        return False
    content = path.read_text(encoding="utf-8")
    if search in content:
        new_content = content.replace(search, replace)
        path.write_text(new_content, encoding="utf-8")
        print(f"Fixed {path_str}")
        return True
    print(f"Search text not found in {path_str}")
    return False

print("Starting protonet fixes...")

# 1. Fix separator in evaluator.py
# Update the helper function definition
fix_file("protonet/code/evaluator.py", 
         'def _aspect_from_joint(label: str) -> str:',
         'def _aspect_from_joint(label: str, separator: str) -> str:')

fix_file("protonet/code/evaluator.py", 
         'return label.split("__", 1)[0]',
         'return label.split(separator, 1)[0]')

# Update the calls to the helper function
fix_file("protonet/code/evaluator.py", 
         '_aspect_from_joint(true_label)',
         '_aspect_from_joint(true_label, cfg.joint_label_separator)')

fix_file("protonet/code/evaluator.py", 
         '_aspect_from_joint(pred_label)',
         '_aspect_from_joint(pred_label, cfg.joint_label_separator)')

# 2. Fix separator in prototype_bank.py
fix_file("protonet/code/prototype_bank.py",
         'label = str(item.get("joint_label") or f"{item.get(\'aspect\')}__{item.get(\'sentiment\')}")',
         'label = str(item.get("joint_label") or f"{item.get(\'aspect\')}{cfg.joint_label_separator}{item.get(\'sentiment\')}")')

# 3. Fix weight loading in runtime_infer.py
runtime_path = Path("protonet/code/runtime_infer.py")
if runtime_path.exists():
    c = runtime_path.read_text(encoding="utf-8")
    if 'projection.load_state_dict(payload["projection_state_dict"])' in c:
        fix_str = 'projection.load_state_dict(payload["projection_state_dict"])'
        replacement = fix_str + '\n        if "encoder_state" in payload:\n            encoder_state = payload["encoder_state"]\n            if "state_dict" in encoder_state and encoder.model is not None:\n                encoder.model.load_state_dict(encoder_state["state_dict"])'
        if replacement not in c:
            c = c.replace(fix_str, replacement)
            runtime_path.write_text(c, encoding="utf-8")
            print("Fixed runtime_infer.py weight loading")
        else:
            print("runtime_infer.py weight loading already fixed")

print("Fixes completed.")
