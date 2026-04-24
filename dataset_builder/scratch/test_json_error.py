import json

# Simulated LLM response with unescaped quotes in the reason field
bad_json = '[{"index": 0, "action": "keep", "reason": "correctly identifies "special" pizza"}]'

try:
    json.loads(bad_json)
except Exception as e:
    print(f"Error: {e}")

# Simulated LLM response with missing comma
missing_comma = '[{"index": 0, "action": "keep", "reason": "ok"} {"index": 1, "action": "keep", "reason": "ok"}]'

try:
    json.loads(missing_comma)
except Exception as e:
    print(f"Error: {e}")
