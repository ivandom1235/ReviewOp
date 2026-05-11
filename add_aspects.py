import json

file_path = "protonet/configs/generic_aspect_descriptions.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

new_aspects = {
  "display": {
    "level_1": "domain_holdout_laptop",
    "description": "screen quality, brightness, resolution, color accuracy, viewing clarity",
    "aliases": ["screen", "monitor", "lcd", "panel"],
    "behavior_triggers": [
      "screen is bright",
      "display looks sharp",
      "colors look washed out",
      "viewing angles are poor"
    ]
  },
  "keyboard": {
    "level_1": "domain_holdout_laptop",
    "description": "typing experience, key layout, key travel, key responsiveness, keyboard comfort",
    "aliases": ["keys", "keypad"],
    "behavior_triggers": [
      "keys feel stiff",
      "typing feels comfortable",
      "keyboard layout is cramped"
    ]
  },
  "battery_life": {
    "level_1": "domain_holdout_laptop",
    "description": "battery endurance, discharge rate, screen-on time",
    "aliases": ["battery", "power"],
    "behavior_triggers": [
      "battery lasts long",
      "battery drains fast",
      "died after 2 hours"
    ]
  },
  "customer_support": {
    "level_1": "domain_holdout_laptop",
    "description": "quality of customer support, warranty claims, tech support helpfulness",
    "aliases": ["support", "tech support", "warranty"],
    "behavior_triggers": [
      "support was helpful",
      "terrible customer service",
      "warranty covered it"
    ]
  }
}

# Just add a few representative ones from the list
# "trackpad, software, hard_drive, hardware, graphics, customer_support, portability, audio, built_in_mic, hdmi_port, installation_time, browser, bios, ram, windows, operating_system, battery_life, charging, fan_noise, heat, build_quality, screen_brightness, touchpad_responsiveness, storage_capacity"

additional = ["trackpad", "software", "hard_drive", "hardware", "graphics", "portability", "audio", "built_in_mic", "hdmi_port", "installation_time", "browser", "bios", "ram", "windows", "operating_system", "charging", "fan_noise", "heat", "build_quality", "screen_brightness", "touchpad_responsiveness", "storage_capacity"]

for a in additional:
    if a not in new_aspects:
        new_aspects[a] = {
            "level_1": "domain_holdout_laptop",
            "description": f"quality and performance of the {a.replace('_', ' ')}",
            "aliases": [a.replace('_', ' ')],
            "behavior_triggers": [f"{a.replace('_', ' ')} is good", f"{a.replace('_', ' ')} is bad"]
        }

for k, v in new_aspects.items():
    if k not in data:
        data[k] = v

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Added new aspects. Total aspects now: {len(data)}")
