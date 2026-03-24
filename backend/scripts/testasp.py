import json
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.implicit.dataset_reader import load_episode_split
from services.implicit.label_maps import load_label_encoder
from services.implicit.config import CONFIG

with open(CONFIG.ontology_path, "r", encoding="utf-8") as f:
    ontology = json.load(f)

label_maps = load_label_encoder()
rows = load_episode_split("train") + load_episode_split("val") + load_episode_split("test")

dataset_aspects = sorted({row["implicit_aspect"] for row in rows})

print("label_encoder aspects:", len(label_maps.aspect_to_id))
print("dataset aspects:", len(dataset_aspects))
print("dataset aspect names:", dataset_aspects)