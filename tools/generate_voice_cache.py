# tools/generate_voice_cache.py
from gtts import gTTS
import json
from pathlib import Path
import os

cfg = json.load(open("cameras.json"))
cams = cfg.get("cameras", [])
cache_dir = Path("data/voice_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

for cam in cams:
    name = cam.get("name", "camera")
    text = f"Attention. Fire detected in {name}. Please evacuate immediately."
    slug = "".join(c if c.isalnum() else "_" for c in name).lower()
    out = cache_dir / f"{slug}.mp3"
    if out.exists():
        print("exists:", out)
        continue
    try:
        t = gTTS(text=text, lang="en")
        t.save(str(out))
        print("saved:", out)
    except Exception as e:
        print("failed:", name, e)
