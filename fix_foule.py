"""Fix foule icon in solution-spec.json"""
import json

with open("static/config/solution-spec.json", "r", encoding="utf-8") as f:
    d = json.load(f)

for cat in d.get("categories", []):
    if cat.get("key") == "human":
        for it in cat.get("items", []):
            if it.get("id") == "foule":
                it["icon"] = "Ycrowdfoule.svg"
                break
        break

with open("static/config/solution-spec.json", "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

print("OK - foule icon fixed")
