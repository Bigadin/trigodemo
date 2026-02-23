"""
Migration script: seed lieux.json, sites.json, benefits.json from existing data.

Reads:
  - data/cameras.json   (existing cameras → attach to a default site)
  - data/zones.json     (existing zones → create benefits from them)
Writes:
  - data/lieux.json
  - data/sites.json
  - data/benefits.json
  - data/cameras.json   (adds site_id to each camera)

Safe to run multiple times: skips entries that already exist.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

DATA = Path(__file__).parent / "data"

def load(f):
    p = DATA / f
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def save(f, data):
    (DATA / f).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def snake(name):
    import unicodedata
    s = unicodedata.normalize("NFD", str(name or ""))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().strip()
    import re
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or f"id_{int(datetime.now().timestamp())}"

def main():
    now = datetime.now(timezone.utc).isoformat()

    lieux = load("lieux.json")
    sites = load("sites.json")
    benefits = load("benefits.json")
    cameras = load("cameras.json")
    zones = load("zones.json")

    DEFAULT_LIEU = "default"
    DEFAULT_SITE = "site_default"

    if DEFAULT_LIEU not in lieux:
        lieux[DEFAULT_LIEU] = {
            "name": "Default",
            "address": "",
            "description": "Auto-created during migration",
            "icon": "",
            "created_at": now,
        }
        print(f"[+] Lieu '{DEFAULT_LIEU}' created")

    if DEFAULT_SITE not in sites:
        sites[DEFAULT_SITE] = {
            "name": "Default Site",
            "lieu_id": DEFAULT_LIEU,
            "address": "",
            "description": "Auto-created during migration",
            "icon": "",
            "created_at": now,
        }
        print(f"[+] Site '{DEFAULT_SITE}' created")

    modified_cameras = False
    for cam_id, cam_data in cameras.items():
        if not cam_data.get("site_id"):
            cam_data["site_id"] = DEFAULT_SITE
            modified_cameras = True
            print(f"[~] Camera '{cam_id}' assigned to site '{DEFAULT_SITE}'")

    if modified_cameras:
        save("cameras.json", cameras)
        print("[*] cameras.json updated with site_id")

    for video_name, video_zones in zones.items():
        cam_id = video_name
        if video_name.startswith("camera:"):
            cam_id = video_name[len("camera:"):]

        for zone_name, zone_data in video_zones.items():
            ben_id = snake(f"migrated_{cam_id}_{zone_name}")
            if ben_id in benefits:
                continue
            polys = zone_data.get("polygons", [])
            if not polys:
                continue
            benefits[ben_id] = {
                "name": f"Zone {zone_name} (migrated)",
                "skill": "detection",
                "skill_item": "detection_presence",
                "categories": ["human::silhouette"],
                "camera_id": cam_id,
                "zone_polygons": polys,
                "active": True,
                "canvas": {},
                "created_at": now,
            }
            print(f"[+] Benefit '{ben_id}' created from zone '{zone_name}' on '{video_name}'")

    save("lieux.json", lieux)
    save("sites.json", sites)
    save("benefits.json", benefits)
    print(f"\n[OK] Migration complete: {len(lieux)} lieu(x), {len(sites)} site(s), {len(benefits)} benefit(s)")

if __name__ == "__main__":
    main()
