#!/usr/bin/env python3
"""Update counting human items in main.py SKILLS_CONFIG."""
path = "main.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '"items": [{"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"}]},\n            {"key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg", "items": [{"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"}, {"id": "velo", "label": "Vélo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"}]}\n        ],\n        "heatmap":'

new = '"items": [{"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"}, {"id": "visage", "label": "Visage", "icon": "/static/assets_youn/SvIcons/SVGnew/Yface.svg"}, {"id": "foule", "label": "Foule", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg"}]},\n            {"key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg", "items": [{"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"}, {"id": "velo", "label": "Vélo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"}]}\n        ],\n        "heatmap":'

if old in content:
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("OK - counting human updated")
else:
    print("Pattern not found - checking...")
    if '"counting":' in content and 'silhouette' in content:
        print("Counting section exists, trying alternate pattern")
    else:
        print("Content check failed")
