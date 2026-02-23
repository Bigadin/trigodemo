#!/usr/bin/env python3
"""Replace heatmap_density icon from Ycrowdfoule to grid-svgrepo-com in main.py"""
with open("main.py", "r", encoding="utf-8") as f:
    c = f.read()

old = 'heatmap_density", "label": "Densité de flux", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg"'
new = 'heatmap_density", "label": "Densité de flux", "icon": "/static/assets_youn/SvIcons/SVGnew/grid-svgrepo-com.svg"'

if old in c:
    c = c.replace(old, new)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(c)
    print("main.py: heatmap_density icon updated to grid-svgrepo-com.svg")
else:
    print("Pattern not found - may already be updated")
