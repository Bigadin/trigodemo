#!/usr/bin/env python3
"""Patch transport config in main.py"""
path = "main.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '{"key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg", "items": [{"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"}, {"id": "velo", "label": "Vélo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"}]}'

new = '{"key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ytransport2.svg", "items": [{"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"}, {"id": "velo", "label": "Vélo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"}, {"id": "public_transport", "label": "Transport public", "icon": "/static/assets_youn/SvIcons/SVGnew/Ypublic%20transport.svg"}, {"id": "avion", "label": "Avion", "icon": "/static/assets_youn/SvIcons/SVGnew/plane-svgrepo-com.svg"}, {"id": "moto", "label": "Moto", "icon": "/static/assets_youn/SvIcons/SVGnew/motorcycle.svg"}]}'

count = content.count(old)
content = content.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"main.py: {count} occurrence(s) remplacée(s)")

# Patch skills-adapter.js
path2 = "static/js/skills-adapter.js"
with open(path2, "r", encoding="utf-8") as f:
    c = f.read()
c = c.replace(", { id: 'flux', label: 'Flux', icon: '/static/assets_youn/SvIcons/SVGnew/Yheatmapdense.svg' }", "")
c = c.replace("'Objet', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg', items: [{ id: 'fissure'", "'Objet', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg', items: [{ id: 'fissure'")
with open(path2, "w", encoding="utf-8") as f:
    f.write(c)
print("skills-adapter.js: flux supprimé, objet→Yobstruction")
