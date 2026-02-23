#!/usr/bin/env python3
"""Patch main.py to add 4 skills to SKILLS_CONFIG."""

# Voir docs/SPEC_MODAL_CREATION_BENEFICES.md
SKILLS_REPLACEMENT = '''SKILLS_CONFIG = {
    "skills": [
        {"key": "detection", "label": "Détection", "icon": "/static/assets_youn/SvIcons/SVGnew/Yclassify.svg", "items": [{"id": "detection_presence", "label": "Présence / Absence", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"}, {"id": "detection_linecross", "label": "Franchissement ligne", "icon": "/static/assets_youn/SvIcons/SVGnew/Ylinecross.svg"}]},
        {"key": "counting", "label": "Comptage", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycounting.svg", "items": [{"id": "counting_people", "label": "Comptage personnes", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycountingppl.svg"}, {"id": "counting_objects", "label": "Comptage objets", "icon": "/static/assets_youn/SvIcons/SVGnew/Ylinecross.svg"}]},
        {"key": "heatmap", "label": "Heatmap", "icon": "/static/assets_youn/SvIcons/SVGnew/Yheatmap.svg", "items": [{"id": "heatmap_density", "label": "Densité de flux", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg"}, {"id": "heatmap_presence", "label": "Heatmap présence", "icon": "/static/assets_youn/SvIcons/SVGnew/Yheatmapdense.svg"}, {"id": "heatmap_trajectory", "label": "Heatmap trajectoires", "icon": "/static/assets_youn/SvIcons/SVGnew/Ytraj.svg"}]},
        {"key": "quality", "label": "Qualité", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg", "items": [{"id": "quality_defect", "label": "Détection défauts", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg"}]}
    ],
    "categories_by_skill": {
        "detection": [
            {"key": "human", "label": "Humain", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg", "items": [{"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"}, {"id": "visage", "label": "Visage", "icon": "/static/assets_youn/SvIcons/SVGnew/Yface.svg"}, {"id": "foule", "label": "Foule", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg"}]},
            {"key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg", "items": [{"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"}, {"id": "velo", "label": "Vélo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"}]}
        ],
        "counting": [
            {"key": "human", "label": "Humain", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg", "items": [{"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"}]},
            {"key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg", "items": [{"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"}, {"id": "velo", "label": "Vélo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"}]}
        ],
        "heatmap": [
            {"key": "human", "label": "Humain", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg", "items": [{"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"}]},
            {"key": "object", "label": "Objet", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg", "items": [{"id": "encombrement", "label": "Encombrement", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg"}, {"id": "zone_encombrée", "label": "Zone encombrée", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg"}]}
        ],
        "quality": [
            {"key": "object", "label": "Objet", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg", "items": [{"id": "defaut", "label": "Défaut", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg"}, {"id": "fissure", "label": "Fissure", "icon": "/static/assets_youn/SvIcons/SVGnew/Yfissure.svg"}]}
        ]
    }
}
'''

def main():
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    import re
    pattern = r"SKILLS_CONFIG = \{[^}]+\"categories_by_skill\": \{[^}]+\}[^}]*\}"
    if not re.search(pattern, content, re.DOTALL):
        # Simpler pattern
        start = content.find("SKILLS_CONFIG = {")
        # Find the closing } of SKILLS_CONFIG (before @app.get)
        end = content.find("@app.get(\"/api/skills\")")
        if end == -1:
            end = content.find("@app.get(\"/api/skills\"")
        if end > 0:
            end = content.rfind("}", 0, end)  # last } before @app.get
        if start == -1 or end == -1 or end <= start:
            print("Could not find SKILLS_CONFIG block")
            return 1
        # end points to '}'; replace from start through that '}' (inclusive)
        new_content = content[:start] + SKILLS_REPLACEMENT + content[end + 1:]
        with open("main.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Patched main.py successfully")
        return 0
    return 1

if __name__ == "__main__":
    exit(main())
