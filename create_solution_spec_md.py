#!/usr/bin/env python3
"""Create docs/SOLUTION_SPEC.md from solution-spec.json"""
import json

with open("static/config/solution-spec.json", "r", encoding="utf-8") as f:
    spec = json.load(f)

lines = [
    "# Spécification centrale — Solution YRYS",
    "",
    "> **Source de vérité** : `static/config/solution-spec.json`",
    "",
    "---",
    "",
    "## 1. Les 4 skills",
    "",
    "| Skill | Label | Icône |",
    "|-------|-------|-------|",
]
for s in spec["skills"]:
    lines.append(f"| **{s['key']}** | {s['label']} | `{s['icon']}` |")

lines.extend([
    "",
    "---",
    "",
    "## 2. Types de skill (sous-skills)",
    "",
])
for s in spec["skills"]:
    lines.append(f"### {s['label']}")
    lines.append("| id | label | icône |")
    lines.append("|----|-------|-------|")
    for item in s["items"]:
        lines.append(f"| `{item['id']}` | {item['label']} | `{item['icon']}` |")
    lines.append("")

lines.extend([
    "---",
    "",
    "## 3. Les 3 catégories principales",
    "",
    "| key | label | Icône catégorie |",
    "|-----|-------|-----------------|",
])
for c in spec["categories"]:
    lines.append(f"| **{c['key']}** | {c['label']} | `{c['icon']}` |")

lines.extend([
    "",
    "---",
    "",
    "## 4. Sous-catégories par catégorie",
    "",
])
for c in spec["categories"]:
    lines.append(f"### {c['label']}")
    lines.append("| id | label | icône |")
    lines.append("|----|-------|-------|")
    for item in c["items"]:
        lines.append(f"| `{item['id']}` | {item['label']} | `{item['icon']}` |")
    lines.append("")

lines.extend([
    "---",
    "",
    "## 5. Mapping catégories par skill",
    "",
    "| Skill | Catégories |",
    "|-------|------------|",
])
for skill, cats in spec["categories_by_skill"].items():
    lines.append(f"| {skill} | {', '.join(cats)} |")

lines.extend([
    "",
    "---",
    "",
    "## 6. Fichiers à synchroniser",
    "",
] + [f"- `{f}`" for f in spec.get("synced_files", [])] + [
    "",
])

with open("docs/SOLUTION_SPEC.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("docs/SOLUTION_SPEC.md created")
