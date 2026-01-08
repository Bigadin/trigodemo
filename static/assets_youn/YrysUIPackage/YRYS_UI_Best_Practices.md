# YRYS — UI / Design System Best Practices (Cursor Guide)
Version: 1.0  
Scope: Front-end UI implementation + design consistency rules for the YRYS product.

> Objective: help Cursor (and any contributor) produce UI code that is visually consistent with the YRYS brand guidelines, maintainable, theme-ready (light/dark), and accessible.

---

## 1) Non‑negotiables
1. **No hardcoded colors in components.** Use design tokens (CSS variables) only.
2. **One typography family:** Manrope (with a small, explicit scale of sizes/weights).
3. **Consistent spacing & radii:** use the shared scale; do not invent ad-hoc values.
4. **Every interactive element has states:** default / hover / active / disabled / focus.
5. **Accessibility is part of “done”:** focus visible, contrast, hit targets.

---

## 2) Brand color system (source-of-truth)
### 2.1 Primary colors (brand accents)
Use these as *accent/brand* colors. Do not use all of them at once in a single screen unless explicitly designed.

- **Primary 01 — Orange:** `#F08321`  
- **Primary 02 — Cyan:** `#10B0F9`  
- **Primary 03 — Blue:** `#062DB6`  
- **Primary 04 — Purple:** `#BD44D5`  

Recommended roles (default mapping):
- `#10B0F9` (Cyan): primary CTA / active states / links
- `#062DB6` (Blue): secondary CTA / emphasis / “serious” actions
- `#F08321` (Orange): highlights, warnings-like emphasis (not error), metrics callouts
- `#BD44D5` (Purple): special accent (badges, featured items, highlights)

### 2.2 Grayscale palette (UI structure)
Use grays for the majority of the interface (backgrounds, surfaces, borders, text).

- Cloud: `#EDEFF7`
- Smoke: `#D3D6E0`
- Steel: `#BCBFCC`
- Space: `#9DA2B3`
- Graphite: `#6E7180`
- Arsenic: `#40424D`
- Phantom: `#1E1E24`
- Black: `#000000`

---

## 3) Themes (Light / Dark)
### 3.1 Principle
- **Light theme:** lighter surfaces, darker text.
- **Dark theme:** dark surfaces, lighter text, keep borders subtle (avoid high-contrast borders everywhere).

### 3.2 Token-first approach
Define semantic tokens, then map them to palette values per theme.

#### Example CSS variables (recommended baseline)
```css
:root {
  /* Typography */
  --font-sans: "Manrope", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;

  /* Spacing scale */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;

  /* Radius */
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;

  /* Brand accents */
  --brand-01: #F08321;
  --brand-02: #10B0F9;
  --brand-03: #062DB6;
  --brand-04: #BD44D5;

  /* Grays */
  --gray-01: #EDEFF7; /* Cloud */
  --gray-02: #D3D6E0; /* Smoke */
  --gray-03: #BCBFCC; /* Steel */
  --gray-04: #9DA2B3; /* Space */
  --gray-05: #6E7180; /* Graphite */
  --gray-06: #40424D; /* Arsenic */
  --gray-07: #1E1E24; /* Phantom */
  --gray-08: #000000; /* Black */
}

/* Light theme */
:root[data-theme="light"] {
  --color-bg: var(--gray-01);
  --color-surface: #FFFFFF;
  --color-surface-2: var(--gray-01);
  --color-border: var(--gray-02);

  --color-text: var(--gray-07);
  --color-text-muted: var(--gray-05);

  --color-primary: var(--brand-02);
  --color-primary-hover: color-mix(in srgb, var(--brand-02) 88%, #000 12%);
  --color-primary-pressed: color-mix(in srgb, var(--brand-02) 78%, #000 22%);

  --color-focus: var(--brand-02);
  --color-link: var(--brand-02);
}

/* Dark theme */
:root[data-theme="dark"] {
  --color-bg: #0B0F14;         /* recommended deep background */
  --color-surface: #111825;    /* elevated surface */
  --color-surface-2: #0E141F;  /* secondary surface */
  --color-border: color-mix(in srgb, var(--gray-07) 55%, #000 45%);

  --color-text: #F5F7FF;
  --color-text-muted: color-mix(in srgb, #F5F7FF 65%, #000 35%);

  --color-primary: var(--brand-02);
  --color-primary-hover: color-mix(in srgb, var(--brand-02) 88%, #fff 12%);
  --color-primary-pressed: color-mix(in srgb, var(--brand-02) 78%, #fff 22%);

  --color-focus: var(--brand-02);
  --color-link: var(--brand-02);
}
```

Notes:
- If `color-mix()` is not supported in your stack, precompute hover/pressed colors in tokens.
- Keep **semantic tokens** stable; only theme mappings should change.

---

## 4) Typography (Manrope)
### 4.1 Rules
- Use **Manrope** for all UI text.
- Keep a **small** set of sizes; avoid “random” font sizes.

### 4.2 Suggested type scale
| Token | Size | Line-height | Use |
|---|---:|---:|---|
| `--text-xs` | 12px | 16px | captions, helper text |
| `--text-sm` | 14px | 20px | secondary text |
| `--text-md` | 16px | 24px | body default |
| `--text-lg` | 18px | 28px | section lead |
| `--text-xl` | 24px | 32px | page titles |
| `--text-2xl` | 32px | 40px | hero titles (rare) |

Weights:
- 400 Regular (body)
- 500 Medium (UI labels)
- 600 SemiBold (titles)
- 700 Bold (rare, emphasis)

---

## 5) Layout & spacing
### 5.1 Spacing scale
Only use: 4 / 8 / 12 / 16 / 24 / 32 / 48 px (via tokens).

### 5.2 Grid and density
- Prefer a **12-column grid** on desktop; use consistent page margins.
- Avoid overly tight UI: minimum vertical rhythm of **8px** increments.
- For B2B dashboards: support a “comfortable” density (default) and optionally a “compact” mode (future).

---

## 6) Components: baseline specifications
### 6.1 Buttons
- Primary button uses `--color-primary`.
- Height: 40px (desktop), 44–48px (touch contexts).
- Radius: `--radius-md`.
- States:
  - Hover: `--color-primary-hover`
  - Pressed: `--color-primary-pressed`
  - Disabled: reduce opacity + remove shadow + disable pointer events
  - Focus: visible ring using `--color-focus` (2px)

### 6.2 Inputs
- Use `--color-surface` background.
- Border uses `--color-border`.
- Focus: border + ring (do not rely on color alone; ring is mandatory).
- Error state: add an explicit error token (do not reuse orange).

### 6.3 Cards / surfaces
- Use subtle elevation (shadow) sparingly.
- Prefer **surface contrast** (surface vs bg) over heavy shadows.
- Borders should be light in light theme, subtle in dark theme.

### 6.4 Modals / dialogs
- Overlay: dark translucent layer (tokenized).
- Modal surface uses `--color-surface`.
- Close affordance always visible and reachable via keyboard.

---

## 7) Textures (backgrounds)
Textures are allowed as a **brand signature**, but must not harm readability.

Rules:
- Use textures only on **background layers** (hero, section backgrounds, splash screens).
- Recommended opacity range: **4% → 12%**.
- Do not place textures behind dense tables, forms, or code blocks.
- Prefer a clean “surface” card on top of textured backgrounds.

Implementation hint:
- Apply texture via `background-image` on container, then keep content in a solid surface layer.

---

## 8) Iconography
- Use a **24px grid** for UI icons.
- Keep consistent stroke weight (e.g., 1.5px or 2px; choose one).
- Align optically (centers, vertical alignment in buttons).
- Do not mix radically different icon styles in the same UI.

---

## 9) Accessibility checklist (minimum)
- Focus visible on all interactive elements (keyboard).
- Minimum hit target: 44×44px where relevant.
- Contrast: aim for WCAG AA (especially body text).
- Error states: use **text + icon + color** (not color only).

---

## 10) “Cursor instructions” (how to generate UI code here)
When producing or modifying UI code for YRYS, Cursor must:
1. Use **semantic tokens** (`--color-*`, `--space-*`, `--radius-*`, typography tokens).
2. Avoid introducing new hex colors unless:
   - They are explicitly added to the token set, and
   - There is a clear semantic purpose (e.g., error/success).
3. Keep components consistent:
   - Same button heights, radii, and spacing across the app.
4. Implement full component state coverage (hover/active/focus/disabled).
5. Validate dark mode (no illegible text, no harsh borders).

PR review checklist:
- [ ] No hardcoded colors outside the token file
- [ ] Consistent spacing via tokens
- [ ] Focus ring present
- [ ] Hover/pressed/disabled states implemented
- [ ] Dark mode checked on key screens
- [ ] Text remains readable on textured backgrounds

---

## 11) Project structure (recommended)
- `src/styles/tokens.css` (all variables)
- `src/styles/theme.css` (light/dark mappings)
- `src/components/ui/*` (atomic UI components)
- `src/styles/textures/*` (light/dark textures)
- `docs/ui/YRYS_UI_Best_Practices.md` (this file)

---

## 12) Change control
Any change to tokens must be:
- documented in a short changelog section (date + reason),
- validated in both themes,
- reviewed with at least one UI owner (design/dev).
