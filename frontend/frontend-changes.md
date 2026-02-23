# Frontend Changes

## Feature: Dark / Light Theme Toggle

### Files Modified

| File | Change Summary |
|------|---------------|
| `style.css` | Added light theme CSS variables, body transition, and toggle button styles |
| `index.html` | Added theme toggle button element with sun/moon SVG icons |
| `script.js` | Added `initTheme`, `toggleTheme` functions and wired up the button |

---

### style.css

**Light theme variables** (`[data-theme="light"]` selector):
- `--background: #f8fafc` — very light gray page background
- `--surface: #ffffff` — white card/sidebar background
- `--surface-hover: #f1f5f9` — light hover state
- `--text-primary: #0f172a` — dark text for high contrast
- `--text-secondary: #475569` — medium slate for secondary text
- `--border-color: #e2e8f0` — soft light border
- `--assistant-message: #f1f5f9` — light bubble for assistant messages
- `--shadow: 0 4px 6px -1px rgba(0,0,0,0.1)` — lighter shadow
- `--welcome-bg: #eff6ff` — pale blue for welcome card
- Primary/hover/focus-ring colours unchanged (blue accent preserved)

**Body transition**: `transition: background-color 0.3s ease, color 0.3s ease` added so theme switch animates smoothly.

**`#themeToggle` button styles**:
- Fixed position: top-right (`top: 1rem; right: 1rem`)
- 40 × 40 px circle, matches sidebar border/surface colours
- Hover: scale up slightly + blue accent colour
- Focus: 3 px blue focus ring (accessible)
- Sun/moon icons use absolute positioning with `opacity` + `transform` transitions so only the correct icon is visible per theme

---

### index.html

- Added `<button id="themeToggle">` immediately inside `<body>`, before `.container`
- Contains two inline SVGs: `.icon-sun` (shown in dark mode) and `.icon-moon` (shown in light mode)
- `aria-label="Toggle light/dark theme"` and `title="Toggle theme"` for accessibility
- SVGs have `aria-hidden="true"` since the button label covers their meaning

---

### script.js

**`initTheme()`** — called on `DOMContentLoaded`:
- Reads `localStorage.getItem('theme')`
- If `'light'`, sets `data-theme="light"` on `<html>` so the saved preference is restored on page load

**`toggleTheme()`**:
- Checks current theme by reading `document.documentElement.getAttribute('data-theme')`
- Toggles `data-theme` attribute on `<html>` (add/remove `"light"`)
- Persists choice to `localStorage` under key `'theme'`
- Updates `aria-label` on the button to reflect the *next* action

**`setupEventListeners()`** updated to attach `click → toggleTheme` on `#themeToggle`.
