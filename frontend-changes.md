# Frontend Changes

## Code Quality Tooling — Prettier Setup

### Summary
Added **Prettier** as the opinionated code formatter for the frontend (the frontend-equivalent of `black` for Python). All three frontend source files were reformatted to be consistent with the Prettier configuration.

---

### New Files

| File | Purpose |
|---|---|
| `frontend/package.json` | Defines `format`, `format:check`, and `lint` npm scripts; declares Prettier as a dev dependency |
| `frontend/.prettierrc` | Prettier configuration: 2-space indent, 80-char print width, double quotes, LF line endings, trailing commas in ES5 positions |
| `frontend/.prettierignore` | Excludes `node_modules/` from formatting |
| `frontend/format.sh` | Shell script for running quality checks without needing to remember npm commands |

---

### Modified Files

#### `frontend/index.html`
- Indentation changed from 4 spaces to 2 spaces throughout
- Self-closing void elements (`<meta>`, `<link>`, `<input>`) now use the `/>` form as required by Prettier's HTML strict whitespace mode
- Long `<button data-question="...">` attributes broken to multiple lines to respect the 80-character print width

#### `frontend/script.js`
- All string literals switched from single quotes (`'`) to double quotes (`"`) for Prettier consistency
- Indentation changed from 4 spaces to 2 spaces throughout
- Trailing commas added in function arguments and object literals where Prettier's `trailingComma: "es5"` applies (e.g., `JSON.stringify({...},)`, arrow function parameters)
- Blank double-lines collapsed to single blank lines
- Arrow functions and callbacks reformatted for consistent spacing

#### `frontend/style.css`
- Indentation changed from 4 spaces to 2 spaces throughout
- Multi-selector rules (`*::before, *::after`) each placed on their own line
- Single-line rules (e.g., `h1 { font-size: 1.5rem; }`) expanded to multi-line blocks
- `@keyframes bounce` selector list (`0%, 80%, 100%`) reformatted to Prettier's preferred layout
- Font stack long value wrapped to a second line to stay within print width

---

### How to Use

**Check formatting (CI / pre-commit):**
```bash
cd frontend
npm install          # first time only
npm run format:check
# or
./format.sh check
```

**Auto-format all files:**
```bash
cd frontend
npm run format
# or
./format.sh format
```
