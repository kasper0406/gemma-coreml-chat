# AI coding assistants

Applies to **Cursor, GitHub Copilot, Claude, Gemini**, and similar tools working in this repository.

## Solo-iteration defaults

1. **Minimal surface area** — Prefer fewer CLI options, environment variables, config fields, and `**kwargs` bridges. One clear code path.

2. **No compatibility toggles unless asked** — Avoid legacy modes, dual implementations (e.g. Python fallback + in-graph behavior), version switches, and deprecation shims for hypothetical downstream users.

3. **Breaking changes are acceptable** — Remove or rename APIs freely; do not preserve old signatures or flags “for compatibility.”

If something needs a newer export, model, or setup, document that requirement instead of adding runtime switches.

See also: `.cursor/rules/minimal-api-surface.mdc` (same intent, Cursor-native).
