# Contributing

1. Open an issue first for anything bigger than a typo.
2. Keep the library dependency-free. `beamdial/` must import with plain Python.
3. Add a test for any behavior change and run `python -m pytest -q`.
4. If you touch `features.py` or `router.py`, update `space_static/router.js` too and run `python scripts/check_js_parity.py`.
5. Never commit API keys or tokens.
6. New task sources need an automatic grader that passes on the gold answers before any API spend.

Good first contributions: an async client, streaming when no `verify` is set, tool-calling and JSON task sets.
