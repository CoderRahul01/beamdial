# Contributing

Thanks for helping. Short version:

1. Open an issue first for anything bigger than a typo, so we agree on the approach.
2. Keep the library dependency-free. `beamdial/` must import with plain Python.
3. Add a test in `tests/test_beamdial.py` for any behavior change. Run `python -m pytest -q`.
4. Never commit API keys, tokens or raw provider responses that contain them.
5. New task sources need an automatic grader, and the grader must pass on gold answers before any API spend.

Good first contributions:

- `AsyncBeamDial` for async clients
- Streaming passthrough when no `verify` is set
- New task families: tool calling, JSON output, long context
- A `python -m beamdial check-provider` command that runs the probe for a user's own provider
