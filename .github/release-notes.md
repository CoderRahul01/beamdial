First release.

**Finding.** On 410 standard math and code tasks run at all five of Beam's reasoning levels (2,050 graded runs), `low` ties `max` on accuracy (96.3% vs 96.6%, exact McNemar p = 1.0) and uses 55% of the tokens.

**Library**
- `BeamDial(client)` wraps any OpenAI-compatible client and picks `reasoning_effort` per request
- escalates one level only when your own check fails
- retries cut-off replies at the same level with more room
- treats tool calls as success
- `to_message()` always carries the `reasoning_content` Beam requires on follow-up turns
- `beamdial calibrate prompts.jsonl` measures the gap on your own prompts
- no runtime dependencies, router weights are under 1 MB

**Data and demo**
- dataset: https://huggingface.co/datasets/PengoSword123/beamdial-v0.1
- demo (router runs in your browser): https://huggingface.co/spaces/PengoSword123/beamdial

**Limits.** Standard benchmarks that Beam may have seen in training, one provider (Baseten), one sample per cell. Frontier math and fresh long-context tasks are next.
