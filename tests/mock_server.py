"""Tiny fake OpenAI-compatible server for offline tests. Reasoning tokens grow with effort; answers are
correct more often at higher effort, so the whole pipeline (collect -> grade -> analyze -> train) can be tested."""
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TASKS = {}
LVL = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        eff = body.get("reasoning_effort", "medium")
        task = TASKS.get(prompt)
        lvl = LVL[eff]
        diff = task.get("difficulty_hint", 3) if task else 3
        rng = random.Random(hash((prompt, eff)) & 0xFFFF)
        ok = rng.random() < min(0.97, 0.35 + 0.15 * lvl + 0.1 * (5 - diff))
        if task and task["family"] == "math":
            content = "So \\boxed{%s}" % (task["answer"] if ok else "123456789")
        elif task and task["source"] == "mbpp":
            content = "```python\nprint('stub')\n```"
        else:
            content = "```python\npass\n```"
        rt = int((300 + 900 * diff) * (1.8 ** lvl) * (0.6 + rng.random()))
        resp = {
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content, "reasoning_content": "thinking " * 5}}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": rt + 40, "total_tokens": rt + 40 + len(prompt) // 4,
                      "completion_tokens_details": {"reasoning_tokens": rt}},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    for line in open(sys.argv[1]):
        t = json.loads(line)
        TASKS[t["prompt"]] = t
    HTTPServer(("127.0.0.1", int(sys.argv[2])), H).serve_forever()
