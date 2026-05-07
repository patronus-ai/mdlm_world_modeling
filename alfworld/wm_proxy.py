#!/usr/bin/env python3
"""Local proxy for SDAR World Model.

Accepts the payload format produced by alfworld_plugin / RunPod handler and
proxies to local lmdeploy / sglang. Listens on port 30000, backend on 30001.
"""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests

BACKEND_PORT = int(os.environ.get("WM_BACKEND_PORT", "30001"))
SERVE_PORT   = int(os.environ.get("WM_PORT", "30000"))
COMPLETIONS_URL = f"http://localhost:{BACKEND_PORT}/v1/chat/completions"
MODELS_URL      = f"http://localhost:{BACKEND_PORT}/v1/models"
DEFAULT_DIFFUSION_STEPS = 1
_MODEL_ID = None  # lazily resolved from the backend's /v1/models


def _resolve_model() -> str:
    global _MODEL_ID
    if _MODEL_ID is not None:
        return _MODEL_ID
    try:
        r = requests.get(MODELS_URL, timeout=5)
        r.raise_for_status()
        _MODEL_ID = r.json()["data"][0]["id"]
    except Exception:
        _MODEL_ID = "ANONYMOUS/SDAR_world_model_v2"  # fallback
    return _MODEL_ID


class WMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        inp = body.get("input", body)

        state = inp.get("state")
        action = inp.get("action")
        sys_prompt = inp.get("system_prompt") or None

        state_str  = state  if isinstance(state, str)  else json.dumps(state)
        action_str = action if isinstance(action, str) else json.dumps(action)
        user_content = f"State: {state_str}\n\nAction: {action_str}"

        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": user_content})

        steps = inp.get("diffusion_steps", inp.get("num_diffusion_steps", DEFAULT_DIFFUSION_STEPS))
        payload = {
            "model": _resolve_model(),
            "messages": msgs,
            "max_tokens": inp.get("max_tokens", 256),
            "temperature": inp.get("temperature", 0),
            "repetition_penalty": inp.get("repetition_penalty", 1.3),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # Only add diffusion steps for SDAR (lmdeploy), skip for vLLM
        if steps and steps > 0:
            payload["num_diffusion_steps"] = steps
        try:
            r = requests.post(COMPLETIONS_URL, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            out = {"generated_text": data["choices"][0]["message"]["content"] or "",
                   "usage": data.get("usage", {})}
        except Exception as e:
            out = {"generated_text": "", "error": str(e)}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a, **kw): pass


if __name__ == "__main__":
    print(f"WM proxy on port {SERVE_PORT}, backend at {COMPLETIONS_URL}")
    HTTPServer(("0.0.0.0", SERVE_PORT), WMHandler).serve_forever()
