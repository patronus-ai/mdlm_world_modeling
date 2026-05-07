#!/usr/bin/env python3
"""
Local proxy for SDAR World Model.
Accepts the same payload format as the RunPod handler and proxies to local SGLang.
Listens on port 30000, SGLang runs on port 30001.
"""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests

SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30001"))
SERVE_PORT = int(os.environ.get("WM_PORT", "30000"))
COMPLETIONS_URL = f"http://localhost:{SGLANG_PORT}/v1/chat/completions"
DEFAULT_DIFFUSION_STEPS = 4


class WMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        input_data = body.get("input", body)

        state = input_data.get("state")
        action = input_data.get("action")
        system_prompt = input_data.get("system_prompt") or None

        # Build messages in the same format as the RunPod handler
        # state and action can be either JSON strings or actual objects
        if isinstance(state, str):
            state_str = state
        else:
            state_str = json.dumps(state)

        if isinstance(action, str):
            action_str = action
        else:
            action_str = json.dumps(action)

        user_content = f"State: {state_str}\n\nAction: {action_str}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        diffusion_steps = input_data.get("diffusion_steps",
                          input_data.get("num_diffusion_steps", DEFAULT_DIFFUSION_STEPS))

        payload = {
            "model": "ANONYMOUS/SDAR_world_model_v2",
            "messages": messages,
            "max_tokens": input_data.get("max_tokens", 768),
            "temperature": input_data.get("temperature", 0),
            "repetition_penalty": input_data.get("repetition_penalty", 1.3),
            "num_diffusion_steps": diffusion_steps,
        }

        try:
            resp = requests.post(COMPLETIONS_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            result = {
                "generated_text": data["choices"][0]["message"]["content"] or "",
                "usage": data.get("usage", {}),
            }
        except Exception as e:
            result = {"generated_text": "", "error": str(e)}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    print(f"WM proxy listening on port {SERVE_PORT}, forwarding to SGLang at {COMPLETIONS_URL}")
    server = HTTPServer(("0.0.0.0", SERVE_PORT), WMHandler)
    server.serve_forever()
