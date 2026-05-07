#!/usr/bin/env python3
"""HTTP server wrapping ScienceWorld environment for eval and reward computation. Port 30003.

Fixes vs v1:
- ThreadingHTTPServer for concurrent requests
- First POST with action="__init__" creates env without stepping (returns initial obs)
- Separate task_id per generation to avoid state corruption during GRPO
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from scienceworld import ScienceWorldEnv

_envs = {}  # task_id -> env
_lock = threading.Lock()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            task_id = body.get("task_id")
            action = body.get("action", "look around")

            if not task_id:
                self._respond({"error": "need task_id"})
                return

            with _lock:
                if task_id not in _envs:
                    # Parse task_id: "{task_name}_v{variation}" or "{task_name}_v{variation}_gen{N}"
                    base_id = task_id.split("_gen")[0]  # strip generation suffix
                    parts = base_id.rsplit("_v", 1)
                    task_name = parts[0]
                    variation = int(parts[1]) if len(parts) > 1 else 0
                    env = ScienceWorldEnv("")
                    env.load(task_name, variation, "easy")
                    obs, info = env.reset()
                    _envs[task_id] = env
                    # Fall through to execute the action (don't return early)

                env = _envs[task_id]

            # Step outside the lock (env.step can be slow)
            obs, reward, done, info = env.step(action)
            valid_actions = env.get_possible_actions()
            valid_objects = env.get_possible_objects()
            self._respond({
                "observation": obs,
                "score": info.get("score", 0.0),
                "reward": reward,
                "done": done,
                "valid_actions": valid_actions[:20],
                "valid_objects": valid_objects[:20],
            })
        except Exception as e:
            try:
                self._respond({"error": f"Server error: {e}"})
            except:
                pass

    def do_DELETE(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            task_id = body.get("task_id")
            with _lock:
                if task_id in _envs:
                    try:
                        _envs[task_id].close()
                    except:
                        pass
                    del _envs[task_id]
            self._respond({"status": "closed"})
        except Exception as e:
            try:
                self._respond({"error": str(e)})
            except:
                pass

    def _respond(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args): pass

if __name__ == "__main__":
    print("ScienceWorld env server on port 30003 (threaded)")
    ThreadingHTTPServer(("0.0.0.0", 30003), Handler).serve_forever()
