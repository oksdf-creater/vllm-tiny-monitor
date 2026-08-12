#!/usr/bin/env python3
"""Tiny, dependency-free dashboard for vLLM and NVIDIA GPUs."""

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VLLM_URL = os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
HOST = os.getenv("MONITOR_HOST", "0.0.0.0")
PORT = int(os.getenv("MONITOR_PORT", "8088"))
INTERVAL = max(float(os.getenv("POLL_INTERVAL", "2")), 0.5)
HISTORY_SIZE = max(int(os.getenv("HISTORY_SIZE", "900")), 10)
STATIC = Path(__file__).with_name("static")

SAMPLE_RE = re.compile(r'^([^\s{]+)(?:\{([^}]*)\})?\s+([-+\w.]+)(?:\s+\d+)?$')
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"])*)"')


def parse_prometheus(text):
    samples = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        labels = dict(LABEL_RE.findall(match.group(2) or ""))
        samples.append((match.group(1), labels, value))
    return samples


def metric_sum(samples, *names):
    wanted = set(names)
    values = [value for name, _, value in samples if name in wanted]
    return sum(values) if values else None


def metric_gauge(samples, *names):
    value = metric_sum(samples, *names)
    return value


def gpu_stats():
    fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=3, stderr=subprocess.DEVNULL)
        result = []
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 7:
                result.append({"index": int(parts[0]), "name": parts[1],
                    "util": float(parts[2]), "memory_used": float(parts[3]),
                    "memory_total": float(parts[4]), "temperature": float(parts[5]),
                    "power": None if parts[6] == "[N/A]" else float(parts[6])})
        return result, None
    except Exception as exc:
        return [], str(exc)


class Collector:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY_SIZE)
        self.latest = {"online": False, "error": "Waiting for first sample", "history": []}
        self.previous = None

    def collect(self):
        now = time.time()
        error = None
        try:
            req = urllib.request.Request(VLLM_URL, headers={"User-Agent": "vllm-tiny-monitor/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                samples = parse_prometheus(response.read().decode("utf-8", "replace"))
            prompt_total = metric_sum(samples, "vllm:prompt_tokens_total", "vllm_prompt_tokens_total")
            generation_total = metric_sum(samples, "vllm:generation_tokens_total", "vllm_generation_tokens_total")
            running = metric_gauge(samples, "vllm:num_requests_running", "vllm_num_requests_running")
            waiting = metric_gauge(samples, "vllm:num_requests_waiting", "vllm_num_requests_waiting")
            cache = metric_gauge(samples, "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
                                 "vllm_gpu_cache_usage_perc", "vllm_kv_cache_usage_perc")
            success = metric_sum(samples, "vllm:request_success_total", "vllm_request_success_total")
            prefill_rate = decode_rate = 0.0
            if self.previous:
                elapsed = max(now - self.previous[0], 0.001)
                if prompt_total is not None and self.previous[1] is not None:
                    prefill_rate = max(0.0, prompt_total - self.previous[1]) / elapsed
                if generation_total is not None and self.previous[2] is not None:
                    decode_rate = max(0.0, generation_total - self.previous[2]) / elapsed
            self.previous = (now, prompt_total, generation_total)
            online = True
        except Exception as exc:
            prompt_total = generation_total = running = waiting = cache = success = None
            prefill_rate = decode_rate = 0.0
            online, error = False, str(exc)

        gpus, gpu_error = gpu_stats()
        point = {"time": int(now * 1000), "prefill_rate": round(prefill_rate, 2),
                 "decode_rate": round(decode_rate, 2)}
        with self.lock:
            self.history.append(point)
            self.latest = {"online": online, "error": error, "source": VLLM_URL,
                "updated_at": int(now * 1000), "prefill_rate": point["prefill_rate"],
                "decode_rate": point["decode_rate"], "prompt_tokens_total": prompt_total,
                "generation_tokens_total": generation_total,
                "tokens_total": None if prompt_total is None or generation_total is None else prompt_total + generation_total,
                "requests_running": running, "requests_waiting": waiting,
                "requests_finished": success, "kv_cache_usage": None if cache is None else cache * 100,
                "gpus": gpus, "gpu_error": gpu_error}

    def snapshot(self):
        with self.lock:
            return {**self.latest, "history": list(self.history)}


collector = Collector()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/status":
            body = json.dumps(collector.snapshot(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        if self.path != "/api/status":
            super().log_message(fmt, *args)


def poll():
    while True:
        started = time.monotonic()
        collector.collect()
        time.sleep(max(0.1, INTERVAL - (time.monotonic() - started)))


if __name__ == "__main__":
    threading.Thread(target=poll, daemon=True).start()
    print(f"vLLM Monitor: http://{HOST}:{PORT}  source={VLLM_URL}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
