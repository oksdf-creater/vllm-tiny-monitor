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
INTERVAL = max(float(os.getenv("POLL_INTERVAL", "1")), 0.5)
ACTIVE_INTERVAL = max(float(os.getenv("ACTIVE_POLL_INTERVAL", "0.5")), 0.2)
GPU_INTERVAL = max(float(os.getenv("GPU_POLL_INTERVAL", "2")), INTERVAL)
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
    fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=3, stderr=subprocess.DEVNULL)
        result = []
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 8:
                result.append({"index": int(parts[0]), "name": parts[1],
                    "util": float(parts[2]), "memory_used": float(parts[3]),
                    "memory_total": float(parts[4]), "temperature": float(parts[5]),
                    "power": None if parts[6] == "[N/A]" else float(parts[6]),
                    "power_limit": None if parts[7] == "[N/A]" else float(parts[7])})
        return result, None
    except Exception as exc:
        return [], str(exc)


def server_stats(previous=None):
    """Read lightweight host counters from procfs and calculate utilization rates."""
    try:
        cpu = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        cpu_values = [int(value) for value in cpu]
        cpu_total = sum(cpu_values)
        cpu_idle = cpu_values[3] + (cpu_values[4] if len(cpu_values) > 4 else 0)

        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.split()[0])
        memory_total = memory["MemTotal"] * 1024
        memory_used = memory_total - memory.get("MemAvailable", memory.get("MemFree", 0)) * 1024

        network_rx = network_tx = 0
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            interface, values = line.split(":", 1)
            if interface.strip() == "lo":
                continue
            fields = values.split()
            network_rx += int(fields[0])
            network_tx += int(fields[8])

        block_devices = {path.name for path in Path("/sys/block").iterdir()
                         if not path.name.startswith(("loop", "ram"))}
        disk_read = disk_write = 0
        for line in Path("/proc/diskstats").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 10 and fields[2] in block_devices:
                disk_read += int(fields[5]) * 512
                disk_write += int(fields[9]) * 512

        now = time.monotonic()
        cpu_percent = network_rx_rate = network_tx_rate = disk_read_rate = disk_write_rate = 0.0
        if previous:
            elapsed = max(now - previous[0], 0.001)
            total_delta = cpu_total - previous[1]
            idle_delta = cpu_idle - previous[2]
            if total_delta > 0:
                cpu_percent = 100 * max(0, total_delta - idle_delta) / total_delta
            network_rx_rate = max(0, network_rx - previous[3]) / elapsed
            network_tx_rate = max(0, network_tx - previous[4]) / elapsed
            disk_read_rate = max(0, disk_read - previous[5]) / elapsed
            disk_write_rate = max(0, disk_write - previous[6]) / elapsed
        counters = (now, cpu_total, cpu_idle, network_rx, network_tx, disk_read, disk_write)
        stats = {"cpu_percent": round(cpu_percent, 1),
                 "memory_percent": round(100 * memory_used / memory_total, 1),
                 "memory_used": memory_used, "memory_total": memory_total,
                 "network_rx_rate": round(network_rx_rate, 1),
                 "network_tx_rate": round(network_tx_rate, 1),
                 "disk_read_rate": round(disk_read_rate, 1),
                 "disk_write_rate": round(disk_write_rate, 1)}
        return stats, counters, None
    except Exception as exc:
        return {}, previous, str(exc)


class Collector:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY_SIZE)
        self.latest = {"online": False, "error": "Waiting for first sample", "history": []}
        self.previous = None
        self.prefill_rate = 0.0
        self.prefill_rate_state = "waiting"
        self.task_active = False
        self.task_started_at = None
        self.task_duration = None
        self.task_start_generation = None
        self.task_generated_tokens = None
        self.task_avg_decode_rate = None
        self.gpus = []
        self.gpu_error = "Waiting for first sample"
        self.last_gpu_poll = 0.0
        self.server_previous = None

    def collect(self):
        now = time.time()
        error = None
        trend_prefill_rate = 0.0
        try:
            req = urllib.request.Request(VLLM_URL, headers={"User-Agent": "vllm-tiny-monitor/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                samples = parse_prometheus(response.read().decode("utf-8", "replace"))
            prompt_total = metric_sum(samples, "vllm:prompt_tokens_total", "vllm_prompt_tokens_total")
            generation_total = metric_sum(samples, "vllm:generation_tokens_total", "vllm_generation_tokens_total")
            prefill_tokens_total = metric_sum(
                samples, "vllm:request_prefill_kv_computed_tokens_sum",
                "vllm_request_prefill_kv_computed_tokens_sum")
            prefill_time_total = metric_sum(
                samples, "vllm:request_prefill_time_seconds_sum",
                "vllm_request_prefill_time_seconds_sum")
            completed_prefill_requests = metric_sum(
                samples, "vllm:prefill_completed_requests_total",
                "vllm_prefill_completed_requests_total")
            completed_prefill_tokens = metric_sum(
                samples, "vllm:prefill_completed_tokens_total",
                "vllm_prefill_completed_tokens_total")
            completed_prefill_seconds = metric_sum(
                samples, "vllm:prefill_completed_seconds_total",
                "vllm_prefill_completed_seconds_total")
            running = metric_gauge(samples, "vllm:num_requests_running", "vllm_num_requests_running")
            waiting = metric_gauge(samples, "vllm:num_requests_waiting", "vllm_num_requests_waiting")
            cache = metric_gauge(samples, "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
                                 "vllm_gpu_cache_usage_perc", "vllm_kv_cache_usage_perc")
            success = metric_sum(samples, "vllm:request_success_total", "vllm_request_success_total")
            decode_rate = 0.0
            active = (running or 0) + (waiting or 0) > 0
            if active and not self.task_active:
                self.task_started_at = now
                self.task_duration = 0.0
                self.task_start_generation = (
                    self.previous["generation"] if self.previous is not None
                    and self.previous["generation"] is not None else generation_total)
                self.task_generated_tokens = 0.0
                self.task_avg_decode_rate = 0.0
                self.prefill_rate = 0.0
                self.prefill_rate_state = "processing"
            if self.previous:
                elapsed = max(now - self.previous["time"], 0.001)
                generation_delta = 0.0
                if generation_total is not None and self.previous["generation"] is not None:
                    generation_delta = max(0.0, generation_total - self.previous["generation"])
                    decode_rate = generation_delta / elapsed
                has_live_prefill_metrics = completed_prefill_requests is not None
                if (has_live_prefill_metrics
                        and self.previous["completed_prefill_requests"] is not None
                        and completed_prefill_tokens is not None
                        and self.previous["completed_prefill_tokens"] is not None
                        and completed_prefill_seconds is not None
                        and self.previous["completed_prefill_seconds"] is not None):
                    request_delta = (completed_prefill_requests
                                     - self.previous["completed_prefill_requests"])
                    token_delta = (completed_prefill_tokens
                                   - self.previous["completed_prefill_tokens"])
                    time_delta = (completed_prefill_seconds
                                  - self.previous["completed_prefill_seconds"])
                    if request_delta > 0 and token_delta > 0 and time_delta > 0:
                        self.prefill_rate = token_delta / time_delta
                        trend_prefill_rate = self.prefill_rate
                        self.prefill_rate_state = "final"
                elif (not has_live_prefill_metrics
                        and prefill_tokens_total is not None
                        and prefill_time_total is not None
                        and self.previous["prefill_tokens"] is not None
                        and self.previous["prefill_time"] is not None):
                    token_delta = prefill_tokens_total - self.previous["prefill_tokens"]
                    time_delta = prefill_time_total - self.previous["prefill_time"]
                    if token_delta > 0 and time_delta > 0:
                        self.prefill_rate = token_delta / time_delta
                        trend_prefill_rate = self.prefill_rate
                        self.prefill_rate_state = "final"
            if active and self.task_started_at is not None:
                self.task_duration = now - self.task_started_at
            elif self.task_active and self.task_started_at is not None:
                self.task_duration = now - self.task_started_at
            if (self.task_duration is not None and self.task_duration > 0
                    and generation_total is not None
                    and self.task_start_generation is not None):
                generated_for_task = max(
                    0.0, generation_total - self.task_start_generation)
                self.task_generated_tokens = generated_for_task
                self.task_avg_decode_rate = generated_for_task / self.task_duration
            self.task_active = active
            self.previous = {"time": now, "generation": generation_total,
                             "prefill_tokens": prefill_tokens_total,
                             "prefill_time": prefill_time_total,
                             "completed_prefill_requests": completed_prefill_requests,
                             "completed_prefill_tokens": completed_prefill_tokens,
                             "completed_prefill_seconds": completed_prefill_seconds}
            online = True
        except Exception as exc:
            prompt_total = generation_total = running = waiting = cache = success = None
            decode_rate = 0.0
            online, error = False, str(exc)

        if now - self.last_gpu_poll >= GPU_INTERVAL:
            self.gpus, self.gpu_error = gpu_stats()
            self.last_gpu_poll = now
        gpus, gpu_error = self.gpus, self.gpu_error
        server, self.server_previous, server_error = server_stats(self.server_previous)
        point = {"time": int(now * 1000), "prefill_rate": round(trend_prefill_rate, 2),
                 "decode_rate": round(decode_rate, 2)}
        with self.lock:
            self.history.append(point)
            self.latest = {"online": online, "error": error, "source": VLLM_URL,
                "updated_at": int(now * 1000), "prefill_rate": round(self.prefill_rate, 2),
                "prefill_trend_rate": point["prefill_rate"],
                "prefill_rate_state": self.prefill_rate_state,
                "decode_rate": point["decode_rate"], "prompt_tokens_total": prompt_total,
                "generation_tokens_total": generation_total,
                "task_generation_tokens": self.task_generated_tokens,
                "tokens_total": None if prompt_total is None or generation_total is None else prompt_total + generation_total,
                "requests_running": running, "requests_waiting": waiting,
                "requests_finished": success, "kv_cache_usage": None if cache is None else cache * 100,
                "task_active": self.task_active,
                "task_duration_seconds": None if self.task_duration is None else round(self.task_duration, 1),
                "task_avg_decode_rate": (None if self.task_avg_decode_rate is None
                                         else round(self.task_avg_decode_rate, 2)),
                "gpus": gpus, "gpu_error": gpu_error,
                "server": server, "server_error": server_error}

    def snapshot(self):
        with self.lock:
            return {**self.latest, "history": list(self.history)}

    def poll_interval(self):
        return ACTIVE_INTERVAL if self.task_active else INTERVAL


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
        time.sleep(max(0.1, collector.poll_interval() - (time.monotonic() - started)))


if __name__ == "__main__":
    threading.Thread(target=poll, daemon=True).start()
    print(f"vLLM Monitor: http://{HOST}:{PORT}  source={VLLM_URL}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
