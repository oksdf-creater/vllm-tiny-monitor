import unittest
from unittest.mock import patch

from app import Collector, metric_sum, parse_prometheus


class MetricsResponse:
    def __init__(self, text):
        self.body = text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class MetricsTest(unittest.TestCase):
    def test_parse_colon_and_labels(self):
        data = '# HELP x x\nvllm:prompt_tokens_total{model_name="qwen"} 120.0\n'
        samples = parse_prometheus(data)
        self.assertEqual(samples[0], ("vllm:prompt_tokens_total", {"model_name": "qwen"}, 120.0))

    def test_sum_workers(self):
        samples = parse_prometheus('vllm:generation_tokens_total{pid="1"} 7\nvllm:generation_tokens_total{pid="2"} 9')
        self.assertEqual(metric_sum(samples, "vllm:generation_tokens_total"), 16)

    def test_prefill_updates_at_first_token_then_trend_returns_to_zero(self):
        def metrics(running, generation, completed, tokens, seconds):
            return MetricsResponse(f"""
vllm:num_requests_running {running}
vllm:num_requests_waiting 0
vllm:generation_tokens_total {generation}
vllm:prompt_tokens_total {tokens}
vllm:request_prefill_kv_computed_tokens_sum 0
vllm:request_prefill_time_seconds_sum 0
vllm:prefill_completed_requests_total {completed}
vllm:prefill_completed_tokens_total {tokens}
vllm:prefill_completed_seconds_total {seconds}
""")

        responses = [metrics(0, 10, 0, 0, 0),
                     metrics(1, 10, 0, 0, 0),
                     metrics(1, 11, 1, 1304, 1),
                     metrics(1, 12, 1, 1304, 1)]
        collector = Collector()
        with patch("app.urllib.request.urlopen", side_effect=responses), \
                patch("app.time.time", side_effect=[100, 101, 102, 103]), \
                patch("app.gpu_stats", return_value=([], None)), \
                patch("app.server_stats", return_value=({}, None, None)):
            collector.collect()
            collector.collect()
            self.assertEqual(collector.latest["prefill_rate_state"], "processing")
            self.assertEqual(collector.latest["prefill_trend_rate"], 0)
            collector.collect()
            self.assertEqual(collector.latest["prefill_rate_state"], "final")
            self.assertEqual(collector.latest["prefill_rate"], 1304)
            self.assertEqual(collector.latest["prefill_trend_rate"], 1304)
            self.assertEqual(collector.latest["task_generation_tokens"], 1)
            self.assertEqual(collector.latest["task_avg_decode_rate"], 0)
            self.assertEqual(collector.latest["task_peak_decode_rate"], 0)
            collector.collect()
            self.assertEqual(collector.latest["prefill_rate"], 1304)
            self.assertEqual(collector.latest["prefill_trend_rate"], 0)
            self.assertEqual(collector.latest["task_generation_tokens"], 2)
            self.assertEqual(collector.latest["task_avg_decode_rate"], 1)
            self.assertEqual(collector.latest["task_peak_decode_rate"], 1)

    def test_legacy_vllm_falls_back_to_finished_prefill_metrics(self):
        def metrics(running, generation, prefill_tokens, prefill_time):
            return MetricsResponse(f"""
vllm:num_requests_running {running}
vllm:num_requests_waiting 0
vllm:generation_tokens_total {generation}
vllm:prompt_tokens_total {prefill_tokens}
vllm:request_prefill_kv_computed_tokens_sum {prefill_tokens}
vllm:request_prefill_time_seconds_sum {prefill_time}
""")

        responses = [metrics(0, 10, 1000, 1),
                     metrics(1, 10, 1000, 1),
                     metrics(0, 20, 2304, 2)]
        collector = Collector()
        with patch("app.urllib.request.urlopen", side_effect=responses), \
                patch("app.time.time", side_effect=[100, 101, 102]), \
                patch("app.gpu_stats", return_value=([], None)), \
                patch("app.server_stats", return_value=({}, None, None)):
            collector.collect()
            collector.collect()
            self.assertEqual(collector.latest["prefill_rate_state"], "processing")
            collector.collect()
            self.assertEqual(collector.latest["prefill_rate"], 1304)
            self.assertEqual(collector.latest["prefill_rate_state"], "final")
            self.assertEqual(collector.latest["task_generation_tokens"], 10)
            self.assertEqual(collector.latest["last_task_avg_decode_rate"], 10)
            self.assertEqual(collector.latest["last_task_peak_decode_rate"], 10)

    def test_average_cannot_exceed_peak_when_tokens_arrive_before_detection(self):
        def metrics(running, generation):
            return MetricsResponse(f"""
vllm:num_requests_running {running}
vllm:num_requests_waiting 0
vllm:generation_tokens_total {generation}
vllm:prompt_tokens_total 0
""")

        responses = [metrics(0, 0), metrics(1, 5), metrics(1, 15)]
        collector = Collector()
        with patch("app.urllib.request.urlopen", side_effect=responses), \
                patch("app.time.time", side_effect=[100, 101, 102]), \
                patch("app.gpu_stats", return_value=([], None)), \
                patch("app.server_stats", return_value=({}, None, None)):
            collector.collect()
            collector.collect()
            collector.collect()
            self.assertEqual(collector.latest["task_avg_decode_rate"], 10)
            self.assertEqual(collector.latest["task_peak_decode_rate"], 10)
            self.assertLessEqual(collector.latest["task_avg_decode_rate"],
                                 collector.latest["task_peak_decode_rate"])


if __name__ == "__main__":
    unittest.main()
