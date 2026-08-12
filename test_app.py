import unittest
from app import parse_prometheus, metric_sum


class MetricsTest(unittest.TestCase):
    def test_parse_colon_and_labels(self):
        data = '# HELP x x\nvllm:prompt_tokens_total{model_name="qwen"} 120.0\n'
        samples = parse_prometheus(data)
        self.assertEqual(samples[0], ("vllm:prompt_tokens_total", {"model_name": "qwen"}, 120.0))

    def test_sum_workers(self):
        samples = parse_prometheus('vllm:generation_tokens_total{pid="1"} 7\nvllm:generation_tokens_total{pid="2"} 9')
        self.assertEqual(metric_sum(samples, "vllm:generation_tokens_total"), 16)


if __name__ == "__main__":
    unittest.main()
