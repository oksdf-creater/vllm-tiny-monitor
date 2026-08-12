# vLLM Tiny Monitor

面向单机 vLLM 的轻量实时面板：无需 pip、Node、数据库或 Docker，直接读取 vLLM 的 Prometheus `/metrics`，并通过 `nvidia-smi` 展示双卡状态。

## 显示内容

- 最新 Prefill 吞吐（prompt tokens/s）与输出吞吐（generation tokens/s）
- 累计输入、输出及总 Token 数
- 运行中/等待中/已完成请求、KV Cache 使用率
- 每张 NVIDIA GPU 的利用率、显存、温度和功耗
- 最近约 30 分钟吞吐趋势（默认每 2 秒采样、内存保存 900 点）

同时兼容 `vllm:xxx` 和 `vllm_xxx` 两种 Prometheus 指标命名。速率由相邻两次累计计数差计算；服务刚启动时需等待一个采样周期。

## Ubuntu 直接运行

先确认 vLLM 指标可访问：

```bash
curl http://127.0.0.1:8000/metrics | grep -E 'prompt_tokens|generation_tokens' | head
```

启动面板：

```bash
cd vllm-monitor
VLLM_METRICS_URL=http://127.0.0.1:8000/metrics python3 app.py
```

浏览器打开 `http://服务器IP:8088`。若 vLLM 在 Docker 容器内，请将 URL 改成宿主机可访问的 vLLM 地址。

可用环境变量：`VLLM_METRICS_URL`、`MONITOR_HOST`、`MONITOR_PORT`、`POLL_INTERVAL`、`HISTORY_SIZE`。

## 注册为 systemd 服务

```bash
sudo mkdir -p /opt/vllm-monitor
sudo cp -r app.py static /opt/vllm-monitor/
sed "s/CHANGE_ME/$USER/" vllm-monitor.service | sudo tee /etc/systemd/system/vllm-monitor.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-monitor
sudo systemctl status vllm-monitor
```

如启用了 UFW：`sudo ufw allow 8088/tcp`。建议仅在可信局域网开放；若暴露到公网，应在前面加带认证的 Nginx/Caddy。

## 指标说明

“Prefill 速度”是整台服务每秒处理的 prompt token 数；“当前输出速度”是所有并发请求合计的 generation token/s，并非单条请求速度。计数器在 vLLM 重启后归零，面板会避免显示负速率。
