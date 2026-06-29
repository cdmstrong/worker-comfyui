#!/usr/bin/env python3
"""
Runpod worker-comfyui 端到端测试脚本。

用法:
  # 使用 test_input.json 中的 workflow（默认）
  python test_request.py

  # 指定 workflow 文件
  python test_request.py --workflow test_resources/workflows/workflow_flux1_schnell.json

  # 设置 API Key（也可通过环境变量设置）
  RUNPOD_API_KEY=xxx python test_request.py

流程:
  1. POST /run → 获取 job id
  2. 轮询 GET /status/{id} 直到完成
  3. 输出结果
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
import requests

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENDPOINT_ID = "pux2uzv7ti6vox"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")

POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "5"))
MAX_POLL_TIME_S = int(os.environ.get("MAX_POLL_TIME_S", "600"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _headers() -> dict:
    if not API_KEY:
        print("⚠️  RUNPOD_API_KEY 未设置，使用占位符 'YOUR_API_KEY'", file=sys.stderr)
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY or 'YOUR_API_KEY'}",
    }


def submit_job(workflow: dict) -> str:
    """提交 workflow，返回 prompt_id (job id)。"""
    payload = {"input": {"workflow": workflow}}
    resp = requests.post(f"{BASE_URL}/run", headers=_headers(), json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(f"  submit response: {json.dumps(data, indent=2)}")
    prompt_id = data.get("id")
    if not prompt_id:
        raise RuntimeError(f"响应中没有 'id' 字段: {data}")
    return prompt_id


def poll_job(prompt_id: str) -> dict:
    """轮询任务状态直到完成或超时，返回最终 output。"""
    url = f"{BASE_URL}/status/{prompt_id}"
    deadline = time.time() + MAX_POLL_TIME_S

    while time.time() < deadline:
        resp = requests.get(url, headers=_headers())
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        print(f"  [{time.strftime('%H:%M:%S')}] status={status}", end="")

        if status == "COMPLETED":
            output = data.get("output")
            print(f"  output keys: {list(output.keys()) if isinstance(output, dict) else type(output).__name__}")
            return output
        elif status == "FAILED":
            error = data.get("error", "未知错误")
            raise RuntimeError(f"任务失败: {error}")
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            print()
            time.sleep(POLL_INTERVAL_S)
        else:
            print(f"  未知状态，等待重试...")
            time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(f"轮询超时（{MAX_POLL_TIME_S}s），prompt_id={prompt_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Runpod worker-comfyui 端到端测试")
    parser.add_argument(
        "--workflow",
        default="test_input.json",
        help="workflow JSON 文件路径（默认 test_input.json）",
    )
    args = parser.parse_args()

    # 加载 workflow
    with open(args.workflow) as f:
        raw = json.load(f)

    # test_input.json 外层包了一层 {"input": {"workflow": ..., "images": ...}}
    # 而 test_resources/workflows/*.json 也是同样的结构
    # 统一提取 workflow
    if isinstance(raw, dict) and "input" in raw:
        workflow = raw["input"].get("workflow", raw)
    else:
        workflow = raw

    print(f"Endpoint : {BASE_URL}")
    print(f"Workflow : {args.workflow}")
    print(f"API Key  : {'***' + API_KEY[-4:] if API_KEY else '(未设置)'}")
    print()

    # 1. 提交
    print("▶ 提交任务...")
    prompt_id = submit_job(workflow)
    print(f"  prompt_id = {prompt_id}")
    print()

    # 2. 轮询
    print("▶ 轮询结果...")
    output = poll_job(prompt_id)
    print()
    print("✅ 任务完成")
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()