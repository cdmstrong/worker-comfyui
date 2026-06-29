#!/usr/bin/env python3
"""
使用 validition_v1 测试集运行 MSR workflow 的端到端测试。

用法:
  # 测试 01 号样本
  python test_msr.py test_resources/validition_v1/01

  # 测试指定样本
  python test_msr.py test_resources/validition_v1/03

  # 自定义 workflow 和超时
  python test_msr.py test_resources/validition_v1/01 --workflow workflows/LTX-2.3_MSR_sample_workflow_V2_api.json --timeout 900
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENDPOINT_ID = "pux2uzv7ti6vox"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")

POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "10"))
MAX_POLL_TIME_S = int(os.environ.get("MAX_POLL_TIME_S", "600"))

# ---------------------------------------------------------------------------
# 已知的 widget 名称映射（class_type → [(widget_index, input_name), ...]）
# 用于补全转换时丢失的 widget 值
# ---------------------------------------------------------------------------
WIDGET_MAP = {
    "LoadImage": [(0, "image"), (1, "upload")],
    "CLIPTextEncode": [(0, "text")],
    "LowVRAMCheckpointLoader": [(0, "ckpt_name")],
    "ManualSigmas": [(0, "sigmas")],
    "LTXAVTextEncoderLoader": [(0, "model_name")],
    "LTXVAudioVAELoader": [(0, "model_name")],
    "PromptRelayEncode": [(0, "prompt"), (1, "motion"), (2, "negative"), (3, "cfg")],
    "INTConstant": [(0, "value")],
    "KSamplerSelect": [(0, "sampler_name")],
    "RandomNoise": [(0, "noise_seed"), (1, "mode")],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _headers() -> dict:
    if not API_KEY:
        print("⚠️  RUNPOD_API_KEY 未设置", file=sys.stderr)
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY or 'YOUR_API_KEY'}",
    }


def base64_encode_file(path: str) -> str:
    """读取文件并返回 base64 编码字符串。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def patch_widget_values(api: dict, saved: dict):
    """
    从保存格式的 workflow 中提取 widgets_values，
    按 WIDGET_MAP 补全 API workflow 中缺失的 widget 输入。
    """
    # 构建 nodes 查找表
    nodes_map = {}
    for node in saved.get("nodes", []):
        nodes_map[str(node["id"])] = node

    for node_id, node_api in api.items():
        saved_node = nodes_map.get(node_id)
        if not saved_node:
            continue

        wv = saved_node.get("widgets_values", [])
        class_type = node_api["class_type"]
        widget_defs = WIDGET_MAP.get(class_type, [])

        for idx, input_name in widget_defs:
            if idx < len(wv) and input_name not in node_api["inputs"]:
                node_api["inputs"][input_name] = wv[idx]


def map_images(api: dict, test_dir: Path) -> list[dict]:
    """
    将本地图片映射到 LoadImage 节点。

    策略：
    - 节点 40 → 1.jpg/png
    - 节点 29 → 2.jpg/png
    - 节点 30 → bg.png
    - 节点 95 → 4.png（已禁用，跳过）
    - 节点 33 → 3.png（已禁用，跳过）
    """
    # 查找 test_dir 中所有图片文件
    image_files = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for f in test_dir.glob(ext):
            stem = f.stem  # 不带扩展名
            image_files[stem] = str(f)

    # 映射规则：节点 ID → 文件名前缀
    ref_map = {
        "40": "1",
        "29": "2",
        "30": "bg",
    }

    images_payload = []
    for node_id, prefix in ref_map.items():
        node = api.get(node_id)
        if not node or node["class_type"] != "LoadImage":
            continue

        # 找到匹配的文件
        matched = None
        for stem, fpath in image_files.items():
            if stem == prefix:
                matched = fpath
                break

        if not matched:
            print(f"  ⚠️  节点 {node_id}: 未找到匹配文件 (期望 {prefix}.*)，跳过")
            continue

        # 更新 workflow 中的文件名
        filename = Path(matched).name
        node["inputs"]["image"] = filename

        # 添加到上传列表
        images_payload.append({
            "name": filename,
            "image": f"data:image/{Path(matched).suffix[1:]};base64,{base64_encode_file(matched)}",
        })
        print(f"  📷 节点 {node_id} ← {filename}")

    return images_payload


def inject_prompt(api: dict, prompt_text: str):
    """
    将 prompt.txt 内容注入到相关节点。

    PromptRelayEncode (节点 99) 的 widgets_values 格式：
      [0] = 参考图描述（prompt）
      [1] = 动作/运镜描述（motion）
      [2] = 负面提示（negative）
      [3] = cfg

    CLIPTextEncode (节点 6) 的 text 设为空（因为 PromptRelayEncode 接管了编码）。
    """
    # 从 prompt.txt 分割：空行之前是参考图描述，空行之后是动作描述
    parts = prompt_text.strip().split("\n\n", 1)
    ref_desc = parts[0].strip() if parts else ""
    motion_desc = parts[1].strip() if len(parts) > 1 else ""

    # PromptRelayEncode (节点 99)
    if "99" in api:
        api["99"]["inputs"]["prompt"] = ref_desc
        api["99"]["inputs"]["motion"] = motion_desc
        api["99"]["inputs"]["negative"] = ""
        api["99"]["inputs"]["cfg"] = 0.0022
        print(f"  📝 节点 99 (PromptRelayEncode) 已注入 prompt")

    # CLIPTextEncode (节点 6) — 作为 negative 编码
    if "6" in api:
        api["6"]["inputs"]["text"] = ""
        print(f"  📝 节点 6 (CLIPTextEncode) 已注入空文本")


# ---------------------------------------------------------------------------
# Submit & Poll
# ---------------------------------------------------------------------------
def submit_job(workflow: dict, images: list) -> str:
    payload = {"input": {"workflow": workflow}}
    if images:
        payload["input"]["images"] = images

    resp = requests.post(f"{BASE_URL}/run", headers=_headers(), json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(f"  submit response: id={data.get('id')}")
    prompt_id = data.get("id")
    if not prompt_id:
        raise RuntimeError(f"响应中没有 'id' 字段: {data}")
    return prompt_id


def poll_job(prompt_id: str) -> dict:
    url = f"{BASE_URL}/status/{prompt_id}"
    deadline = time.time() + MAX_POLL_TIME_S

    while time.time() < deadline:
        resp = requests.get(url, headers=_headers())
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] status={status}")

        if status == "COMPLETED":
            return data.get("output", data)
        elif status == "FAILED":
            error = data.get("error", "未知错误")
            raise RuntimeError(f"任务失败: {error}")
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            time.sleep(POLL_INTERVAL_S)
        else:
            time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(f"轮询超时（{MAX_POLL_TIME_S}s），prompt_id={prompt_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MSR workflow 端到端测试")
    parser.add_argument("test_dir", help="测试集目录路径（如 test_resources/validition_v1/01）")
    parser.add_argument(
        "--workflow",
        default="workflows/LTX-2.3_MSR_sample_workflow_V2_api.json",
        help="API 格式 workflow JSON（默认 workflows/LTX-2.3_MSR_sample_workflow_V2_api.json）",
    )
    parser.add_argument(
        "--saved",
        default="workflows/LTX-2.3_MSR_sample_workflow_V2.json",
        help="原始保存格式 workflow JSON（用于提取 widget 值）",
    )
    parser.add_argument("--timeout", type=int, default=None, help="轮询超时秒数")
    args = parser.parse_args()

    if args.timeout is not None:
        global MAX_POLL_TIME_S
        MAX_POLL_TIME_S = args.timeout

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        print(f"❌ 目录不存在: {test_dir}")
        sys.exit(1)

    # 加载 API workflow
    with open(args.workflow) as f:
        api = json.load(f)

    # 加载原始保存格式 workflow（用于补全 widget 值）
    with open(args.saved) as f:
        saved = json.load(f)

    # 加载 prompt
    prompt_file = test_dir / "prompt.txt"
    prompt_text = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""

    print(f"Endpoint : {BASE_URL}")
    print(f"Test dir : {test_dir}")
    print(f"Workflow : {args.workflow}")
    print(f"API Key  : {'***' + API_KEY[-4:] if API_KEY else '(未设置)'}")
    print()

    # 1. 补全 widget 值
    print("▶ 补全 widget 值...")
    patch_widget_values(api, saved)

    # 2. 注入 prompt 文本
    print("▶ 注入 prompt...")
    inject_prompt(api, prompt_text)

    # 3. 映射图片
    print("▶ 映射图片...")
    images = map_images(api, test_dir)

    print()
    print("▶ 提交任务...")
    prompt_id = submit_job(api, images)
    print(f"  prompt_id = {prompt_id}")
    print()

    print("▶ 轮询结果...")
    output = poll_job(prompt_id)
    print()
    print("✅ 任务完成")
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()