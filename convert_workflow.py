#!/usr/bin/env python3
"""
将 ComfyUI 保存格式的 workflow（nodes + links 数组）
转换为 API prompt 格式（{node_id: {inputs, class_type}} 扁平 dict）。

用法:
  # 转换并输出到终端
  python convert_workflow.py workflows/LTX-2.3_MSR_sample_workflow_V2.json

  # 转换并保存到文件
  python convert_workflow.py workflows/LTX-2.3_MSR_sample_workflow_V2.json -o api_workflow.json

  # 同时列出需要提供的图片
  python convert_workflow.py workflows/LTX-2.3_MSR_sample_workflow_V2.json --list-images
"""

import argparse
import json
import sys


def convert_saved_to_api(saved: dict) -> dict:
    """
    将 ComfyUI 保存格式 workflow 转换为 API prompt 格式。

    ComfyUI 保存格式:
        { "nodes": [{ "id": 6, "type": "CLIPTextEncode", "widgets_values": [...],
                       "inputs": [{"name": "text", "link": null, "widget": {"name": "text"}}, ...],
                       "outputs": [{"name": "CONDITIONING", "links": [123]}, ...] }],
          "links": [[link_id, from_id, from_slot, to_id, to_slot, type_str], ...] }

    API prompt 格式:
        { "6": { "class_type": "CLIPTextEncode",
                 "inputs": { "text": "hello", "clip": ["30", 1] } } }
    """
    nodes = saved.get("nodes", [])
    links = saved.get("links", [])

    # --- 构建 link 查找表: link_id → {from_id, from_slot, to_id, to_slot} ---
    link_map = {}
    for link in links:
        link_id, from_id, from_slot, to_id, to_slot = link[:5]
        link_map[link_id] = {
            "from_id": str(from_id),
            "from_slot": from_slot,
            "to_id": str(to_id),
            "to_slot": to_slot,
        }

    # --- 构建节点查找表 ---
    api = {}
    for node in nodes:
        node_id = str(node["id"])
        class_type = node["type"]
        inputs = {}

        # 收集 widget 输入（有 widget 定义且 link 为 null 的输入）
        # widgets_values 按顺序对应有 widget 的 input
        widget_idx = 0
        for inp in node.get("inputs", []):
            name = inp.get("name", "")
            link_id = inp.get("link")
            has_widget = "widget" in inp

            if link_id is not None:
                # 连接到其他节点
                link_info = link_map.get(link_id)
                if link_info:
                    inputs[name] = [link_info["from_id"], link_info["from_slot"]]
            elif has_widget:
                # widget 输入，从 widgets_values 取值
                wv = node.get("widgets_values", [])
                if widget_idx < len(wv):
                    inputs[name] = wv[widget_idx]
                widget_idx += 1
            # else: 未连接且无 widget，跳过

        api[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }

        # 保留 _meta 信息（title 等）
        meta = node.get("properties", {}).get("Node name for S&R")
        if meta:
            api[node_id]["_meta"] = {"title": meta}

    return api


def list_load_images(api: dict) -> list[dict]:
    """从转换后的 API 中找出所有 LoadImage 节点，返回 [{node_id, filename}]。"""
    images = []
    for node_id, node in api.items():
        if node["class_type"] == "LoadImage":
            filename = node["inputs"].get("image", "")
            images.append({"node_id": node_id, "filename": filename})
    return images


def main():
    parser = argparse.ArgumentParser(description="ComfyUI workflow 格式转换")
    parser.add_argument("input", help="输入的 workflow JSON 文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（默认输出到终端）")
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="列出所有 LoadImage 节点需要的图片",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        saved = json.load(f)

    api = convert_saved_to_api(saved)

    if args.list_images:
        images = list_load_images(api)
        if images:
            print(f"找到 {len(images)} 个 LoadImage 节点：")
            for img in images:
                print(f"  节点 {img['node_id']}: {img['filename']}")
        else:
            print("没有 LoadImage 节点")
        return

    output = json.dumps(api, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"已写入 {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()