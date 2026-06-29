#!/usr/bin/env python3
"""
下载 HuggingFace 验证集 validition_v1。

用法:
  python download_validition.py                        # 下载全部
  python download_validition.py --dir test_resources/validition_v1  # 指定目录
  python download_validition.py --subset 01 02         # 只下载 01, 02
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE_HF = "https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference/resolve/main"
API_TREE = "https://huggingface.co/api/models/LiconStudio/LTX-2.3-Multiple-Subject-Reference/tree/main"


def list_remote_files(subdir: str) -> list[dict]:
    """获取远程目录的文件列表。"""
    url = f"{API_TREE}/{subdir}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    return [f for f in data if f["type"] == "file"]


def download_file(url: str, dest: Path):
    """下载单个文件，显示进度。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ⏭  {dest.name} 已存在，跳过")
        return

    print(f"  ↓ {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="下载 validition_v1 验证集")
    parser.add_argument(
        "--dir",
        default="test_resources/validition_v1",
        help="本地保存目录（默认 test_resources/validition_v1）",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        help="只下载指定子目录，如 01 02",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    subsets = args.subset or [f"{i:02d}" for i in range(1, 9)]

    for sub in subsets:
        print(f"\n📁 {sub}/")
        files = list_remote_files(f"validition_v1/{sub}")
        for f in files:
            rel_path = f["path"]  # e.g. validition_v1/01/1.jpg
            url = f"{BASE_HF}/{rel_path}"
            dest = base_dir / sub / Path(rel_path).name
            download_file(url, dest)

    print("\n✅ 下载完成")


if __name__ == "__main__":
    main()