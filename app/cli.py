"""CLI：阶段验证入口。

用法（在项目根目录下）：
    python -m app.cli                 # 目录文档同步入库（增量）
    python -m app.cli --no-prune      # 入库时保留目录外的旧文档
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.config import settings
from app.rag.ingest import ingest_directory


def cmd_ingest(args: argparse.Namespace) -> None:
    report = ingest_directory(settings.docs_dir, prune=not args.no_prune)
    print(f"目录：{settings.docs_dir}")
    if report.added_files:
        print(f"新增 {len(report.added_files)} 个文件：{', '.join(report.added_files)}")
    if report.updated_files:
        print(f"更新 {len(report.updated_files)} 个文件：{', '.join(report.updated_files)}")
    if report.skipped_files:
        print(f"跳过 {len(report.skipped_files)} 个未改动文件（命中缓存）")
    if report.pruned_files:
        print(f"清理 {len(report.pruned_files)} 个已删除文件的孤儿块：{', '.join(report.pruned_files)}")
    print(f"本次入库 {report.added_chunks} 块，知识库现有 {report.total_chunks} 块")


def main() -> None:
    parser = argparse.ArgumentParser(description="企业知识库文档摄入")
    parser.add_argument("--no-prune", action="store_true", help="不清理已删除文件的孤儿块")
    args = parser.parse_args()
    # 直接调用 ingest，args 结构与原来兼容
    cmd_ingest(args)


if __name__ == "__main__":
    main()
