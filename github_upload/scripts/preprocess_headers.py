"""预处理脚本：将原始 Excel 的列名依据映射表替换为标准列名，并输出新的 Excel。

- 输入：原始 Excel（data/raw 下）、列名映射 JSON（config 下）
- 输出：标准化 Excel（data/processed 下），并转置为“字段在行、样本在列”的结构，
  以便下游 `generate_alpaca.py` 使用 `index_col=0` 读取。

不修改任何旧版代码，仅以脚本形式组织流程。
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd


def preprocess_headers(input_excel: Path, replace_json: Path, output_excel: Path) -> None:
    """读取 Excel，按映射表替换列名，并保存到指定输出路径。

    参数说明：
    - input_excel: 原始 Excel 文件路径
    - replace_json: 列名替换映射的 JSON 文件路径（形如 {旧列名: 新列名}），支持子串替换
    - output_excel: 输出的标准化 Excel 路径

    处理流程：列名子串替换 -> 数据转置（字段在行、样本在列） -> 写出 Excel。
    """
    # 基本校验：确保输入文件存在
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_excel}")
    if not replace_json.exists():
        raise FileNotFoundError(f"Replace dict JSON not found: {replace_json}")

    # 读取映射 JSON，并校验类型
    with open(replace_json, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise ValueError("replace_dict.json must be a flat dict of {old: new}")

    # 读取原始 Excel 为 DataFrame
    df = pd.read_excel(input_excel)

    # 旧版逻辑：按子串批量替换列名（而非精确键替换）
    def replace_multiple_chars(text: str, replace_dict: dict) -> str:
        pattern = re.compile("|".join(re.escape(key) for key in replace_dict.keys()))
        return pattern.sub(lambda match: replace_dict[match.group(0)], text)

    changed_headers = [replace_multiple_chars(str(c), mapping) for c in df.columns]

    # 转置为“字段在行、样本在列”，并使用替换后的表头作为行索引
    new_df = pd.DataFrame(df.values.T, index=changed_headers)

    # 确保输出目录存在并写出结果（保留索引，供下游 index_col=0 使用）
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_excel(output_excel)
    print(f"Preprocessed Excel saved: {output_excel}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，设置默认输入/输出路径。"""
    parser = argparse.ArgumentParser(
        description="Replace Excel headers using mapping and output standardized Excel"
    )
    # ROOT 为项目根（数据生成-新版），便于设置默认路径
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input-excel",
        type=Path,
        default=root / "data" / "raw" / "神内-周瑛华-20250212.xlsx",
        help="Raw Excel path"
    )
    parser.add_argument(
        "--replace-json",
        type=Path,
        default=root / "configs" / "replace_dict.json",
        help="Header replace mapping JSON"
    )
    parser.add_argument(
        "--output-excel",
        type=Path,
        default=root / "data" / "processed" / "data-old.xlsx",
        help="Standardized Excel output path"
    )
    return parser


def main():
    """主函数：解析参数并执行预处理。"""
    parser = build_parser()
    args = parser.parse_args()
    preprocess_headers(args.input_excel, args.replace_json, args.output_excel)


if __name__ == "__main__":
    main()
