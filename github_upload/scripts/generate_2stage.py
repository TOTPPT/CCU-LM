"""2-Stage 数据生成脚本：

从标准化 Excel 读取 '超声所见' 与 '超声结论'，以所见作为 instruction 前缀拼接，
结论作为 output，生成 2-Stage 格式的 JSON 列表。

可选参数 --drop-index 用于丢弃某个已知异常样本（与旧版流程保持一致）。
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def generate_2stage(
    data: pd.DataFrame,
    start_text: str,
    drop_index: int | None = None,
):
    """将 DataFrame 转换为 2-Stage 数据列表。

    参数：
    - data: 以列为样本的 DataFrame，行索引包含 '超声所见' 与 '超声结论'
    - start_text: instruction 前缀文本（例如：请根据以下信息得出结论）
    - drop_index: 可选，删除指定的样本索引（与旧版流程一致）
    """
    # 读取两行为列表：所见（instruction 的主体）与结论（output）
    seen_list = data.loc['超声所见'].tolist()
    conclusion_list = data.loc['超声结论'].tolist()

    # 根据需要删除异常样本（索引有效性防护）
    if drop_index is not None and 0 <= drop_index < len(seen_list):
        del seen_list[drop_index]
        del conclusion_list[drop_index]

    # 拼装 2-Stage 格式的数据结构
    result = []
    start_text = start_text or '你是一名资深的脑颈动脉超声专家，请根据以下检查信息得出脑颈动脉超声检查结论:'
    for i in range(len(seen_list)):
        entry = {
            "instruction": start_text,
            "input": seen_list[i],
            "output": conclusion_list[i],
        }
        result.append(entry)
    return result


def main():
    """命令行入口：读取 Excel，生成并写出 2-Stage JSON。"""
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate 2-Stage JSON from standardized Excel")
    parser.add_argument("--excel", type=Path, default=ROOT / "data" / "processed" / "test-old.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "2stage_data.json")
    parser.add_argument("--start-text", type=str, default=None)
    parser.add_argument("--drop-index", type=int, default=None, help="Optionally drop one problematic sample index, e.g., 343")
    args = parser.parse_args()

    # 基本文件存在性校验
    if not args.excel.exists():
        raise FileNotFoundError(f"Excel not found: {args.excel}")

    # 读取 Excel（以第一列为索引，即行为标签）
    data = pd.read_excel(args.excel, index_col=0)

    # 生成 2-Stage 数据并写出到文件
    out_data = generate_2stage(data, args.start_text, args.drop_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as jf:
        json.dump(out_data, jf, indent=4, ensure_ascii=False)
    print(f"2-Stage JSON written: {args.output}")


if __name__ == "__main__":
    main()
