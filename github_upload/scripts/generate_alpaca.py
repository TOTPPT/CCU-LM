"""Alpaca 数据生成脚本：

读取标准化 Excel 与标准配置，调用 core/writeV2_allvas.py 的规则生成 instruction，
将 '超声所见' 作为 output，拼装成 Alpaca 格式的 JSON 列表。

可选参数 --drop-index 用于丢弃某个已知异常样本（与旧版流程保持一致）。
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# 将 core/ 加入模块搜索路径，复用旧版核心逻辑（不修改源码）
ROOT = Path(__file__).resolve().parents[1]
# 核心规则模块与本脚本一起放在 scripts/ 中，便于统一目录直接运行。
CORE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE_DIR))

import writeV2_allvas as W2  # noqa: E402  # 延迟导入以确保 core/ 路径已注入


def generate_alpaca(
    data: pd.DataFrame,
    standard: dict,
    start_text: str,
    drop_index: int | None = None,
    new_instruction: str | None = None,
):
    """从标准化 DataFrame 生成 Alpaca 格式数据。

    参数：
    - data: 以列为样本的 DataFrame，行索引包含 '超声所见' 等字段
    - standard: 从 standard.json 读取的标准配置字典
    - start_text: instruction 前缀文本（例如：请根据以下信息写一份报告）
    - drop_index: 可选，删除指定的样本索引（与旧版流程中删除第 343 列保持一致）
    - new_instruction: 可选，参照 modify_json_dataset.py，将最终 instruction 改写为该值；
      若为 None，则保留 start_text。
    """
    v1_list = []
    # 逐列处理样本，调用旧版规则生成每个样本的 instruction 文本片段
    for i in range(data.shape[1]):
        v1_list.append(W2.writeV2_all(standard, data.iloc[:, i]))

    # 将 "超声所见" 行作为 output
    origindata = data.loc['超声所见'].tolist()

    # 根据需要删除一个异常样本（索引有效性防护）
    if drop_index is not None and 0 <= drop_index < len(v1_list):
        del v1_list[drop_index]
        del origindata[drop_index]

    # 参照 modify_json_dataset.py 的处理：
    # - 原先的 instruction = start_text + 规则文本；
    # - 现在将规则文本放入 input；instruction 保留或改写为 new_instruction。
    result = []
    new_instr_value = new_instruction if new_instruction is not None else start_text
    for i in range(len(v1_list)):
        entry = {
            "instruction": new_instr_value,
            "input": v1_list[i].strip(),
            "output": origindata[i],
        }
        result.append(entry)
    return result

prompt = '''
你是一名资深的脑颈动脉超声专家，根据病例信息写一份脑颈动脉超声检查报告，以下规则为部分血管疾病判断规则：
【规则】：
一、	颈总动脉、颈内动脉、颈外动脉重度狭窄（70%–99%）判断：
1.	峰值流速、舒张末期流速、流速比值参数均提示严重升高，必须满足三者都严重升高。
2.	残余内径小于1.5 mm
必须同时满足以上两个条件，则“***满足重度狭窄条件”（***指代血管名）
二、	椎动脉重度狭窄（70%–99%）判断：
与一种1、2两点相同，同时还需满足狭窄处搏动指数较未狭窄处下降0.1以上，都满足则“椎动脉满足重度狭窄条件”
三、颈外动脉、椎动脉代偿现象判断：
1.	在颈内动脉≥70%狭窄或闭塞时，颈外动脉或者椎动脉出现流速升高但无管径变细或残余内径缩小，应诊断为代偿性血流加速，输出“***血流速度代偿性升高”。
四、颈总动脉盗血现象判断
1.	颈总动脉重度狭窄
2.	颈内、颈外动脉血流速度明显降低、搏动性减弱，频谱出现切迹
满足上述两个条件则输出“怀疑颈总动脉盗血”

【病例】：
'''

def main():
    """命令行入口：读取 Excel 与标准配置，生成并写出 Alpaca JSON。"""
    parser = argparse.ArgumentParser(description="Generate Alpaca JSON from standardized Excel using V2 writer")
    parser.add_argument("--excel", type=Path, default=ROOT / "data" / "processed" / "test-old.xlsx")
    parser.add_argument("--standard", type=Path, default=ROOT / "configs" / "standard.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "alpaca_data-test-V4.json")
    parser.add_argument("--start-text", type=str, default = prompt)
    parser.add_argument("--drop-index", type=int, default=None, help="Optionally drop one problematic sample index, e.g., 343")
    parser.add_argument("--new-instruction", type=str, default=None, help="Rewrite instruction per modify_json_dataset logic")
    args = parser.parse_args()

    # 基本文件存在性校验
    if not args.excel.exists():
        raise FileNotFoundError(f"Excel not found: {args.excel}")
    if not args.standard.exists():
        raise FileNotFoundError(f"Standard JSON not found: {args.standard}")

    # 读取 Excel（以第一列为索引，即行为标签）与标准配置 JSON
    data = pd.read_excel(args.excel, index_col=0)
    with open(args.standard, 'r', encoding='utf-8') as f:
        standard = json.load(f)

    # 生成 Alpaca 数据并写出到文件
    alpaca_data = generate_alpaca(data, standard, args.start_text, args.drop_index, args.new_instruction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as jf:
        json.dump(alpaca_data, jf, indent=4, ensure_ascii=False)
    print(f"Alpaca JSON written: {args.output}")


if __name__ == "__main__":
    main()
