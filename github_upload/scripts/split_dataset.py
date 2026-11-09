"""通用数据集拆分脚本：

将任意 JSON 列表（样本序列）按比例随机拆分为 train/test 两份。
默认训练集占 90%，测试集占 10%，并支持设置随机种子保证复现实验。

不限定数据格式，只要求输入 JSON 顶层为 list。输出路径可自定义或使用默认命名。
"""

import argparse
import json
from pathlib import Path
import random


def split_json_dataset(
    input_json: Path,
    train_ratio: float,
    seed: int | None,
    out_train: Path,
    out_test: Path,
):
    """读取 JSON 列表并按比例拆分为训练/测试两份。

    参数：
    - input_json: 输入 JSON 文件路径（顶层应为 list）
    - train_ratio: 训练集所占比例（0~1），例如 0.9
    - seed: 随机种子，保证洗牌与切分的可复现性；为 None 时不固定随机性
    - out_train: 训练集输出文件路径
    - out_test: 测试集输出文件路径
    """
    # 输入文件校验与读取
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples")

    # 可选设置随机种子，并打乱样本顺序
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)

    # 计算训练集大小并切分
    n = len(data)
    n_train = int(n * train_ratio)
    train_data = data[:n_train]
    test_data = data[n_train:]

    # 写出两个结果文件
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, 'w', encoding='utf-8') as tf:
        json.dump(train_data, tf, indent=4, ensure_ascii=False)
    with open(out_test, 'w', encoding='utf-8') as tf:
        json.dump(test_data, tf, indent=4, ensure_ascii=False)
    print(f"Train written: {out_train} ({len(train_data)})")
    print(f"Test written: {out_test} ({len(test_data)})")


def main():
    """命令行入口：解析参数并执行拆分。"""
    parser = argparse.ArgumentParser(description="Split a JSON dataset into train/test")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON path")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--out-train", type=Path, required=False, help="Output train JSON path")
    parser.add_argument("--out-test", type=Path, required=False, help="Output test JSON path")
    args = parser.parse_args()

    # 若未指定输出路径，按输入文件名自动生成 -train/-test 后缀文件
    if args.out_train is None or args.out_test is None:
        stem = args.input.stem
        parent = args.input.parent
        args.out_train = parent / f"{stem}-train.json"
        args.out_test = parent / f"{stem}-test.json"

    split_json_dataset(args.input, args.train_ratio, args.seed, args.out_train, args.out_test)


if __name__ == "__main__":
    main()