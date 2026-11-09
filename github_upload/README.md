# 脑颈动脉超声数据处理与模型测评

本仓库把数据处理和后续测评放在同一套目录下。每个脚本独立运行，通过 JSON 文件衔接，不需要统一编排脚本。

## 目录结构

```text
.
├── configs/                   # 表头映射、判定标准、测评提示词和 chat template
├── data/
│   ├── raw/                    # 本地原始数据（Git 忽略）
│   ├── processed/              # 处理输出（Git 忽略）
│   ├── processing_examples/    # 两例数据处理格式样例
│   ├── stage1_examples.json    # 两阶段测评样例
│   ├── stage1_fewshot.json     # stage1 few-shot 样例
│   ├── stage2_examples.json    # stage2 测评样例
│   └── stage2_fewshot.json     # stage2 few-shot 样例
├── scripts/                    # 数据处理、推理与测评脚本
├── outputs/                    # 运行结果（Git 忽略）
└── requirements.txt
```

## 安装

从仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 数据处理

将授权且脱敏的原始 Excel 放入 `data/raw/`，按实际表头检查 `configs/replace_dict.json`，再按需单独运行：

```bash
python scripts/preprocess_headers.py --input-excel data/raw/input.xlsx --replace-json configs/replace_dict.json --output-excel data/processed/standardized.xlsx
python scripts/generate_alpaca.py --excel data/processed/standardized.xlsx --standard configs/standard.json --output data/processed/stage1.json
python scripts/generate_2stage.py --excel data/processed/standardized.xlsx --output data/processed/stage2.json
python scripts/split_dataset.py --input data/processed/stage2.json --train-ratio 0.9 --seed 42
```

标准化表格结构为“字段在行、病例在列”。字段要求和可选参数可通过 `python scripts/<脚本名>.py --help` 查看。两例处理格式样例位于 `data/processing_examples/`。

## 后续测评

API 两阶段推理与测评示例（在仓库根目录运行）：

```powershell
$env:INFERENCE_API_KEY = "YOUR_INFERENCE_KEY"
$env:BAILIAN_API_KEY = "YOUR_EVALUATION_KEY"
python scripts/run_two_stage_api.py --stage1-data ./data/stage1_examples.json --stage2-data ./data/stage2_examples.json --prompt-json ./configs/prompts_fewshot.json --limit 0 --no-bertscore --output-jsonl ./outputs/two_stage.jsonl --output-xlsx ./outputs/two_stage.xlsx
```

单独运行 stage2：

```powershell
python scripts/run_stage2_only_api.py --stage2-data ./data/stage2_examples.json --prompt-json ./configs/prompts_fewshot.json --limit 0 --output-jsonl ./outputs/stage2_only.jsonl --output-xlsx ./outputs/stage2_only.xlsx
```

数据处理生成的 JSON 可直接传给测评脚本的 `--stage1-data` / `--stage2-data`。底层 API 脚本为 `scripts/vllm_lora_example_api.py` 和 `scripts/stage2_only_fewshot_api.py`；本地 vLLM + LoRA 脚本为 `scripts/vllm_lora_example.py` 和 `scripts/stage2_only_vllm_lora_example.py`。密钥、模型路径和完整参数请查看脚本 `--help` 与配置说明。

## 上传说明

仓库样例仅保留病例文本，不含姓名、患者 ID 或机构字段。原始数据、处理结果、模型权重、API 密钥和运行产物均由 `.gitignore` 排除。分享真实数据前，请确认已获得授权并完成脱敏。
