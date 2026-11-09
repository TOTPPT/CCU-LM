# 脑颈动脉超声模型测评示例

本仓库仅包含模型推理与测评示例，不包含原始数据处理流程。API 和本地 vLLM + LoRA 脚本相互独立，可按需要运行。

## 目录结构

```text
.
├── configs/                   # 测评提示词与 chat template
├── data/
│   ├── stage1_examples.json   # 两阶段测评样例
│   ├── stage1_fewshot.json    # stage1 few-shot 样例
│   ├── stage2_examples.json   # stage2 测评样例
│   └── stage2_fewshot.json    # stage2 few-shot 样例
├── scripts/evaluation/        # API / 本地 vLLM 推理与测评
├── outputs/                   # 运行结果（Git 忽略）
└── requirements.txt
```

## 安装

在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## API 两阶段推理与测评

在仓库根目录运行：

```powershell
$env:INFERENCE_API_KEY = "YOUR_INFERENCE_KEY"
$env:BAILIAN_API_KEY = "YOUR_EVALUATION_KEY"
python scripts/evaluation/run_two_stage_api.py --stage1-data ./data/stage1_examples.json --stage2-data ./data/stage2_examples.json --prompt-json ./configs/prompts_fewshot.json --limit 0 --no-bertscore --output-jsonl ./outputs/two_stage.jsonl --output-xlsx ./outputs/two_stage.xlsx
```

## 单独运行 stage2

```powershell
python scripts/evaluation/run_stage2_only_api.py --stage2-data ./data/stage2_examples.json --prompt-json ./configs/prompts_fewshot.json --limit 0 --output-jsonl ./outputs/stage2_only.jsonl --output-xlsx ./outputs/stage2_only.xlsx
```

推理和裁判模型的 API 地址、模型名、评测开关均可通过 `--help` 查看。密钥通过环境变量提供，不要写入脚本或提交到 GitHub。本地 vLLM + LoRA 入口也位于 `scripts/evaluation/`，模型与 adapter 路径请按实际环境配置。

## 样例数据

样例 JSON 仅用于展示 `instruction/input/output` 格式。对外分享前请确认样例数据和提示词符合项目授权与数据管理要求。
