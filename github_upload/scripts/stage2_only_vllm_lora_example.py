"""
stage2-only（本地 vLLM + LoRA）：只做第二阶段生成与要点评测。

输入：stage2 数据 JSON（列表，含 instruction/input/output）
生成：system=stage2 instruction（可 per-sample 对齐），user=stage2.input（等价于 stage1_pred）
测评：二阶段要点 PRF（strict + relaxed），relaxed 采用“方案一”：沿用 strict points，只放宽匹配。

示例：
  export BAILIAN_API_KEY=你的百炼Key   # 仅 --eval 且需二阶段大模型评测时
  export OMP_NUM_THREADS=1
  python3 stage2_only_vllm_lora_example.py --limit 5
  python3 stage2_only_vllm_lora_example.py --eval --limit 5
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    from vllm.lora.request import LoRARequest
except Exception:
    LoRARequest = None

import vllm_lora_example as api


def _make_lora_request(lora_name: str, lora_id: int, lora_path: str):
    if LoRARequest is None:
        raise RuntimeError("无法导入 vllm.lora.request.LoRARequest，请确认 vLLM 版本支持 LoRA。")
    import inspect

    sig = inspect.signature(LoRARequest)
    params = sig.parameters
    if "lora_local_path" in params:
        return LoRARequest(lora_name=lora_name, lora_int_id=lora_id, lora_local_path=lora_path)
    if "lora_path" in params:
        return LoRARequest(lora_name=lora_name, lora_int_id=lora_id, lora_path=lora_path)
    return LoRARequest(lora_name, lora_id, lora_path)


class Stage2OnlyVLLMRunner:
    def __init__(
        self,
        base_model: str,
        stage2_adapter: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
        dtype: str,
        trust_remote_code: bool,
        chat_template_file: str = "",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        api._maybe_override_chat_template(self.tokenizer, chat_template_file)
        base_kwargs = {
            "model": base_model,
            "enable_lora": True,
            "max_loras": 1,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "max_model_len": max_model_len,
        }
        try:
            self.llm = LLM(**base_kwargs, enforce_eager=enforce_eager)
        except Exception:
            # 兼容旧版本参数差异
            self.llm = LLM(**base_kwargs, enforce_eager=enforce_eager)
        self.lora_stage2 = _make_lora_request("stage2", 2, stage2_adapter)

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: SamplingParams,
        *,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        if not messages_list:
            return []
        prompts = [api._apply_chat_template(self.tokenizer, m, enable_thinking=enable_thinking) for m in messages_list]
        try:
            outs = self.llm.generate(prompts, sampling_params=sampling_params, lora_request=[self.lora_stage2] * len(prompts))
        except TypeError:
            outs = self.llm.generate(prompts, sampling_params, lora_requests=[self.lora_stage2] * len(prompts))
        texts: List[str] = []
        for o in outs or []:
            if o.outputs:
                texts.append(o.outputs[0].text)
            else:
                texts.append("")
        while len(texts) < len(messages_list):
            texts.append("")
        return texts


def build_parser() -> argparse.ArgumentParser:
    # 禁用参数缩写，避免诸如 --eval 与 --eval-stage1/--eval-stage2-relaxed-only 的歧义
    parser = argparse.ArgumentParser(description="stage2-only（本地 vLLM + LoRA）", allow_abbrev=False)
    cur_dir = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser.add_argument("--base-model", type=str, default="./models/base_model")
    parser.add_argument("--stage2-adapter", type=str, default="./adapters/stage2")
    parser.add_argument(
        "--stage2-data",
        type=str,
        default="./data/stage2_examples.json",
        help="第二阶段数据 JSON（列表，含 instruction/input/output）",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--vllm-batch-size", type=int, default=1)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--chat-template-file",
        type=str,
        default="./configs/qwen3_nothink.jinja",
    )

    parser.add_argument("--think", action="store_true", help="允许模型输出思考过程（默认禁止）")
    parser.set_defaults(eval=True)
    parser.add_argument("--eval", action="store_true", dest="eval")
    parser.add_argument("--no-eval", action="store_false", dest="eval")
    # stage2-only：默认只做 stage2 测评；stage1 为占位结构，开启会导致无意义的 BERTScore 计算
    parser.set_defaults(eval_stage1=False, eval_stage2=True)
    parser.add_argument("--eval-stage1", action="store_true", dest="eval_stage1")
    parser.add_argument("--no-eval-stage1", action="store_false", dest="eval_stage1")
    parser.add_argument("--no-eval-stage2", action="store_false", dest="eval_stage2")
    parser.add_argument("--eval-stage2-relaxed-only", action="store_true")
    parser.add_argument("--force-relaxed-eval", action="store_true")

    # 兼容 vllm_lora_example._evaluate_outputs_inplace：其内部会访问这些参数
    parser.add_argument("--bert-model-path", type=str, default="./models/bert")
    parser.add_argument("--bertscore-device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--bertscore-batch-size", type=int, default=32)

    parser.add_argument("--bailian-base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--bailian-model", type=str, default="qwen3.5-27b")
    parser.add_argument("--bailian-api-key", type=str, default="")
    parser.add_argument("--bailian-api-key-file", type=str, default="")

    run_dir = cur_dir / "runs" / "stage2_only_vllm" / f"run_{ts}"
    parser.add_argument("--output-jsonl", type=Path, default=run_dir / "stage2_only.jsonl")
    parser.add_argument("--output-xlsx", type=Path, default=run_dir / "stage2_only.xlsx")
    return parser


def main():
    args = build_parser().parse_args()
    stage2_rows = api._load_json_list(args.stage2_data)
    items = stage2_rows[int(args.start) : (len(stage2_rows) if int(args.limit) <= 0 else int(args.start) + int(args.limit))]
    if not items:
        raise ValueError("没有可处理的数据（start/limit 超出范围或数据为空）。")

    no_think = not bool(args.think)
    bailian_key = ""
    if args.eval and getattr(args, "eval_stage2", True):
        bailian_key = api._resolve_bailian_api_key(
            (args.bailian_api_key or "").strip() or None,
            (args.bailian_api_key_file or "").strip() or None,
        )
        if bailian_key:
            try:
                bailian_key = api._require_ascii_http_token("百炼 API Key", bailian_key)
            except ValueError as e:
                print(f"错误：{e}")
                bailian_key = ""

    runner = Stage2OnlyVLLMRunner(
        base_model=args.base_model,
        stage2_adapter=args.stage2_adapter,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        chat_template_file=str(args.chat_template_file or ""),
    )

    outputs: List[Dict[str, Any]] = []
    msgs_batch: List[List[Dict[str, str]]] = []
    meta: List[Tuple[int, Dict[str, Any], str, str, Any]] = []
    for i, row in enumerate(items):
        instruction = str(row.get("instruction") or "").strip()
        inp = str(row.get("input") or "").strip()
        gt = row.get("output")
        if not instruction or not inp:
            raise ValueError(f"stage2 第 {i} 条缺少 instruction/input（源索引 {int(args.start) + i}）")
        ins = instruction + (api.NO_THINK_SUFFIX if no_think else "")
        msgs = api._build_finetune_style_messages(ins, inp)
        msgs_batch.append(msgs)
        meta.append((int(args.start) + i, row, instruction, inp, gt))

    sp = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        seed=0,
    )
    bs = int(args.vllm_batch_size or 1)
    if bs <= 0:
        bs = 1

    preds: List[str] = []
    for start in range(0, len(msgs_batch), bs):
        chunk = msgs_batch[start : start + bs]
        preds.extend(runner.generate_batch(chunk, sp, enable_thinking=not no_think))

    for (sid, _row, instruction, inp, gt), pred in zip(meta, preds):
        p = api._strip_think_blocks(pred) if no_think else pred
        out_row: Dict[str, Any] = {
            "id": sid,
            "stage1": {},  # 占位，便于复用保存/汇总结构
            "stage2": {
                "instruction": instruction,
                "input": inp,
                "gt": gt,
                "pred": p,
            },
        }
        outputs.append(out_row)
        api.save_outputs_jsonl(args.output_jsonl, outputs)

    # 生成落盘
    api.save_outputs_jsonl(args.output_jsonl, outputs)
    try:
        api.save_outputs_xlsx(args.output_xlsx, outputs)
    except Exception as e:
        print(f"警告：写入 XLSX 失败（JSONL 已保存）：{e}")

    # 测评（只 stage2）
    if args.eval and getattr(args, "eval_stage2", True):
        # 尽量释放 vLLM 显存，再开始裁判/文本指标
        try:
            del runner
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        api._evaluate_outputs_inplace(outputs, args, bailian_key, jsonl_path=Path(args.output_jsonl))
        api.save_outputs_jsonl(args.output_jsonl, outputs)
        try:
            api.save_outputs_xlsx(args.output_xlsx, outputs)
        except Exception as e:
            print(f"警告：写入 XLSX 失败（JSONL 已保存）：{e}")
        api.save_eval_artifacts(args.output_jsonl, outputs)

    print(f"完成，共 {len(outputs)} 条；JSONL：{args.output_jsonl}")


if __name__ == "__main__":
    main()

