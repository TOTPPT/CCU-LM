#!/usr/bin/env python3
"""
仅基于 stage2 数据集做 few-shot 生成 + 要点评测（不依赖 stage1）。

核心思路：
- stage2 数据集每条包含 instruction / input / output
- few-shot 也从同一（或另一个）stage2 JSON 列表按 indices 取样本拼成 {few_shot_block}
- 生成时复用 vllm_lora_example_api.py 的 prompt 配置与 OpenAI 兼容 API 调用：
  - 为兼容已有 stage2 prompt 模板（通常用 {stage1_pred}），这里将 stage1_pred 映射为该条 stage2 的 input
  - 即：ctx["stage1_pred"] = row["input"]
- 测评时复用 vllm_lora_example_api.py 的 _evaluate_stage2_points_once（百炼裁判/本地 fallback）

输出：
- JSONL：每条含 stage2{instruction,input,gt,pred,metrics...}，并写入 few_shot_meta
- XLSX：复用 save_outputs_xlsx（stage1 列为空）
- *_eval_summary.json / *_eval_samples.jsonl：复用 save_eval_artifacts
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# 复用已有两阶段脚本的能力（prompt、few-shot、key 解析、评测与落盘）
import vllm_lora_example_api as api


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="stage2-only：few-shot 生成 stage2 结论 + 要点评测（OpenAI 兼容 API）")
    cur_dir = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parents[2] / "outputs" / f"stage2_only_api_{ts}"

    p.add_argument("--prompt-json", type=str, default="", help="Prompt 配置 JSON（建议用 ./configs/prompts_fewshot.json）")
    p.add_argument("--stage2-data", type=str, required=True, help="stage2 数据 JSON（列表，含 instruction/input/output）")
    p.add_argument("--start", type=int, default=0, help="从第几条开始（原始索引）")
    p.add_argument("--limit", type=int, default=5, help="处理多少条；<=0 表示全部")

    p.add_argument("--infer-base-url", type=str, default=None, help="推理 base_url（默认跟随 vllm_lora_example_api.py 的默认/环境变量）")
    p.add_argument("--infer-model", type=str, default=None, help="推理模型（默认跟随 vllm_lora_example_api.py 的默认/环境变量）")
    p.add_argument("--infer-api-key", type=str, default="", help="推理 Key（同 vllm_lora_example_api.py 解析逻辑）")
    p.add_argument("--infer-api-key-file", type=str, default="", help="推理 Key 文件首行")
    p.add_argument("--infer-extra-body-json", type=str, default="", help="合并到请求 extra_body（JSON 字符串）")
    p.add_argument("--api-sleep-seconds", type=float, default=0.0, help="每条请求之间 sleep")

    p.add_argument("--max-tokens", type=int, default=4096, help="stage2 生成 max_tokens")
    p.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    p.add_argument("--top-p", type=float, default=1.0, help="top_p")
    p.add_argument("--think", action="store_true", help="允许思考输出（否则注入 no_think_suffix 并剥离 <think> 块）")

    p.add_argument("--output-jsonl", type=Path, default=run_dir / "stage2_only.jsonl", help="输出 JSONL")
    p.add_argument("--output-xlsx", type=Path, default=run_dir / "stage2_only.xlsx", help="输出 XLSX")

    p.set_defaults(eval=True)
    p.add_argument("--no-eval", action="store_false", dest="eval", help="关闭 stage2 测评")
    p.add_argument("--eval-stage2-relaxed-only", action="store_true", help="只算 relaxed 版本（忽略程度）")
    p.add_argument("--force-relaxed-eval", action="store_true", help="强制重算 relaxed 版本")

    p.add_argument("--bailian-base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="裁判 base_url")
    p.add_argument("--bailian-model", type=str, default="qwen3.5-27b", help="裁判模型")
    p.add_argument("--bailian-api-key", type=str, default="", help="裁判 Key（同 vllm_lora_example_api.py 解析逻辑）")
    p.add_argument("--bailian-api-key-file", type=str, default="", help="裁判 Key 文件首行")

    return p


def _slice_rows(rows: List[Dict[str, Any]], start: int, limit: int) -> List[Dict[str, Any]]:
    s = max(0, int(start or 0))
    if limit is None:
        return rows[s:]
    lim = int(limit)
    if lim <= 0:
        return rows[s:]
    return rows[s : s + lim]


def main() -> None:
    args = build_parser().parse_args()
    prompt_cfg = api.load_prompt_config(str(args.prompt_json or ""))

    if OpenAI is None:
        raise RuntimeError("需要安装 openai 库：pip install openai")

    infer_key = api._resolve_inference_api_key((args.infer_api_key or "").strip() or None, (args.infer_api_key_file or "").strip() or None)
    if not infer_key:
        raise RuntimeError("未配置推理 API Key：请设置 OPENROUTER_API_KEY / INFERENCE_API_KEY / OPENAI_API_KEY 或 --infer-api-key")
    infer_key = api._require_ascii_http_token("推理 API Key", infer_key)

    # 默认沿用 vllm_lora_example_api.py 的环境变量默认（openrouter + qwen/qwen3.6-plus 等）
    base_url = str(args.infer_base_url).strip() if args.infer_base_url else str(getattr(api.build_parser().parse_args([]), "infer_base_url"))
    model = str(args.infer_model).strip() if args.infer_model else str(getattr(api.build_parser().parse_args([]), "infer_model"))
    base_url = api._require_ascii_http_token("推理 base_url", base_url)
    model = api._require_ascii_http_token("推理 model", model)

    extra_body: Optional[Dict[str, Any]] = None
    raw_extra = (args.infer_extra_body_json or "").strip()
    if raw_extra:
        extra_body = json.loads(raw_extra)
        if not isinstance(extra_body, dict):
            raise ValueError("--infer-extra-body-json 必须是 JSON 对象")

    client = OpenAI(**api._openai_client_kwargs_for_infer(infer_key, base_url))

    stage2_rows_all = api._load_json_list(str(Path(args.stage2_data).expanduser()))
    stage2_rows = _slice_rows(stage2_rows_all, args.start, args.limit)
    if not stage2_rows:
        raise ValueError("没有可处理的 stage2 数据。")

    s2_cfg = prompt_cfg.get("stage2") or {}
    if not isinstance(s2_cfg, dict):
        raise ValueError("prompt 配置中 stage2 必须为对象")

    fs_root = prompt_cfg.get("few_shot") or {}
    few_shot_block = ""
    if isinstance(fs_root, dict) and fs_root:
        few_shot_block = api.build_few_shot_block_for_stage(fs_root.get("stage2"), stage_name="stage2")
        if few_shot_block:
            print(f"few_shot：stage2 块长度={len(few_shot_block)} 字符")

    few_shot_row_meta = api.build_few_shot_output_meta(
        fs_root,
        block_len_stage1=0,
        block_len_stage2=len(few_shot_block),
        prompt_json_path=str(args.prompt_json or ""),
    )

    no_think = not bool(args.think)
    nt_template = str(prompt_cfg.get("no_think_suffix_template") or "").strip() or api.NO_THINK_SUFFIX
    nt_suffix = nt_template if no_think else ""

    # 裁判 key（可与推理不同）
    bailian_key = ""
    explicit_b = (args.bailian_api_key or "").strip()
    if args.eval:
        bailian_key = api._resolve_bailian_api_key(explicit_b or None, (args.bailian_api_key_file or "").strip() or None)
        if bailian_key:
            bailian_key = api._require_ascii_http_token("百炼 API Key", bailian_key)
        # demo：同端点且未显式提供裁判 key -> 复用推理 key
        if not explicit_b and (str(args.bailian_base_url).strip().rstrip("/") == base_url.rstrip("/")):
            bailian_key = infer_key
            print("提示（demo）：裁判端点与推理端点一致且未指定 --bailian-api-key，已复用推理 Key。")

    outputs: List[Dict[str, Any]] = []
    sleep_s = float(args.api_sleep_seconds or 0.0)

    print("=" * 80)
    print("stage2-only 生成（通用 API）")
    print("=" * 80)
    print(f"infer base_url={base_url} model={model}")
    print(f"stage2_data={args.stage2_data} start={args.start} limit={args.limit}")
    print(f"prompt-json: {(args.prompt_json or '(内置默认)')}")
    print(f"no_think={no_think} eval={args.eval}")
    print("=" * 80)

    for i, row in enumerate(stage2_rows):
        instruction = str(row.get("instruction") or "")
        inp = str(row.get("input") or "")
        gt = row.get("output")
        if not instruction or not inp:
            raise ValueError(f"stage2 第 {i} 条缺少 instruction/input（源索引 {int(args.start) + i}）")

        # 兼容 prompt：把 stage1_pred 映射为 stage2.input
        ctx = {
            "instruction": instruction,
            "input": inp,
            "stage1_pred": inp,
            "no_think_suffix": nt_suffix,
            "few_shot_block": few_shot_block,
        }
        msgs = api.build_messages_from_stage_config(s2_cfg, ctx)
        pred = api.chat_completion_text(
            client,
            model,
            msgs,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_tokens),
            extra_body=extra_body,
        )
        if no_think:
            pred = api._strip_think_blocks(pred)

        out_row: Dict[str, Any] = {
            "id": int(args.start) + i,
            "stage2_source_index": int(args.start) + i,
            "stage1": {},  # 占位，便于复用 save_outputs_xlsx/save_eval_artifacts
            "stage2": {
                "instruction": instruction,
                "input": inp,
                "gt": gt,
                "pred": pred,
            },
        }
        if few_shot_row_meta:
            out_row["few_shot_meta"] = few_shot_row_meta
        outputs.append(out_row)
        api.save_outputs_jsonl(args.output_jsonl, outputs)

        print(f"\n[{i}] stage2 pred 预览：{pred[:500]}{'...' if len(pred) > 500 else ''}")

        if sleep_s > 0:
            import time

            time.sleep(sleep_s)

    print(f"\n生成阶段已结束（{len(outputs)} 条），正在强制落盘 JSONL / XLSX / 生成检查点…")
    api.persist_generation_outputs(args.output_jsonl, args.output_xlsx, outputs)

    if args.eval:
        relaxed_only = bool(getattr(args, "eval_stage2_relaxed_only", False))
        force_relaxed = bool(getattr(args, "force_relaxed_eval", False))
        for idx_row, r in enumerate(outputs, start=1):
            s2 = r.get("stage2") or {}
            p2 = str(s2.get("pred") or "")
            s2_gt = s2.get("gt")
            need_strict = (not relaxed_only) and (not api._has_stage2_prf_metrics(s2.get("metrics")))
            need_relaxed = force_relaxed or (not api._has_stage2_prf_metrics(s2.get("metrics_relaxed")))

            if not need_strict and not need_relaxed:
                api.save_outputs_jsonl(args.output_jsonl, outputs, fsync=True)
                continue

            if bailian_key and s2_gt is not None:
                pred_s2 = str(p2 or "")
                label_s2 = str(s2_gt or "")
                if need_strict:
                    m, ev, note, fb = api._evaluate_stage2_points_once(
                        pred_s2,
                        label_s2,
                        args,
                        bailian_key,
                        api.STAGE2_POINT_SYSTEM,
                        fallback_note="bailian_call_failed_used_manual",
                    )
                    s2["metrics"] = m
                    s2["point_eval"] = ev
                    if note:
                        s2["metrics_note"] = note
                    if fb is not None:
                        s2["metrics_fallback"] = fb
                if need_relaxed:
                    # 方案一：relaxed 沿用 strict 的 points，只放宽匹配规则
                    strict_ev = s2.get("point_eval")
                    pts = api._try_get_points_from_point_eval(strict_ev)
                    if pts is None:
                        # relaxed_only 场景也需要 points：若 strict points 不存在，先跑一次 strict 抽取
                        _m_tmp, strict_ev_tmp, _note_tmp, _fb_tmp = api._evaluate_stage2_points_once(
                            pred_s2,
                            label_s2,
                            args,
                            bailian_key,
                            api.STAGE2_POINT_SYSTEM,
                            fallback_note="bailian_call_failed_used_manual",
                        )
                        s2.setdefault("point_eval", strict_ev_tmp)
                        pts = api._try_get_points_from_point_eval(strict_ev_tmp)

                    if pts is None:
                        # 兜底：沿用旧逻辑（可能导致 points 粒度不一致，但不中断）
                        m2, ev2, note2, fb2 = api._evaluate_stage2_points_once(
                            pred_s2,
                            label_s2,
                            args,
                            bailian_key,
                            api.STAGE2_POINT_SYSTEM_RELAXED_DEGREE,
                            fallback_note="bailian_call_failed_used_manual",
                        )
                        s2["metrics_relaxed"] = m2
                        s2["point_eval_relaxed"] = ev2
                        if note2:
                            s2["metrics_relaxed_note"] = note2
                        if fb2 is not None:
                            s2["metrics_relaxed_fallback"] = fb2
                    else:
                        predict_points, label_points = pts
                        raw_corr: Optional[str] = None
                        corr_err: Optional[str] = None
                        try:
                            raw_corr = api.evaluate_correspondences_bailian_from_points(
                                predict_points,
                                label_points,
                                bailian_key,
                                args.bailian_base_url,
                                args.bailian_model,
                                system_content=api.STAGE2_POINT_SYSTEM_RELAXED_FROM_POINTS,
                            )
                        except ValueError as e:
                            corr_err = f"bailian_config_error: {e}"
                        except Exception as e:
                            corr_err = f"bailian_http_error: {type(e).__name__}: {e}"

                        if raw_corr:
                            try:
                                corr = api._normalize_correspondences_only(json.loads(raw_corr))
                                ev_relaxed = {
                                    "predict_points": predict_points,
                                    "label_points": label_points,
                                    "correspondences": corr,
                                }
                                s2["point_eval_relaxed"] = ev_relaxed
                                s2["metrics_relaxed"] = api.calculate_metrics([ev_relaxed])
                                if corr_err:
                                    s2["metrics_relaxed_note"] = corr_err
                            except Exception:
                                ev_fb = {
                                    "predict_points": predict_points,
                                    "label_points": label_points,
                                    "correspondences": [],
                                }
                                s2["point_eval_relaxed"] = ev_fb
                                s2["metrics_relaxed"] = api.calculate_metrics([ev_fb])
                                s2["metrics_relaxed_note"] = "invalid_json_used_manual_match_on_strict_points"
                        else:
                            ev_fb = {
                                "predict_points": predict_points,
                                "label_points": label_points,
                                "correspondences": [],
                            }
                            s2["point_eval_relaxed"] = ev_fb
                            s2["metrics_relaxed"] = api.calculate_metrics([ev_fb])
                            s2["metrics_relaxed_note"] = corr_err or "no_bailian_output_used_manual_match_on_strict_points"
            else:
                # 无裁判 key 或无标签 -> 本地 fallback
                if s2_gt is None:
                    if need_strict:
                        s2["metrics"] = {"skipped": True, "reason": "no_label"}
                    if need_relaxed:
                        s2["metrics_relaxed"] = {"skipped": True, "reason": "no_label"}
                else:
                    ev_fb = api.manual_parse_and_match(str(p2 or ""), str(s2_gt or ""))
                    if need_strict:
                        s2["metrics"] = api.calculate_metrics([ev_fb])
                        s2["metrics_note"] = "no_api_key_used_manual"
                        s2["point_eval"] = ev_fb
                    if need_relaxed:
                        s2["metrics_relaxed"] = api.calculate_metrics([ev_fb])
                        s2["metrics_relaxed_note"] = "no_api_key_used_manual"
                        s2["point_eval_relaxed"] = ev_fb

            # 增量落盘 JSONL
            api.save_outputs_jsonl(args.output_jsonl, outputs, fsync=True)
            if idx_row % 10 == 0 or idx_row == len(outputs):
                print(f"stage2 eval progress: {idx_row}/{len(outputs)}")

        # 最终落盘与 artifacts
        api.save_outputs_jsonl(args.output_jsonl, outputs, fsync=True)
        try:
            api.save_outputs_xlsx(args.output_xlsx, outputs)
        except Exception as e:
            print(f"警告：写入 XLSX 失败（JSONL 已保存）：{e}")
        api.save_eval_artifacts(args.output_jsonl, outputs)

    print("\n" + "=" * 80)
    print(f"完成，共 {len(outputs)} 条；JSONL：{args.output_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    main()

