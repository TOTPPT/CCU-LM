"""
vLLM 多 LoRA 适配器两阶段测试脚本（本地 vLLM Python API）

基座模型：./models/base_model
第一阶段 LoRA 默认：./adapters/stage1
第二阶段 LoRA：./adapters/stage2

第一阶段 system prompt：stage1 数据 JSON 的 instruction；user 为 input。
第二阶段 system prompt：stage2 数据 JSON 的 instruction（可与样本对齐）；user 为第一阶段模型输出。

在 val 环境运行示例：
    export BAILIAN_API_KEY=你的百炼Key   # 仅 --eval 且需二阶段大模型评测时
    export OMP_NUM_THREADS=1
    python3 vllm_lora_example.py --limit 5
    python3 vllm_lora_example.py --eval --limit 5
    # 需要模型输出思考过程时用：--think
"""

import argparse
import gc
import inspect
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
except Exception:
    np = None
    precision_score = recall_score = f1_score = accuracy_score = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from vllm.lora.request import LoRARequest
except Exception:
    LoRARequest = None


NO_THINK_SUFFIX = (
    "\n\n【严格要求】请直接输出最终答案。"
    "禁止输出任何思考过程/分析步骤/草稿/推理链。"
    "禁止输出 <think>...</think> 或任何类似标签包裹的中间内容；也不要输出未闭合的 <think> 开标签。"
)

STAGE2_POINT_SYSTEM = """
你是一位资深的医学评估专家，擅长分析诊断信息的要点对应关系。
你的任务是评估predict和label中的诊断信息要点之间的对应关系。

要点识别规则（严格执行）：
-该文本为脑颈动脉超声检查结果，分为颅内和颅外两部分，请对其都进行要点提取
-要点多为某根脑劲动脉加患病症状，例如’双侧颈动脉球部狭窄（右侧：50%-69%；左侧：小于50%）‘

要点匹配原则（严谨执行）：
1. 医学术语匹配：严格按照医学专业术语的准确性进行匹配，尤其注意左右侧、血管名称、患病程度的匹配
2. 同义词识别：识别医学同义词和近义词（如：心肌梗死=心梗，高血压=血压升高）
3. 严格判断（）内表示患病程度的百分数程度对应，程度必须严格对应，例如’右侧 carotid动脉狭窄（50%）‘与’右侧 carotid动脉狭窄（60%）‘不匹配
4. 严格判断（）内表示患病程度的’轻、中、重‘程度对应，程度必须严格对应，例如’右侧大脑中动脉狭窄（中度）‘与’右侧大脑中动脉狭窄（重度）‘不匹配
5. 严格判断起始左右侧血管是否对应，例如’左侧 carotid动脉狭窄‘与’右侧 carotid动脉狭窄‘不匹配
6. 左右侧、血管名称、（）内程度有任意一者不匹配，此要点即不匹配
7. 完整性检查：确保所有医学要点都被正确识别和匹配，不得有遗漏或错误识别
8. 排除非医学内容：忽略连接词、修饰词等非医学核心内容

质量控制要求：
- 要点识别必须100%覆盖所有医学专业内容
- 严禁将不同的医学概念归并为同一要点
- 每个要点必须具有明确的医学意义
- 对应关系建立必须基于严格的医学逻辑

输出要求：
- 输出必须是严格的JSON格式，不要包含任何其他无关内容
- predict_points和label_points必须包含所有识别出的医学要点（包括所有小要点）
- correspondences数组表示要点间的精确对应关系，索引从0开始
- 对于层次化要点，所有层级的要点都应该单独列出并建立准确的对应关系
""".strip()

STAGE2_POINT_SYSTEM_RELAXED_DEGREE = """
你是一位资深的医学评估专家，擅长分析诊断信息的要点对应关系。
你的任务是评估predict和label中的诊断信息要点之间的对应关系。

要点识别规则（严格执行）：
-该文本为脑颈动脉超声检查结果，分为颅内和颅外两部分，请对其都进行要点提取
-要点多为某根脑劲动脉加患病症状，例如’双侧颈动脉球部狭窄（右侧：50%-69%；左侧：小于50%）‘

要点匹配原则（忽略程度版，严谨执行）：
1. 医学术语匹配：严格按照医学专业术语的准确性进行匹配，尤其注意左右侧、血管名称、患病类型的匹配
2. 同义词识别：识别医学同义词和近义词（如：心肌梗死=心梗，高血压=血压升高）
3. 建立对应关系时，不考虑括号内表示病变程度的内容，例如“小于50%”“50%-69%”“70%-99%”“轻度”“中度”“重度”“中-重度”等程度差异不影响匹配
4. 如果括号内包含左右侧信息，左右侧信息仍必须严格对应；只忽略程度，不忽略左右侧、血管名称和病变类型
5. 严格判断起始左右侧血管是否对应，例如’左侧 carotid动脉狭窄‘与’右侧 carotid动脉狭窄‘不匹配
6. 左右侧、血管名称、病变类型有任意一者不匹配，此要点即不匹配；仅病变程度不同仍可匹配
7. 完整性检查：确保所有医学要点都被正确识别和匹配，不得有遗漏或错误识别
8. 排除非医学内容：忽略连接词、修饰词等非医学核心内容

质量控制要求：
- 要点识别必须100%覆盖所有医学专业内容
- 严禁将不同的医学概念归并为同一要点
- 每个要点必须具有明确的医学意义
- 对应关系建立必须基于严格的医学逻辑

输出要求：
- 输出必须是严格的JSON格式，不要包含任何其他无关内容
- predict_points和label_points必须包含所有识别出的医学要点（包括所有小要点）
- correspondences数组表示要点间的精确对应关系，索引从0开始
- 对于层次化要点，所有层级的要点都应该单独列出并建立准确的对应关系
""".strip()

STAGE2_POINT_SYSTEM_RELAXED_FROM_POINTS = """
你是一位资深的医学评估专家。你的任务是：在给定的 predict_points 与 label_points 列表上，建立对应关系 correspondences。

重要约束（必须遵守）：
1) predict_points 与 label_points 已由严格版本抽取得到；你不得新增/删除/改写任何要点文本，只能建立对应关系。
2) 只输出 correspondences（二维数组，元素为 [predict_index, label_index]），索引从 0 开始。
3) relaxed 规则：建立对应关系时，忽略括号内表示病变程度的内容（如“小于50%”“50%-69%”“70%-99%”“轻度/中度/重度”“中-重度”等），但左右侧、血管名称、病变类型仍必须严格一致。
4) 如果括号内包含左右侧信息，左右侧仍必须严格对应；只忽略“程度”，不忽略左右侧、血管名称和病变类型。
5) 一个 predict 要点可以对应多个 label 要点（例如“椎-基底动脉系统血流正常”可同时覆盖“椎动脉正常”“基底动脉正常”），但不得牵强匹配不相关概念。

输出必须是严格 JSON：
{"correspondences": [[0, 0], [1, 2]]}
""".strip()


def _strip_think_blocks(text: str) -> str:
    if not text:
        return text
    pat = r"<" + "think" + r">[\s\S]*?</" + "think" + r">\s*"
    text = re.sub(pat, "", text, flags=re.IGNORECASE)
    # 有些模型会只输出开标签（无 </think>），此时直接截断到末尾。
    pat_open_to_end = r"<" + "think" + r">[\s\S]*$"
    text = re.sub(pat_open_to_end, "", text, flags=re.IGNORECASE)
    return text.strip()


def _normalize_point_eval_dict(ev: Dict[str, Any]) -> Dict[str, Any]:
    c_raw = ev.get("correspondences") or []
    pairs: List[List[int]] = []
    for item in c_raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append([int(item[0]), int(item[1])])
    return {
        "predict_points": [str(x) for x in (ev.get("predict_points") or [])],
        "label_points": [str(x) for x in (ev.get("label_points") or [])],
        "correspondences": pairs,
    }


def _normalize_correspondences_only(obj: Any) -> List[List[int]]:
    if not isinstance(obj, dict):
        raise ValueError("relaxed-from-points 返回不是对象")
    c_raw = obj.get("correspondences")
    if not isinstance(c_raw, list):
        raise ValueError("relaxed-from-points 缺少 correspondences 列表")
    out: List[List[int]] = []
    for it in c_raw:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            out.append([int(it[0]), int(it[1])])
    return out


def _try_get_points_from_point_eval(ev: Any) -> Optional[Tuple[List[str], List[str]]]:
    if not isinstance(ev, dict):
        return None
    pp = ev.get("predict_points")
    lp = ev.get("label_points")
    if not isinstance(pp, list) or not isinstance(lp, list):
        return None
    return ([str(x) for x in pp], [str(x) for x in lp])


def _resolve_bailian_api_key(explicit: Optional[str], key_file: Optional[str]) -> str:
    if explicit:
        return explicit.strip()
    env = (os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if env:
        return env
    paths = []
    if key_file:
        paths.append(Path(key_file).expanduser())
    paths.append(Path.home() / ".bailian_api_key")
    for p in paths:
        if p.is_file():
            v = p.read_text(encoding="utf-8").strip().splitlines()
            if v:
                return v[0].strip()
    return ""


def _require_ascii_http_token(name: str, value: str) -> str:
    v = (value or "").strip()
    if v.startswith("\ufeff"):
        v = v[1:].strip()
    try:
        v.encode("ascii")
    except UnicodeEncodeError:
        bad_preview = "".join(c if ord(c) < 128 else "?" for c in v[:32])
        raise ValueError(
            f"{name} 含非 ASCII 字符，无法用 OpenAI/httpx 作为请求头发送。"
            "常见原因：把中文说明写进了 ~/.bailian_api_key、环境变量里混入了全角字符、"
            "或复制了「Key：xxxx」整行。请只保留纯英文/数字的 key 字符串。"
            f"（预览已脱敏非 ASCII：{bad_preview!r}）"
        ) from None
    return v


def manual_parse_and_match(predict_text: str, label_text: str) -> Dict[str, Any]:
    predict_pattern = r"(\d+\.)([^\d]+)(?=\d+\.|$)"
    label_pattern = r"(\d+\.)([^\d]+)(?=\d+\.|$)"
    predict_matches = re.findall(predict_pattern, predict_text)
    label_matches = re.findall(label_pattern, label_text)
    predict_points = [f"{num}{content}" for num, content in predict_matches]
    label_points = [f"{num}{content}" for num, content in label_matches]
    correspondences: List[List[int]] = []
    for p_idx, p_point in enumerate(predict_points):
        for l_idx, l_point in enumerate(label_points):
            if (
                p_point in l_point
                or l_point in p_point
                or any(word in l_point for word in p_point.split() if len(word) > 2)
                or any(word in p_point for word in l_point.split() if len(word) > 2)
            ):
                correspondences.append([p_idx, l_idx])
                break
    return {
        "predict_points": predict_points,
        "label_points": label_points,
        "correspondences": correspondences,
    }


def calculate_sample_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    # 注意：这里的“准确率”按信息抽取常用口径取 precision（匹配到的预测要点 / 预测要点总数）。
    sample_precisions: List[float] = []
    sample_recalls: List[float] = []
    sample_f1s: List[float] = []
    for result in evaluation_results:
        if not isinstance(result, dict):
            continue
        predict_points = result["predict_points"]
        label_points = result["label_points"]
        correspondences = result["correspondences"]
        if len(predict_points) == 0 and len(label_points) == 0:
            sample_precisions.append(1.0)
            sample_recalls.append(1.0)
            sample_f1s.append(1.0)
        elif len(predict_points) == 0 or len(label_points) == 0:
            sample_precisions.append(0.0)
            sample_recalls.append(0.0)
            sample_f1s.append(0.0)
        else:
            matched_predict = len(set(corr[0] for corr in correspondences))
            matched_label = len(set(corr[1] for corr in correspondences))
            precision = matched_predict / len(predict_points) if predict_points else 0.0
            recall = matched_label / len(label_points) if label_points else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            sample_precisions.append(precision)
            sample_recalls.append(recall)
            sample_f1s.append(f1)
    return {
        "sample_precisions": sample_precisions,
        "sample_recalls": sample_recalls,
        "sample_f1s": sample_f1s,
    }


def calculate_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    二阶段要点评测指标：
    - 单样本：只计算 precision(准确率)/recall/F1（不算 micro/macro 等汇总口径）
    - 多样本整体：按“样本平均”汇总（在 save_eval_artifacts 里对 per-sample 结果取均值）
    """
    sample_metrics = calculate_sample_metrics(evaluation_results)
    if not sample_metrics["sample_precisions"]:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    # 这里返回“样本平均”口径，便于单样本/多样本统一复用
    return {
        "precision": float(sum(sample_metrics["sample_precisions"]) / len(sample_metrics["sample_precisions"])),
        "recall": float(sum(sample_metrics["sample_recalls"]) / len(sample_metrics["sample_recalls"])),
        "f1": float(sum(sample_metrics["sample_f1s"]) / len(sample_metrics["sample_f1s"])),
    }


def _evaluate_stage2_points_once(
    pred_s2: str,
    label_s2: str,
    args,
    bailian_key: str,
    system_content: str,
    *,
    fallback_note: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    raw: Optional[str] = None
    bailian_err: Optional[str] = None
    if bailian_key:
        try:
            raw = evaluate_points_bailian(
                pred_s2,
                label_s2,
                bailian_key,
                args.bailian_base_url,
                args.bailian_model,
                system_content=system_content,
            )
        except ValueError as e:
            bailian_err = f"bailian_config_error: {e}"

    if raw:
        try:
            ev = _normalize_point_eval_dict(json.loads(raw))
            return calculate_metrics([ev]), ev, None, None
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            ev_fb = manual_parse_and_match(pred_s2, label_s2)
            return (
                {"parse_error": "invalid_json", "raw_preview": raw[:500]},
                ev_fb,
                None,
                calculate_metrics([ev_fb]),
            )

    ev_fb = manual_parse_and_match(pred_s2, label_s2)
    return calculate_metrics([ev_fb]), ev_fb, bailian_err or fallback_note, None


def _has_stage2_prf_metrics(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    return all(isinstance(metrics.get(k), (int, float)) and not isinstance(metrics.get(k), bool) for k in ("precision", "recall", "f1"))


def evaluate_points_bailian(
    predict_text: str,
    label_text: str,
    api_key: str,
    base_url: str,
    model: str,
    system_content: str = STAGE2_POINT_SYSTEM,
) -> Optional[str]:
    if OpenAI is None:
        return None
    api_key = _require_ascii_http_token("百炼 API Key", api_key)
    base_url = _require_ascii_http_token("百炼 base_url", base_url)
    model = _require_ascii_http_token("百炼 model", model)
    user_content = f"""
请评估以下predict和label中的诊断信息要点对应关系：
predict: {predict_text}
label: {label_text}

请判断predict中的每个要点与label中的哪些要点对应，并以JSON格式输出评估结果。
输出格式要求：
{{"predict_points": ["要点1", "要点2", ...], "label_points": ["要点1", "要点2", ...], "correspondences": [[predict_index, label_index], ...]}}

其中correspondences是一个二维数组，表示predict中的第几个要点与label中的第几个要点对应，索引从0开始。
如果没有对应关系，correspondences为空数组。
"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    chat_outputs = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return chat_outputs.choices[0].message.content


def evaluate_correspondences_bailian_from_points(
    predict_points: List[str],
    label_points: List[str],
    api_key: str,
    base_url: str,
    model: str,
    system_content: str = STAGE2_POINT_SYSTEM_RELAXED_FROM_POINTS,
) -> Optional[str]:
    """只在给定 points 上建立 correspondences（不允许改变 points 粒度）。"""
    if OpenAI is None:
        return None
    api_key = _require_ascii_http_token("百炼 API Key", api_key)
    base_url = _require_ascii_http_token("百炼 base_url", base_url)
    model = _require_ascii_http_token("百炼 model", model)
    payload = {
        "predict_points": [str(x) for x in (predict_points or [])],
        "label_points": [str(x) for x in (label_points or [])],
    }
    user_content = "请根据以下 points 建立 correspondences（索引从0开始）。\n\n" + json.dumps(payload, ensure_ascii=False)
    client = OpenAI(api_key=api_key, base_url=base_url)
    chat_outputs = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return chat_outputs.choices[0].message.content


def compute_stage1_text_metrics(
    pred: str,
    ref: str,
    bert_model_path: str,
    bertscore_device: str = "cuda",
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not (ref or "").strip():
        metrics["skipped"] = True
        return metrics
    try:
        # 与 local text metrics 对齐：
        # - 字符级 token（中文更稳定）
        # - NLTK sentence_bleu + method3 smoothing
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        pred_tokens = list((pred or "").strip())
        ref_tokens = list((ref or "").strip())
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            metrics["bleu"] = 0.0
        else:
            smoothing = SmoothingFunction().method3
            metrics["bleu"] = float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing))
    except Exception as e:
        metrics["bleu_error"] = repr(e)
    try:
        from rouge_score import rouge_scorer

        pred_s = " ".join(list((pred or "").strip()))
        ref_s = " ".join(list((ref or "").strip()))
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        rs = scorer.score(ref_s, pred_s)
        metrics["rouge1_f"] = float(rs["rouge1"].fmeasure)
        metrics["rouge2_f"] = float(rs["rouge2"].fmeasure)
        metrics["rougeL_f"] = float(rs["rougeL"].fmeasure)
    except Exception as e:
        metrics["rouge_error"] = repr(e)
    try:
        import torch
        from transformers import AutoConfig
        from bert_score import score as bert_score_fn

        device_pref = (bertscore_device or "cuda").strip().lower()
        if device_pref == "cuda" and not torch.cuda.is_available():
            device_pref = "cpu"
        model_type = bert_model_path
        num_layers = None
        # bert-score 对自定义本地目录的 model_type 可能会尝试在内置映射表里查层数，导致 KeyError。
        # 若给出本地目录，则从 config 里推断 num_hidden_layers 并显式传入。
        try:
            p = Path(bert_model_path)
            if p.exists():
                cfg = AutoConfig.from_pretrained(str(p))
                n = getattr(cfg, "num_hidden_layers", None)
                if isinstance(n, int) and n > 0:
                    num_layers = n
        except Exception:
            num_layers = None
        def _run_bertscore(device_name: str):
            return bert_score_fn(
                [pred],
                [ref],
                lang="zh",
                model_type=model_type,
                num_layers=num_layers,
                verbose=False,
                device=device_name,
            )

        try:
            P, R, F1 = _run_bertscore(device_pref)
        except Exception as e:
            # 默认走 GPU；若显存不足则自动回退 CPU，避免整轮评测失败。
            err_txt = repr(e)
            oom_like = "out of memory" in err_txt.lower() or "cuda error" in err_txt.lower()
            if device_pref == "cuda" and oom_like:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                P, R, F1 = _run_bertscore("cpu")
                metrics["bertscore_device_fallback"] = "cpu_after_cuda_oom"
            else:
                raise
        metrics["bertscore_precision"] = float(P.mean())
        metrics["bertscore_recall"] = float(R.mean())
        metrics["bertscore_f1"] = float(F1.mean())
    except Exception as e:
        metrics["bertscore_error"] = repr(e)
    return metrics


def _infer_bertscore_num_layers(bert_model_path: str) -> Optional[int]:
    try:
        from transformers import AutoConfig

        p = Path(bert_model_path)
        if p.exists():
            cfg = AutoConfig.from_pretrained(str(p))
            n = getattr(cfg, "num_hidden_layers", None)
            if isinstance(n, int) and n > 0:
                return n
    except Exception:
        return None
    return None


def compute_bertscore_batch(
    preds: List[str],
    refs: List[str],
    bert_model_path: str,
    bertscore_device: str = "cuda",
    batch_size: int = 32,
) -> Tuple[List[float], List[float], List[float], Dict[str, Any]]:
    """
    批量计算 BERTScore（比逐条 score() 快很多）。
    返回：P_list, R_list, F1_list，以及 meta 信息（device / fallback 等）。
    """
    if len(preds) != len(refs):
        raise ValueError("preds/refs 长度必须一致")
    meta: Dict[str, Any] = {}
    if not preds:
        return [], [], [], meta

    import torch
    from bert_score import score as bert_score_fn

    device_pref = (bertscore_device or "cuda").strip().lower()
    if device_pref == "cuda" and not torch.cuda.is_available():
        device_pref = "cpu"
    num_layers = _infer_bertscore_num_layers(bert_model_path)

    bs = int(batch_size or 1)
    if bs <= 0:
        bs = 1

    def _run(device_name: str, bs_now: int):
        return bert_score_fn(
            preds,
            refs,
            lang="zh",
            model_type=bert_model_path,
            num_layers=num_layers,
            verbose=False,
            device=device_name,
            batch_size=bs_now,
        )

    try:
        P, R, F1 = _run(device_pref, bs)
        meta["bertscore_device"] = device_pref
        meta["bertscore_batch_size"] = bs
    except Exception as e:
        err_txt = repr(e)
        oom_like = "out of memory" in err_txt.lower() or "cuda error" in err_txt.lower()
        # CUDA OOM：优先自动降 batch_size 重试（仍用 GPU），再不行回退 CPU。
        if device_pref == "cuda" and oom_like:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            bs_try = max(1, bs // 2)
            while bs_try >= 1:
                try:
                    P, R, F1 = _run("cuda", bs_try)
                    meta["bertscore_device"] = "cuda"
                    meta["bertscore_batch_size"] = bs_try
                    meta["bertscore_batch_size_fallback"] = "reduced_after_oom"
                    break
                except Exception as e2:
                    if "out of memory" in repr(e2).lower() or "cuda error" in repr(e2).lower():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if bs_try == 1:
                            # 最后回退 CPU
                            P, R, F1 = _run("cpu", 1)
                            meta["bertscore_device"] = "cpu"
                            meta["bertscore_batch_size"] = 1
                            meta["bertscore_device_fallback"] = "cpu_after_cuda_oom"
                            break
                        bs_try = max(1, bs_try // 2)
                        continue
                    raise
        else:
            raise

    P_list = [float(x) for x in P.cpu().tolist()]
    R_list = [float(x) for x in R.cpu().tolist()]
    F1_list = [float(x) for x in F1.cpu().tolist()]
    return P_list, R_list, F1_list, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM 多 LoRA 两阶段测试：stage1 -> stage2（本地 vLLM）")
    cur_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parser.add_argument("--base-model", type=str, default="./models/base_model", help="基座模型路径")
    parser.add_argument("--stage1-adapter", type=str, default="./adapters/stage1", help="第一阶段 LoRA 适配器路径")
    parser.add_argument("--stage2-adapter", type=str, default="./adapters/stage2", help="第二阶段 LoRA 适配器路径")
    parser.add_argument(
        "--stage1-data",
        type=str,
        default="./data/stage1_examples.json",
        help="第一阶段数据 JSON（列表，含 instruction/input/output）",
    )
    parser.add_argument(
        "--stage2-data",
        type=str,
        default="./data/stage2_examples.json",
        help="第二阶段数据 JSON（列表，含 instruction/input/output）",
    )
    parser.add_argument("--start", type=int, default=0, help="从第几条开始")
    parser.add_argument("--limit", type=int, default=5, help="处理多少条；<=0 表示全部")
    parser.add_argument("--temperature1", type=float, default=0.0, help="第一阶段温度（默认 0：更贴近 LLaMA-Factory 评测常见设置）")
    parser.add_argument("--temperature2", type=float, default=0.2, help="第二阶段温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p（默认 1.0：关闭 nucleus 采样，更贴近 greedy/beam 风格评测）")
    parser.add_argument("--max-tokens1", type=int, default=2048, help="第一阶段最大生成 token 数")
    parser.add_argument("--max-tokens2", type=int, default=1024, help="第二阶段最大生成 token 数")
    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=1,
        help="vLLM 每次 generate 的微批大小（批量过大会 OOM；默认 1 最稳）",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96, help="GPU 显存利用率（0-1）")
    parser.add_argument("--max-model-len", type=int, default=10000, help="最大上下文长度（建议显存紧张时调小，例如 2048/4096/8192）")
    parser.add_argument("--enforce-eager", action="store_true", help="启用 eager 模式（可降低编译/图缓存开销，显存紧张时建议开启）")
    parser.add_argument("--dtype", type=str, default="auto", help="模型 dtype：auto/half/bfloat16/float16/float32")
    parser.add_argument("--trust-remote-code", action="store_true", help="是否启用 trust_remote_code")
    parser.add_argument("--seed", type=int, default=0, help="采样随机种子")
    parser.add_argument("--output-jsonl", type=Path, default=cur_dir / f"vllm_two_stage_test_{timestamp}.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--output-xlsx", type=Path, default=cur_dir / f"vllm_two_stage_test_{timestamp}.xlsx", help="输出 Excel 路径")
    parser.add_argument(
        "--think",
        action="store_true",
        help="允许模型输出思考过程（默认禁止：在 system 追加禁止思考提示并剥离输出中的思考块）",
    )
    parser.set_defaults(eval=True)
    parser.add_argument(
        "--no-eval",
        action="store_false",
        dest="eval",
        help="关闭测评（默认开启：一阶段 BLEU/ROUGE/BERTScore；二阶段要点评测优先百炼，缺 key 则降级为本地规则匹配）",
    )
    parser.set_defaults(eval_stage1=True, eval_stage2=True)
    parser.add_argument(
        "--no-eval-stage1",
        action="store_false",
        dest="eval_stage1",
        help="关闭 stage1 测评（BLEU/ROUGE/BERTScore）。",
    )
    parser.add_argument(
        "--no-eval-stage2",
        action="store_false",
        dest="eval_stage2",
        help="关闭 stage2 测评（逐条大模型/本地匹配要点评测）。",
    )
    parser.add_argument(
        "--eval-stage2-relaxed-only",
        action="store_true",
        help="只增量补充 stage2 忽略括号内程度的 relaxed 测评，保留已有严格测评结果。",
    )
    parser.add_argument(
        "--force-relaxed-eval",
        action="store_true",
        help="即使已有 stage2.metrics_relaxed，也强制重算 relaxed 测评。",
    )
    parser.add_argument(
        "--bert-model-path",
        type=str,
        default="./models/bert",
        help="BERTScore 使用的本地中文模型目录（bert-score model_type）",
    )
    parser.add_argument(
        "--bertscore-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="BERTScore 计算设备，默认 cuda（若 CUDA OOM 会自动回退到 CPU）",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=32,
        help="BERTScore 批量计算 batch_size（越大越快但更占显存；默认 32，CUDA OOM 会自动降低再重试）",
    )
    parser.add_argument(
        "--bailian-base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="百炼 OpenAI 兼容接口 base_url",
    )
    parser.add_argument("--bailian-model", type=str, default="qwen3.5-27b", help="百炼上要点评测所用模型名")
    parser.add_argument("--bailian-api-key", type=str, default="", help="百炼 API Key（也可用环境变量 BAILIAN_API_KEY / DASHSCOPE_API_KEY 或 ~/.bailian_api_key）")
    parser.add_argument("--bailian-api-key-file", type=str, default="", help="从文件读取百炼 API Key（首行有效）")
    parser.add_argument(
        "--eval-from-jsonl",
        type=str,
        default="",
        help="从已有回答 JSONL 直接开始测评（跳过生成），并将测评结果回写到该 JSONL。",
    )
    parser.add_argument(
        "--chat-template-file",
        type=str,
        default="./configs/qwen3_nothink.jinja",
        help=(
            "可选：从文件读取 chat_template（Jinja 字符串），覆盖 tokenizer 自带模板。"
            "用于复刻 LLaMA-Factory 的 `qwen3_nothink` 等自定义模板；"
            "默认使用 ./configs/qwen3_nothink.jinja。"
        ),
    )
    return parser


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"数据必须是 JSON 列表：{path}")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 条不是对象：{path}")
        out.append(item)
    return out


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"JSONL 文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {i} 行 JSON 解析失败：{e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path} 第 {i} 行不是对象")
            rows.append(obj)
    return rows


def _build_messages(system_prompt: str, user_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_finetune_style_messages(instruction: str, payload: str) -> List[Dict[str, str]]:
    """
    贴近微调/离线验证时的输入格式：把 instruction 与 input 拼成同一条 user 消息。
    你的 generated_predictions.jsonl 里就是这种：<|im_start|>user\n\n{instruction}\n\n{input}...<|im_end|>
    """
    ins = (instruction or "").strip()
    body = (payload or "").strip()
    if ins and body:
        content = ins + "\n\n" + body
    else:
        content = ins or body
    return [{"role": "user", "content": content}]


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], *, enable_thinking: Optional[bool] = None) -> str:
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        # Qwen3 系列 chat_template 支持 enable_thinking 开关；传入可从源头关闭 think。
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        return apply_fn(messages, **kwargs)

    sys_text = "\n".join([m["content"] for m in messages if m.get("role") == "system"])
    user_text = "\n".join([m["content"] for m in messages if m.get("role") == "user"])
    return f"<|system|>\n{sys_text}\n<|user|>\n{user_text}\n<|assistant|>\n"


def _make_lora_request(lora_name: str, lora_id: int, lora_path: str):
    if LoRARequest is None:
        raise RuntimeError("无法导入 vllm.lora.request.LoRARequest，请确认 vLLM 版本支持 LoRA。")
    sig = inspect.signature(LoRARequest)
    params = sig.parameters
    if "lora_local_path" in params:
        return LoRARequest(lora_name=lora_name, lora_int_id=lora_id, lora_local_path=lora_path)
    if "lora_path" in params:
        return LoRARequest(lora_name=lora_name, lora_int_id=lora_id, lora_path=lora_path)
    return LoRARequest(lora_name, lora_id, lora_path)


def _maybe_override_chat_template(tokenizer, chat_template_file: str) -> None:
    p = (chat_template_file or "").strip()
    if not p:
        return
    path = Path(p).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"--chat-template-file 不存在或不是文件：{path}")
    tpl = path.read_text(encoding="utf-8")
    if tpl.startswith("\ufeff"):
        tpl = tpl[1:]
    tpl = tpl.strip()
    if not tpl:
        raise ValueError(f"--chat-template-file 内容为空：{path}")
    tokenizer.chat_template = tpl


class TwoStageVLLMRunner:
    def __init__(
        self,
        base_model: str,
        stage1_adapter: str,
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
        _maybe_override_chat_template(self.tokenizer, chat_template_file)
        base_kwargs = {
            "model": base_model,
            "enable_lora": True,
            "max_loras": 2,
            "max_lora_rank": 64,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "max_model_len": max_model_len,
        }

        print(
            f"LLM 初始化参数：gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}, "
            f"enforce_eager={enforce_eager}"
        )
        try:
            self.llm = LLM(**base_kwargs, enforce_eager=enforce_eager)
        except TypeError:
            base_kwargs.pop("max_lora_rank", None)
            self.llm = LLM(**base_kwargs, enforce_eager=enforce_eager)
        except Exception as e:
            msg = str(e)
            if (
                "No available memory for the cache blocks" in msg
                or "available KV cache memory" in msg
                or "max seq len" in msg
                or "models's max seq len" in msg
                or "Engine core initialization failed" in msg
            ):
                raise RuntimeError(
                    "vLLM 初始化失败（通常是 KV cache 显存不足/上下文长度过大）。"
                    "请优先尝试：--max-model-len 4096 或 2048（或更小），并配合 --gpu-memory-utilization 0.97/0.98 以及 --enforce-eager。"
                ) from e
            raise

        try:
            actual_len = self.llm.llm_engine.model_config.max_model_len
            print(f"vLLM 实际 max_model_len：{actual_len}")
        except Exception:
            pass
        self.lora_stage1 = _make_lora_request("stage1", 1, stage1_adapter)
        self.lora_stage2 = _make_lora_request("stage2", 2, stage2_adapter)

    def generate(
        self,
        messages: List[Dict[str, str]],
        sampling_params: SamplingParams,
        lora_request,
        *,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        prompt = _apply_chat_template(self.tokenizer, messages, enable_thinking=enable_thinking)
        try:
            outputs = self.llm.generate([prompt], sampling_params=sampling_params, lora_request=lora_request)
        except TypeError:
            outputs = self.llm.generate([prompt], sampling_params=sampling_params, lora_requests=[lora_request])
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        lora_requests: List[Any],
        *,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        if not messages_list:
            return []
        if len(lora_requests) != len(messages_list):
            raise ValueError("lora_requests 长度必须与 messages_list 一致（一 prompt 一 LoRA）。")
        prompts = [_apply_chat_template(self.tokenizer, m, enable_thinking=enable_thinking) for m in messages_list]
        try:
            outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_requests,
            )
        except TypeError:
            outputs = self.llm.generate(
                prompts,
                sampling_params,
                lora_requests=lora_requests,
            )
        texts: List[str] = []
        for i, out in enumerate(outputs or []):
            if out.outputs:
                texts.append(out.outputs[0].text)
            else:
                texts.append("")
        while len(texts) < len(messages_list):
            texts.append("")
        return texts


def save_outputs_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_outputs_xlsx(path: Path, rows: List[Dict[str, Any]]):
    if pd is None:
        raise RuntimeError("未安装 pandas，无法保存 .xlsx。请先安装 pandas（及其 Excel 依赖）或改用 JSONL 输出。")
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows: List[Dict[str, Any]] = []
    for r in rows:
        stage1 = r.get("stage1") or {}
        stage2 = r.get("stage2") or {}
        flat_rows.append(
            {
                "id": r.get("id"),
                "stage1_instruction": stage1.get("instruction"),
                "stage1_input": stage1.get("input"),
                "stage1_gt": stage1.get("gt"),
                "stage1_pred": stage1.get("pred"),
                "stage2_instruction": stage2.get("instruction"),
                "stage2_input": stage2.get("input"),
                "stage2_gt": stage2.get("gt"),
                "stage2_pred": stage2.get("pred"),
                "stage2_metrics": stage2.get("metrics"),
                "stage2_metrics_relaxed": stage2.get("metrics_relaxed"),
            }
        )
    df = pd.DataFrame(flat_rows)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="two_stage", index=False)


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _collect_numeric(d: Dict[str, Any], key: str) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def save_eval_artifacts(base_jsonl_path: Path, rows: List[Dict[str, Any]]):
    """
    额外保存两类评估产物：
    1) per-sample 指标 JSONL：<name>_eval_samples.jsonl
    2) overall 汇总 JSON：<name>_eval_summary.json
    """
    samples_path = base_jsonl_path.with_name(base_jsonl_path.stem + "_eval_samples.jsonl")
    summary_path = base_jsonl_path.with_name(base_jsonl_path.stem + "_eval_summary.json")

    per_sample: List[Dict[str, Any]] = []
    stage1_vals: Dict[str, List[float]] = {
        "bleu": [],
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    }
    stage2_vals: Dict[str, List[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
    }
    stage2_relaxed_vals: Dict[str, List[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
    }

    # 如果保存了 point_eval（predict_points/label_points/correspondences），可用它重算 per-sample PRF 后再取样本平均
    stage2_point_evs: List[Dict[str, Any]] = []

    for r in rows:
        sid = r.get("id")
        s1 = (r.get("stage1") or {}) if isinstance(r, dict) else {}
        s2 = (r.get("stage2") or {}) if isinstance(r, dict) else {}
        s1m = s1.get("metrics") if isinstance(s1, dict) else None
        s2m = s2.get("metrics") if isinstance(s2, dict) else None
        s2m_relaxed = s2.get("metrics_relaxed") if isinstance(s2, dict) else None
        note = s2.get("metrics_note") if isinstance(s2, dict) else None
        note_relaxed = s2.get("metrics_relaxed_note") if isinstance(s2, dict) else None
        point_ev = s2.get("point_eval") if isinstance(s2, dict) else None
        if isinstance(point_ev, dict) and {"predict_points", "label_points", "correspondences"} <= set(point_ev.keys()):
            stage2_point_evs.append(_normalize_point_eval_dict(point_ev))

        per_sample.append(
            {
                "id": sid,
                "stage1_metrics": s1m,
                "stage2_metrics": s2m,
                "stage2_metrics_relaxed": s2m_relaxed,
                "stage2_metrics_note": note,
                "stage2_metrics_relaxed_note": note_relaxed,
            }
        )

        for k in list(stage1_vals.keys()):
            v = _collect_numeric(s1m, k) if isinstance(s1m, dict) else None
            if v is not None:
                stage1_vals[k].append(v)

        for k in list(stage2_vals.keys()):
            v = _collect_numeric(s2m, k) if isinstance(s2m, dict) else None
            if v is not None:
                stage2_vals[k].append(v)

        for k in list(stage2_relaxed_vals.keys()):
            v = _collect_numeric(s2m_relaxed, k) if isinstance(s2m_relaxed, dict) else None
            if v is not None:
                stage2_relaxed_vals[k].append(v)

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(samples_path, "w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stage1_summary = {k: _mean(vs) for k, vs in stage1_vals.items()}
    stage2_summary_avg = {k: _mean(vs) for k, vs in stage2_vals.items()}
    stage2_relaxed_summary_avg = {k: _mean(vs) for k, vs in stage2_relaxed_vals.items()}
    stage2_summary_method = "avg_of_per_sample_metrics"

    summary = {
        "total_samples": len(rows),
        "stage1_avg": stage1_summary,
        "stage2_overall_method": stage2_summary_method,
        "stage2_avg": stage2_summary_avg,
        "stage2_relaxed_degree_avg": stage2_relaxed_summary_avg,
        "stage2_relaxed_degree_note": "ignore disease degree in parentheses; still match side/vessel/disease type",
        "eval_samples_jsonl": str(samples_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))


def _slice_stage1_rows(
    stage1_rows: List[Dict[str, Any]],
    start: int,
    limit: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    n = len(stage1_rows)
    if start < 0:
        start = 0
    end = n if limit <= 0 else min(n, start + limit)
    return [(i, stage1_rows[i]) for i in range(start, end)]


def _is_missing_stage1_output(row: Dict[str, Any]) -> bool:
    output = row.get("output")
    if output is None:
        return True
    text = str(output).strip()
    return not text or text.lower() == "nan"


def _slice_aligned_stage1_rows(
    stage1_rows: List[Dict[str, Any]],
    start: int,
    limit: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    stage2_data-new.json 是从 stage1 数据中去掉空 output 样本后得到的。
    因此两阶段串联时必须先过滤 stage1 空 output，再按过滤后的顺序和 stage2 对齐。

    返回：(stage2_index, original_stage1_index, stage1_row)
    """
    filtered = [(i, row) for i, row in enumerate(stage1_rows) if not _is_missing_stage1_output(row)]
    if start < 0:
        start = 0
    end = len(filtered) if limit <= 0 else min(len(filtered), start + limit)
    return [(stage2_idx, orig_idx, row) for stage2_idx, (orig_idx, row) in enumerate(filtered[start:end], start=start)]


def _pick_stage2_instruction(stage2_rows: List[Dict[str, Any]]) -> str:
    for item in stage2_rows:
        ins = str(item.get("instruction") or "").strip()
        if ins:
            return ins
    raise ValueError("第二阶段数据中没有找到非空 instruction。")


def _stage2_instruction_for_index(stage2_rows: List[Dict[str, Any]], idx: int, fallback: str) -> str:
    if 0 <= idx < len(stage2_rows):
        ins = str(stage2_rows[idx].get("instruction") or "").strip()
        if ins:
            return ins
    return fallback


def _evaluate_outputs_inplace(outputs: List[Dict[str, Any]], args, bailian_key: str, *, jsonl_path: Optional[Path] = None):
    print("\n开始评测：先释放 vLLM 显存，再计算 BERTScore/二阶段指标...")
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if getattr(args, "eval_stage1", True):
        # --- stage1：先算 BLEU/ROUGE（逐条轻量），再批量算 BERTScore ---
        s1_preds: List[str] = []
        s1_refs: List[str] = []
        s1_row_indices: List[int] = []

        for i, row in enumerate(outputs):
            s1 = row.get("stage1") or {}
            p1 = str(s1.get("pred") or "")
            s1_gt = s1.get("gt")
            s1_ref = str(s1_gt) if s1_gt is not None else ""

            # 先初始化 metrics（至少包含 bleu/rouge 或 skipped）
            m = compute_stage1_text_metrics(
                p1,
                s1_ref,
                args.bert_model_path,
                bertscore_device="cpu",  # 这里仅用来复用 bleu/rouge 逻辑；BERTScore 后面批量算
            )
            # 上面调用会尝试算 BERTScore；我们不需要逐条算，因此清掉 bertscore 相关字段
            m.pop("bertscore_precision", None)
            m.pop("bertscore_recall", None)
            m.pop("bertscore_f1", None)
            m.pop("bertscore_error", None)
            m.pop("bertscore_device_fallback", None)
            m.pop("bertscore_meta", None)
            s1["metrics"] = m

            if (s1_ref or "").strip():
                s1_preds.append(p1)
                s1_refs.append(s1_ref)
                s1_row_indices.append(i)

        if s1_preds:
            try:
                P_list, R_list, F1_list, meta = compute_bertscore_batch(
                    s1_preds,
                    s1_refs,
                    args.bert_model_path,
                    bertscore_device=args.bertscore_device,
                    batch_size=int(args.bertscore_batch_size or 32),
                )
                for j, row_i in enumerate(s1_row_indices):
                    s1m = (outputs[row_i].get("stage1") or {}).get("metrics") or {}
                    s1m["bertscore_precision"] = P_list[j]
                    s1m["bertscore_recall"] = R_list[j]
                    s1m["bertscore_f1"] = F1_list[j]
                    # 仅写入一次 meta（避免每条重复）
                    if meta and j == 0:
                        s1m["bertscore_meta"] = meta
            except Exception as e:
                # 批量失败时，不中断整个评测；在有 ref 的样本上记录错误信息
                for row_i in s1_row_indices:
                    s1m = (outputs[row_i].get("stage1") or {}).get("metrics") or {}
                    s1m["bertscore_error"] = repr(e)
    else:
        # stage1 测评关闭时：保留已有 stage1.metrics（便于与其他阶段结果合并）
        pass

    # stage1 评测完成后落盘一次，确保中断可续跑
    if jsonl_path is not None:
        try:
            save_outputs_jsonl(jsonl_path, outputs)
        except Exception as e:
            print(f"警告：stage1 评测后写入 JSONL 失败：{e}")

    # --- stage2：保持逐条大模型/本地匹配测评逻辑不变（加进度条） ---
    if getattr(args, "eval_stage2", True):
        relaxed_only = bool(getattr(args, "eval_stage2_relaxed_only", False))
        force_relaxed = bool(getattr(args, "force_relaxed_eval", False))
        it = outputs
        if tqdm is not None:
            it = tqdm(outputs, desc="stage2 eval", unit="sample")
        else:
            print("提示：未安装 tqdm，将不显示进度条（可 pip install tqdm）。")

        for idx_row, row in enumerate(it, start=1):
            s2 = row.get("stage2") or {}
            p2 = str(s2.get("pred") or "")
            s2_gt = s2.get("gt")
            need_strict = (not relaxed_only) and (not _has_stage2_prf_metrics(s2.get("metrics")))
            need_relaxed = force_relaxed or (not _has_stage2_prf_metrics(s2.get("metrics_relaxed")))

            if not need_strict and not need_relaxed:
                if jsonl_path is not None and (idx_row % 25 == 0 or idx_row == len(outputs)):
                    try:
                        save_outputs_jsonl(jsonl_path, outputs)
                    except Exception as e:
                        print(f"警告：stage2 增量写入 JSONL 失败：{e}")
                continue

            if bailian_key and s2_gt is not None:
                pred_s2 = str(p2 or "")
                label_s2 = str(s2_gt or "")
                if need_strict:
                    metrics, point_eval, note, fallback_metrics = _evaluate_stage2_points_once(
                        pred_s2,
                        label_s2,
                        args,
                        bailian_key,
                        STAGE2_POINT_SYSTEM,
                        fallback_note="bailian_call_failed_used_manual",
                    )
                    s2["metrics"] = metrics
                    s2["point_eval"] = point_eval
                    if note:
                        s2["metrics_note"] = note
                    if fallback_metrics is not None:
                        s2["metrics_fallback"] = fallback_metrics

                if need_relaxed:
                    # 方案一：relaxed 沿用 strict 的 points，只放宽匹配规则（不再二次抽取 points）
                    strict_ev = s2.get("point_eval")
                    pts = _try_get_points_from_point_eval(strict_ev)
                    if pts is None:
                        # relaxed_only 场景也需要 points：若 strict points 不存在，先跑一次 strict 抽取
                        _m_tmp, strict_ev_tmp, _note_tmp, _fb_tmp = _evaluate_stage2_points_once(
                            pred_s2,
                            label_s2,
                            args,
                            bailian_key,
                            STAGE2_POINT_SYSTEM,
                            fallback_note="bailian_call_failed_used_manual",
                        )
                        s2.setdefault("point_eval", strict_ev_tmp)
                        pts = _try_get_points_from_point_eval(strict_ev_tmp)

                    if pts is None:
                        # 兜底：沿用旧逻辑（可能导致 points 粒度不一致，但不中断）
                        metrics_relaxed, point_eval_relaxed, note_relaxed, fallback_metrics_relaxed = _evaluate_stage2_points_once(
                            pred_s2,
                            label_s2,
                            args,
                            bailian_key,
                            STAGE2_POINT_SYSTEM_RELAXED_DEGREE,
                            fallback_note="bailian_call_failed_used_manual",
                        )
                        s2["metrics_relaxed"] = metrics_relaxed
                        s2["point_eval_relaxed"] = point_eval_relaxed
                        if note_relaxed:
                            s2["metrics_relaxed_note"] = note_relaxed
                        if fallback_metrics_relaxed is not None:
                            s2["metrics_relaxed_fallback"] = fallback_metrics_relaxed
                    else:
                        predict_points, label_points = pts
                        raw_corr: Optional[str] = None
                        corr_err: Optional[str] = None
                        if bailian_key:
                            try:
                                raw_corr = evaluate_correspondences_bailian_from_points(
                                    predict_points,
                                    label_points,
                                    bailian_key,
                                    args.bailian_base_url,
                                    args.bailian_model,
                                    system_content=STAGE2_POINT_SYSTEM_RELAXED_FROM_POINTS,
                                )
                            except ValueError as e:
                                corr_err = f"bailian_config_error: {e}"
                            except Exception as e:
                                corr_err = f"bailian_http_error: {type(e).__name__}: {e}"

                        if raw_corr:
                            try:
                                corr = _normalize_correspondences_only(json.loads(raw_corr))
                                ev_relaxed = {
                                    "predict_points": predict_points,
                                    "label_points": label_points,
                                    "correspondences": corr,
                                }
                                s2["point_eval_relaxed"] = ev_relaxed
                                s2["metrics_relaxed"] = calculate_metrics([ev_relaxed])
                                if corr_err:
                                    s2["metrics_relaxed_note"] = corr_err
                            except Exception:
                                ev_fb = {
                                    "predict_points": predict_points,
                                    "label_points": label_points,
                                    "correspondences": [],
                                }
                                s2["point_eval_relaxed"] = ev_fb
                                s2["metrics_relaxed"] = calculate_metrics([ev_fb])
                                s2["metrics_relaxed_note"] = "invalid_json_used_manual_match_on_strict_points"
                        else:
                            ev_fb = {
                                "predict_points": predict_points,
                                "label_points": label_points,
                                "correspondences": [],
                            }
                            s2["point_eval_relaxed"] = ev_fb
                            s2["metrics_relaxed"] = calculate_metrics([ev_fb])
                            s2["metrics_relaxed_note"] = corr_err or "no_bailian_output_used_manual_match_on_strict_points"
            elif s2_gt is None:
                if need_strict:
                    s2["metrics"] = {"skipped": True, "reason": "no_aligned_stage2_label_for_index"}
                if need_relaxed:
                    s2["metrics_relaxed"] = {"skipped": True, "reason": "no_aligned_stage2_label_for_index"}
            else:
                ev_fb = manual_parse_and_match(str(p2 or ""), str(s2_gt or ""))
                if need_strict:
                    s2["metrics"] = calculate_metrics([ev_fb])
                    s2["metrics_note"] = "no_api_key_used_manual"
                    s2["point_eval"] = ev_fb
                if need_relaxed:
                    s2["metrics_relaxed"] = calculate_metrics([ev_fb])
                    s2["metrics_relaxed_note"] = "no_api_key_used_manual"
                    s2["point_eval_relaxed"] = ev_fb

            # stage2 每条结束都增量落盘，确保断点可续跑
            if jsonl_path is not None:
                try:
                    save_outputs_jsonl(jsonl_path, outputs)
                except Exception as e:
                    print(f"警告：stage2 增量写入 JSONL 失败：{e}")

            # 无 tqdm 时，每 25 条打印一次进度
            if tqdm is None and (idx_row % 25 == 0 or idx_row == len(outputs)):
                print(f"stage2 eval progress: {idx_row}/{len(outputs)}")
    else:
        # stage2 测评关闭时：保留已有 stage2 相关字段（便于与其他阶段结果合并）
        pass


def main():
    args = build_parser().parse_args()
    eval_from_jsonl = Path(str(args.eval_from_jsonl)).expanduser() if (args.eval_from_jsonl or "").strip() else None

    if eval_from_jsonl is not None:
        print("=" * 80)
        print("vLLM 多 LoRA 两阶段测试（从已有回答文件开始评测）")
        print("=" * 80)
        print(f"回答文件：{eval_from_jsonl}")
        print(f"eval（默认开）：{args.eval}；--no-eval：{not args.eval}")
        print("=" * 80)

        bailian_key = ""
        if args.eval:
            bailian_key = _resolve_bailian_api_key(
                (args.bailian_api_key or "").strip() or None,
                (args.bailian_api_key_file or "").strip() or None,
            )
            if bailian_key:
                try:
                    bailian_key = _require_ascii_http_token("百炼 API Key", bailian_key)
                except ValueError as e:
                    print(f"错误：{e}")
                    bailian_key = ""
            if not bailian_key:
                print("警告：未配置百炼 API Key，二阶段要点评测将跳过（请设置 BAILIAN_API_KEY 或 --bailian-api-key）。")

        outputs = _load_jsonl_rows(eval_from_jsonl)
        args.output_jsonl = eval_from_jsonl
        args.output_xlsx = eval_from_jsonl.with_suffix(".xlsx")

        if args.eval:
            _evaluate_outputs_inplace(outputs, args, bailian_key, jsonl_path=eval_from_jsonl)

        save_outputs_jsonl(args.output_jsonl, outputs)
        save_outputs_xlsx(args.output_xlsx, outputs)
        if args.eval:
            save_eval_artifacts(args.output_jsonl, outputs)

        print("\n" + "=" * 80)
        print(f"处理完成！共 {len(outputs)} 条记录")
        print(f"JSONL 输出：{args.output_jsonl}")
        print(f"Excel 输出：{args.output_xlsx}")
        if args.eval:
            print(f"评估样本指标：{args.output_jsonl.with_name(args.output_jsonl.stem + '_eval_samples.jsonl')}")
            print(f"评估汇总指标：{args.output_jsonl.with_name(args.output_jsonl.stem + '_eval_summary.json')}")
        print("=" * 80)
        return

    stage1_rows = _load_json_list(args.stage1_data)
    stage2_rows = _load_json_list(args.stage2_data)
    stage1_items = _slice_aligned_stage1_rows(stage1_rows, args.start, args.limit)
    if not stage1_items:
        raise ValueError("没有可处理的数据（start/limit 超出范围或数据为空）。")
    skipped_stage1_indices = [i for i, row in enumerate(stage1_rows) if _is_missing_stage1_output(row)]
    if stage1_items and stage1_items[-1][0] >= len(stage2_rows):
        raise ValueError(
            f"过滤空 output 后请求的最大对齐索引（{stage1_items[-1][0]}）超过 stage2 样本数（{len(stage2_rows)}），"
            "请检查两份数据是否一一对应。"
        )
    stage2_instruction = _pick_stage2_instruction(stage2_rows)

    print("=" * 80)
    print("vLLM 多 LoRA 两阶段测试（本地 vLLM）")
    print("=" * 80)
    print(f"基座模型：{args.base_model}")
    print(f"第一阶段 LoRA：{args.stage1_adapter}")
    print(f"第二阶段 LoRA：{args.stage2_adapter}")
    print(f"第一阶段数据：{args.stage1_data}")
    print(f"第二阶段数据：{args.stage2_data}")
    if skipped_stage1_indices:
        print(f"第一阶段数据中跳过空 output 样本索引：{skipped_stage1_indices}")
    print(f"处理范围：start={args.start}, limit={args.limit}（按过滤后对齐索引；实际 {len(stage1_items)} 条）")
    print(f"no-think（默认开）：{not args.think}；--think：{args.think}；eval（默认开）：{args.eval}；--no-eval：{not args.eval}")
    if (args.chat_template_file or "").strip():
        print(f"chat_template_file：{Path(str(args.chat_template_file)).expanduser()}")
    print("=" * 80)

    no_think = not args.think
    bailian_key = ""
    if args.eval:
        bailian_key = _resolve_bailian_api_key(
            (args.bailian_api_key or "").strip() or None,
            (args.bailian_api_key_file or "").strip() or None,
        )
        if bailian_key:
            try:
                bailian_key = _require_ascii_http_token("百炼 API Key", bailian_key)
            except ValueError as e:
                print(f"错误：{e}")
                bailian_key = ""
        if not bailian_key:
            print("警告：未配置百炼 API Key，二阶段要点评测将跳过（请设置 BAILIAN_API_KEY 或 --bailian-api-key）。")

    runner = TwoStageVLLMRunner(
        base_model=args.base_model,
        stage1_adapter=args.stage1_adapter,
        stage2_adapter=args.stage2_adapter,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        chat_template_file=str(args.chat_template_file or ""),
    )

    s1_messages_batch: List[List[Dict[str, str]]] = []
    meta: List[Tuple[int, int, Dict[str, Any], str, Any, str]] = []
    for idx, s1_source_idx, s1 in stage1_items:
        s1_instruction = str(s1.get("instruction") or "")
        s1_input = str(s1.get("input") or "")
        s1_gt = s1.get("output")
        if not s1_instruction or not s1_input:
            raise ValueError(f"过滤后第 {idx} 条（stage1 原始索引 {s1_source_idx}）缺少必要字段 instruction/input。")
        s2_inst_i = _stage2_instruction_for_index(stage2_rows, idx, stage2_instruction)
        s2_gt = stage2_rows[idx].get("output") if 0 <= idx < len(stage2_rows) else None
        # 贴近微调输入格式：instruction + input 合并为同一条 user 消息。
        # 为对齐 LLaMA-Factory 导出的 prompt，不在 stage1 额外追加 NO_THINK_SUFFIX，
        # no-think 主要依赖 apply_chat_template(enable_thinking=False) 从调用层控制。
        s1_ins = s1_instruction
        s1_messages_batch.append(_build_finetune_style_messages(s1_ins, s1_input))
        meta.append((idx, s1_source_idx, s1, s2_inst_i, s2_gt, s1_instruction))

    s1_params = SamplingParams(
        temperature=args.temperature1,
        top_p=args.top_p,
        max_tokens=args.max_tokens1,
        seed=args.seed,
    )
    n = len(s1_messages_batch)
    bs = int(args.vllm_batch_size or 1)
    if bs <= 0:
        bs = 1
    print(
        f"\n阶段 1：共 {n} 条，同一基座 + LoRA=stage1（按 vllm-batch-size={bs} 分批，避免 OOM）"
    )
    stage1_preds: List[str] = []
    for start in range(0, n, bs):
        chunk = s1_messages_batch[start : start + bs]
        lora_s1 = [runner.lora_stage1] * len(chunk)
        chunk_texts = runner.generate_batch(chunk, s1_params, lora_s1, enable_thinking=not no_think)
        stage1_preds.extend(chunk_texts)
        for off, txt in enumerate(chunk_texts):
            idx_i = meta[start + off][0]
            s1_source_idx = meta[start + off][1]
            txt_show = _strip_think_blocks(txt) if no_think else txt
            print(f"\n[{idx_i}] 阶段 1：LoRA=stage1（stage1 原始索引={s1_source_idx}）")
            print(txt_show)

    s2_messages_batch: List[List[Dict[str, str]]] = []
    for (_, _, _, s2_inst_i, _, _), stage1_pred in zip(meta, stage1_preds):
        p1 = stage1_pred
        if no_think:
            p1 = _strip_think_blocks(p1)
        # 贴近微调输入格式：instruction + stage1 输出合并为同一条 user 消息
        s2_ins = s2_inst_i + (NO_THINK_SUFFIX if no_think else "")
        s2_messages_batch.append(_build_finetune_style_messages(s2_ins, p1))

    s2_params = SamplingParams(
        temperature=args.temperature2,
        top_p=args.top_p,
        max_tokens=args.max_tokens2,
        seed=args.seed,
    )
    print(
        f"\n阶段 2：共 {n} 条，同一基座 + LoRA=stage2（按 vllm-batch-size={bs} 分批，避免 OOM）"
    )
    stage2_preds: List[str] = []
    for start in range(0, n, bs):
        chunk = s2_messages_batch[start : start + bs]
        lora_s2 = [runner.lora_stage2] * len(chunk)
        chunk_texts = runner.generate_batch(chunk, s2_params, lora_s2, enable_thinking=not no_think)
        stage2_preds.extend(chunk_texts)
        for off, txt in enumerate(chunk_texts):
            idx_i = meta[start + off][0]
            txt_show = _strip_think_blocks(txt) if no_think else txt
            print(f"\n[{idx_i}] 阶段 2：LoRA=stage2")
            print(txt_show)

    outputs: List[Dict[str, Any]] = []
    for (idx, s1_source_idx, s1, s2_inst_i, s2_gt, s1_instruction), stage1_pred, stage2_pred in zip(
        meta, stage1_preds, stage2_preds
    ):
        s1_gt = s1.get("output")
        p1 = stage1_pred
        p2 = stage2_pred
        if no_think:
            p1 = _strip_think_blocks(p1)
            p2 = _strip_think_blocks(p2)
        s1_input = str(s1.get("input") or "")

        row: Dict[str, Any] = {
            "id": idx,
            "stage1_source_index": s1_source_idx,
            "stage1": {
                "instruction": s1_instruction,
                "input": s1_input,
                "gt": s1_gt,
                "pred": p1,
            },
            "stage2": {
                "instruction": s2_inst_i,
                "input": p1,
                "gt": s2_gt,
                "pred": p2,
            },
        }
        outputs.append(row)
        # 生成阶段逐条落盘，避免后续评测失败导致已生成内容丢失。
        save_outputs_jsonl(args.output_jsonl, outputs)

    # 先保存纯生成结果；即使后续评测失败，也保留生成产物。
    save_outputs_jsonl(args.output_jsonl, outputs)
    save_outputs_xlsx(args.output_xlsx, outputs)

    if args.eval:
        try:
            del runner
        except Exception:
            pass
        _evaluate_outputs_inplace(outputs, args, bailian_key, jsonl_path=Path(args.output_jsonl))
        # 回写带评测结果的产物
        save_outputs_jsonl(args.output_jsonl, outputs)
        save_outputs_xlsx(args.output_xlsx, outputs)

    if args.eval:
        save_eval_artifacts(args.output_jsonl, outputs)
    print("\n" + "=" * 80)
    print(f"处理完成！共 {len(outputs)} 条记录")
    print(f"JSONL 输出：{args.output_jsonl}")
    print(f"Excel 输出：{args.output_xlsx}")
    if args.eval:
        print(f"评估样本指标：{args.output_jsonl.with_name(args.output_jsonl.stem + '_eval_samples.jsonl')}")
        print(f"评估汇总指标：{args.output_jsonl.with_name(args.output_jsonl.stem + '_eval_summary.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
