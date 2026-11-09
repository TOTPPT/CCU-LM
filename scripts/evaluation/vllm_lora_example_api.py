"""
两阶段测试脚本：通用大模型 API（OpenAI 兼容）替代本地 vLLM + LoRA。

数据对齐、输出 JSONL 结构、一阶段 BLEU/ROUGE/BERTScore、二阶段要点评测（百炼 / 本地规则）
与 vllm_lora_example.py 保持一致；仅推理改为 HTTP API。

Prompt 由 JSON 控制（见 --prompt-json）：可为每阶段配置 messages 列表，支持占位符：
  {instruction} {input} {stage1_pred} {no_think_suffix} {few_shot_block}

可选 few_shot（见 vllm_lora_example_api_prompts_fewshot.json）：拼入 API 的 user 消息（{few_shot_block}）；instruction 宜在 messages 中只出现一次（例如先写 instruction，空行后再接 few_shot_block 与当前 input），示例块内可仅用 input/output，不写入 stage1.instruction 字段；
每条 JSONL 会带 few_shot_meta（路径、indices、注入字符数）便于核对。

环境变量（推理，被测模型）：
  INFERENCE_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / DASHSCOPE_API_KEY（见 _resolve_inference_api_key）
  INFERENCE_BASE_URL（默认 OpenRouter：https://openrouter.ai/api/v1；百炼：https://dashscope.aliyuncs.com/compatible-mode/v1）
  INFERENCE_MODEL（默认 qwen/qwen3.6-plus）；部分模型在部分地区或账号下仍可能 403，可改 INFERENCE_MODEL 或换 --infer-base-url
  OpenRouter 可选头：OPENROUTER_HTTP_REFERER、OPENROUTER_APP_TITLE（HTTP-Referer、X-OpenRouter-Title）

二阶段要点评测（裁判模型，可与推理不同平台）：
  优先 STAGE2_EVAL_API_KEY（正式测评推荐，与推理 Key 分离）；
  其次 BAILIAN_API_KEY、DASHSCOPE_API_KEY、INFERENCE_API_KEY、OPENAI_API_KEY 或 ~/.bailian_api_key。

Demo 阶段（被测与裁判同在百炼、同一兼容端点）：可不设 --bailian-api-key，当 infer_base_url 与 bailian_base_url 一致时自动复用推理 Key。
正式测评：为裁判单独设 STAGE2_EVAL_API_KEY 或 --bailian-api-key，并设 --bailian-base-url 指向评测平台（可与 --infer-base-url 不同）。

示例（默认 OpenRouter 被测；Key 可用 OPENROUTER_API_KEY 或 OPENAI_API_KEY）：
    export OPENROUTER_API_KEY=sk-or-v1-...
    # 可选：export OPENROUTER_HTTP_REFERER=https://your.site  OPENROUTER_APP_TITLE=MyEval
    python vllm_lora_example_api.py --limit 5
    # 百炼作被测时显式：--infer-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --infer-model qwen3.5-flash
    python vllm_lora_example_api.py --eval-from-jsonl runs/out.jsonl

生成结束后会强制落盘主 JSONL、XLSX，并额外写入 *_generation_complete.jsonl 检查点。
跳过 BERTScore（仍算 BLEU/ROUGE 与二阶段要点）：加 --no-bertscore
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


DEFAULT_PROMPT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "stage1": {
        "messages": [
            {"role": "user", "content": "{instruction}\n\n{input}"},
        ],
    },
    "stage2": {
        "messages": [
            {"role": "user", "content": "{instruction}{no_think_suffix}\n\n{stage1_pred}"},
        ],
    },
}


def _strip_think_blocks(text: str) -> str:
    if not text:
        return text
    pat = r"<" + "think" + r">[\s\S]*?</" + "think" + r">\s*"
    text = re.sub(pat, "", text, flags=re.IGNORECASE)
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


def _sanitize_api_key_line(raw: str) -> str:
    """
    从环境变量/文件首行中尽量抽出纯 ASCII 的 API Key。
    常见误用：整行「密钥：sk-xxx」、全角冒号、中文前后缀、复制了说明文字等。
    """
    if not (raw or "").strip():
        return ""
    s = raw.strip()
    if s.startswith("\ufeff"):
        s = s[1:].strip()
    s = s.replace("\uff1a", ":").replace("\u3000", " ")
    # DashScope / OpenAI 风格 sk-...
    m = re.search(r"(sk-[a-zA-Z0-9]+)", s)
    if m and len(m.group(1)) >= 10:
        return m.group(1)
    # 最后一个英文冒号后的 token（如 Some-Label: sk-xxx）
    if ":" in s:
        tail = s.rsplit(":", 1)[-1].strip().strip("'\"）) ")
        tail = re.sub(r"^[^A-Za-z0-9]+", "", tail)
        if tail and all(ord(c) < 128 for c in tail) and re.fullmatch(r"[A-Za-z0-9_\-\.]+", tail) and len(tail) >= 8:
            return tail
    if "=" in s:
        tail = s.split("=", 1)[-1].strip().strip("'\"")
        tail = "".join(c for c in tail if ord(c) < 128).strip()
        if tail and re.fullmatch(r"[A-Za-z0-9_\-\.]+", tail) and len(tail) >= 8:
            return tail
    runs = re.findall(r"[A-Za-z0-9_\-\.]{15,}", s)
    if runs:
        return max(runs, key=len)
    return "".join(c for c in s if ord(c) < 128).strip()


def _resolve_bailian_api_key(explicit: Optional[str], key_file: Optional[str]) -> str:
    if explicit:
        return _sanitize_api_key_line(explicit.strip())
    for env_name in (
        "STAGE2_EVAL_API_KEY",
        "BAILIAN_API_KEY",
        "DASHSCOPE_API_KEY",
        "INFERENCE_API_KEY",
        "OPENAI_API_KEY",
    ):
        env = (os.environ.get(env_name) or "").strip()
        if env:
            return _sanitize_api_key_line(env)
    paths = []
    if key_file:
        paths.append(Path(key_file).expanduser())
    paths.append(Path.home() / ".bailian_api_key")
    for p in paths:
        if p.is_file():
            v = p.read_text(encoding="utf-8").strip().splitlines()
            if v:
                return _sanitize_api_key_line(v[0].strip())
    return ""


def _resolve_inference_api_key(explicit: Optional[str], key_file: Optional[str]) -> str:
    if explicit:
        return _sanitize_api_key_line(explicit.strip())
    for k in ("INFERENCE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return _sanitize_api_key_line(v)
    paths = []
    if key_file:
        paths.append(Path(key_file).expanduser())
    for p in paths:
        if p.is_file():
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                return _sanitize_api_key_line(lines[0].strip())
    return ""


def _openai_client_kwargs_for_infer(api_key: str, base_url: str) -> Dict[str, Any]:
    """构造 OpenAI 兼容客户端参数；OpenRouter 时可选附加推荐头（见 OpenRouter 文档）。"""
    kw: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if "openrouter.ai" in (base_url or "").lower():
        hdrs: Dict[str, str] = {}
        ref = (os.environ.get("OPENROUTER_HTTP_REFERER") or "").strip()
        title = (os.environ.get("OPENROUTER_APP_TITLE") or "").strip()
        if ref:
            hdrs["HTTP-Referer"] = ref
        if title:
            hdrs["X-OpenRouter-Title"] = title
        if hdrs:
            kw["default_headers"] = hdrs
    return kw


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
    sample_metrics = calculate_sample_metrics(evaluation_results)
    if not sample_metrics["sample_precisions"]:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
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
        except Exception as e:
            bailian_err = f"bailian_http_error: {type(e).__name__}: {e}"

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


def _try_get_points_from_point_eval(ev: Any) -> Optional[Tuple[List[str], List[str]]]:
    if not isinstance(ev, dict):
        return None
    pp = ev.get("predict_points")
    lp = ev.get("label_points")
    if not isinstance(pp, list) or not isinstance(lp, list):
        return None
    return ([str(x) for x in pp], [str(x) for x in lp])


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
    user_content = (
        "请根据以下 points 建立 correspondences（索引从0开始）。\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
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
    *,
    skip_bertscore: bool = False,
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
    if skip_bertscore:
        metrics["bertscore_skipped"] = True
        return metrics
    try:
        import torch
        from transformers import AutoConfig
        from bert_score import score as bert_score_fn

        device_pref = (bertscore_device or "cuda").strip().lower()
        if device_pref == "cuda" and not torch.cuda.is_available():
            device_pref = "cpu"
        model_type = bert_model_path
        num_layers = None
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


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_prompt_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_PROMPT_CONFIG))
    p = (path or "").strip()
    if not p:
        return cfg
    fp = Path(p).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"--prompt-json 不存在：{fp}")
    with open(fp, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    if not isinstance(user_cfg, dict):
        raise ValueError("prompt JSON 根节点必须是对象")
    return _deep_merge_dict(cfg, user_cfg)


def _format_prompt_template(template: str, ctx: Dict[str, str]) -> str:
    try:
        return template.format(**ctx)
    except KeyError as e:
        raise ValueError(f"prompt 模板含未知占位符或缺少字段：{e}；可用键：{sorted(ctx.keys())}") from e


def build_messages_from_stage_config(
    stage_cfg: Dict[str, Any],
    ctx: Dict[str, str],
) -> List[Dict[str, str]]:
    raw_msgs = stage_cfg.get("messages")
    if not isinstance(raw_msgs, list) or not raw_msgs:
        raise ValueError("prompt 配置中每个 stage 需要非空 messages 列表")
    out: List[Dict[str, str]] = []
    for i, m in enumerate(raw_msgs):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] 必须是对象")
        role = str(m.get("role") or "").strip().lower()
        content = _format_prompt_template(str(m.get("content") or ""), ctx)
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"messages[{i}].role 非法：{m.get('role')}")
        if not (content or "").strip():
            continue
        out.append({"role": role, "content": content})
    if not out:
        raise ValueError("格式化后的 messages 为空（请检查模板是否全为空）")
    return out


def chat_completion_text(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = client.chat.completions.create(**kwargs)
    ch = resp.choices[0].message
    return (ch.content or "").strip() if ch else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="两阶段测试：通用 API + JSON 控制 prompt（测评逻辑同 vllm_lora_example）")
    cur_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parents[2] / "outputs" / f"two_stage_api_{timestamp}"

    parser.add_argument(
        "--prompt-json",
        type=str,
        default="",
        help="Prompt 配置文件（可选）。未指定则使用内置默认（与原版 finetune 风格 user 拼接一致）。",
    )
    parser.add_argument(
        "--infer-base-url",
        type=str,
        default=os.environ.get("INFERENCE_BASE_URL", "https://openrouter.ai/api/v1"),
        help="推理 API base_url（OpenAI 兼容；默认 OpenRouter）。百炼示例见文档。环境变量 INFERENCE_BASE_URL。",
    )
    parser.add_argument(
        "--infer-model",
        type=str,
        default=os.environ.get("INFERENCE_MODEL", "qwen/qwen3.6-plus"),
        help="推理模型名（OpenRouter 常为 vendor/model，如 qwen/qwen3.6-plus；百炼填 qwen 系列 id）",
    )
    parser.add_argument(
        "--infer-api-key",
        type=str,
        default="",
        help="被测模型 API Key；默认 INFERENCE_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / DASHSCOPE_API_KEY；与裁判 Key 分离时见 STAGE2_EVAL_API_KEY / --bailian-api-key",
    )
    parser.add_argument("--infer-api-key-file", type=str, default="", help="从文件读取推理 API Key（首行）")
    parser.add_argument(
        "--infer-extra-body-json",
        type=str,
        default="",
        help='可选：合并到 chat.completions 的 extra_body（JSON 字符串），如 {"enable_thinking": false}',
    )
    parser.add_argument("--api-sleep-seconds", type=float, default=0.0, help="每条推理请求之间的休眠秒数（限流时调大）")

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
    parser.add_argument("--start", type=int, default=0, help="从第几条开始（过滤空 output 后的对齐索引）")
    parser.add_argument("--limit", type=int, default=5, help="处理多少条；<=0 表示全部")
    parser.add_argument("--temperature1", type=float, default=0.0, help="第一阶段温度")
    parser.add_argument("--temperature2", type=float, default=0.2, help="第二阶段温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p")
    parser.add_argument(
        "--max-tokens1",
        type=int,
        default=8192,
        help="第一阶段 chat 补全 max_tokens（默认加大以适配长报告+few-shot 场景；可按模型上限再调）",
    )
    parser.add_argument(
        "--max-tokens2",
        type=int,
        default=4096,
        help="第二阶段 chat 补全 max_tokens（默认加大以适配长结论输出）",
    )
    parser.add_argument("--output-jsonl", type=Path, default=run_dir / "two_stage.jsonl", help="输出 JSONL")
    parser.add_argument("--output-xlsx", type=Path, default=run_dir / "two_stage.xlsx", help="输出 Excel")
    parser.add_argument(
        "--think",
        action="store_true",
        help="允许思考输出（默认关闭：对阶段2在模板中注入 no_think_suffix，并剥离思考块）",
    )
    parser.set_defaults(eval=True)
    parser.add_argument(
        "--no-eval",
        action="store_false",
        dest="eval",
        help="关闭测评",
    )
    parser.set_defaults(eval_stage1=True, eval_stage2=True)
    parser.add_argument("--no-eval-stage1", action="store_false", dest="eval_stage1", help="关闭 stage1 测评")
    parser.add_argument("--no-eval-stage2", action="store_false", dest="eval_stage2", help="关闭 stage2 测评")
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="stage1 测评仍计算 BLEU/ROUGE，但不加载模型、不计算 BERTScore（省显存/防 OOM）",
    )
    parser.add_argument(
        "--eval-stage2-relaxed-only",
        action="store_true",
        help="只增量补充 stage2 relaxed 测评",
    )
    parser.add_argument("--force-relaxed-eval", action="store_true", help="强制重算 relaxed 测评")
    parser.add_argument(
        "--bert-model-path",
        type=str,
        default="./models/bert",
        help="BERTScore 本地模型目录",
    )
    parser.add_argument("--bertscore-device", type=str, default="cuda", choices=["cuda", "cpu"], help="BERTScore 设备")
    parser.add_argument("--bertscore-batch-size", type=int, default=32, help="BERTScore batch_size")
    parser.add_argument(
        "--bailian-base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="要点评测用百炼 base_url",
    )
    parser.add_argument("--bailian-model", type=str, default="qwen3.5-27b", help="要点评测模型名")
    parser.add_argument(
        "--bailian-api-key",
        type=str,
        default="",
        help="二阶段要点评测（裁判）API Key；不设时读 STAGE2_EVAL_API_KEY → BAILIAN_API_KEY → DASHSCOPE → … → ~/.bailian_api_key；与 --infer-base-url 同 URL 且未指定本项时 demo 复用推理 Key",
    )
    parser.add_argument(
        "--bailian-api-key-file",
        type=str,
        default="",
        help="二阶段评测（裁判）Key 文件首行；正式测评可与 --infer-api-key-file 分文件",
    )
    parser.add_argument(
        "--eval-from-jsonl",
        type=str,
        default="",
        help="从已有回答 JSONL 直接测评并回写",
    )
    parser.add_argument(
        "--resume-from-jsonl",
        type=str,
        default="",
        help="从已有 JSONL 断点续跑生成：只补齐缺失条目（不会覆盖已存在 pred）；生成结束后可继续测评并回写到该文件",
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


def build_few_shot_block_for_stage(fs_stage: Any, *, stage_name: str) -> str:
    """
    根据 prompt JSON 中 few_shot.<stage_name> 从指定 JSON 列表按 indices 取样本，
    用 example_template（及可选 header/footer/examples_join）拼成一段固定文本，供 messages 里 {few_shot_block} 使用。

    配置项：
      - enabled: 默认 true；为 false 时返回空串
      - data_path: 训练集 JSON 列表路径
      - indices: 样本下标列表（如 3-shot：[0,1,2]）
      - example_template: 支持占位符 instruction, input, output, shot_index（1-based）；instruction 与各示例相同时可省略，仅在 messages 顶层写一次 {instruction}
      - header, footer: 可选，包在若干示例外侧
      - examples_join: 示例之间的连接串，默认 "\\n"
    """
    if not isinstance(fs_stage, dict):
        return ""
    if fs_stage.get("enabled") is False:
        return ""
    data_path = str(fs_stage.get("data_path") or "").strip()
    if not data_path:
        return ""
    indices = fs_stage.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"few_shot.{stage_name}.indices 必须为非空列表（整数下标）")
    tmpl = str(fs_stage.get("example_template") or "").strip()
    if not tmpl:
        raise ValueError(f"few_shot.{stage_name}.example_template 不能为空（用于格式化每条示例）")
    rows = _load_json_list(str(Path(data_path).expanduser()))
    n = len(rows)
    chunks: List[str] = []
    for j, raw_idx in enumerate(indices):
        ii = int(raw_idx)
        if ii < 0 or ii >= n:
            raise ValueError(f"few_shot.{stage_name} 索引 {ii} 越界（数据共 {n} 条）：{data_path}")
        row = rows[ii]
        ctx_shot = {
            "instruction": str(row.get("instruction") or ""),
            "input": str(row.get("input") or ""),
            "output": str(row.get("output") or ""),
            "shot_index": str(j + 1),
        }
        try:
            chunks.append(tmpl.format(**ctx_shot))
        except KeyError as e:
            raise ValueError(f"few_shot.{stage_name}.example_template 占位符错误：{e}") from e
    joiner = fs_stage.get("examples_join")
    if joiner is None:
        joiner = "\n"
    middle = str(joiner).join(chunks)
    header = str(fs_stage.get("header") or "")
    footer = str(fs_stage.get("footer") or "")
    return f"{header}{middle}{footer}"


def build_few_shot_output_meta(
    fs_root: Any,
    *,
    block_len_stage1: int,
    block_len_stage2: int,
    prompt_json_path: str,
) -> Optional[Dict[str, Any]]:
    """
    写入 JSONL 的轻量元数据：说明 few_shot 去向并记录 data_path/indices/注入长度（不重复整段示例正文，避免文件膨胀）。
    """
    if not isinstance(fs_root, dict) or not fs_root:
        return None
    if not block_len_stage1 and not block_len_stage2:
        return None
    meta: Dict[str, Any] = {
        "prompt_json": (prompt_json_path or "").strip() or None,
        "note": "few_shot 通过 prompt JSON 的 {few_shot_block} 拼进发给模型的 user 正文；本行 stage1/2 的 instruction、input 仍来自评测集字段，不是完整请求体。",
        "stage1_injected_chars": int(block_len_stage1),
        "stage2_injected_chars": int(block_len_stage2),
    }
    for sk in ("stage1", "stage2"):
        sec = fs_root.get(sk)
        if not isinstance(sec, dict) or sec.get("enabled") is False:
            continue
        meta[f"{sk}_config"] = {"data_path": sec.get("data_path"), "indices": sec.get("indices")}
    return meta


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


def save_outputs_jsonl(path: Path, rows: List[Dict[str, Any]], *, fsync: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        if fsync:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def persist_generation_outputs(jsonl_path: Path, xlsx_path: Path, rows: List[Dict[str, Any]]) -> Path:
    """
    全部生成结束后落盘：主 JSONL（fsync）、Excel、以及同目录 *_generation_complete.jsonl 检查点（防后续评测 OOM/Kill 丢失生成结果）。
    """
    save_outputs_jsonl(jsonl_path, rows, fsync=True)
    try:
        save_outputs_xlsx(xlsx_path, rows)
    except Exception as e:
        print(f"警告：写入 XLSX 失败（JSONL 已保存）：{e}")
    ckpt = jsonl_path.with_name(jsonl_path.stem + "_generation_complete.jsonl")
    save_outputs_jsonl(ckpt, rows, fsync=True)
    print(f"生成结果已落盘：{jsonl_path}\n生成检查点：{ckpt}")
    return ckpt


def save_outputs_xlsx(path: Path, rows: List[Dict[str, Any]]):
    if pd is None:
        raise RuntimeError("未安装 pandas，无法保存 .xlsx。")
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
                "few_shot_meta": json.dumps(r.get("few_shot_meta"), ensure_ascii=False)
                if isinstance(r.get("few_shot_meta"), dict)
                else "",
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
    stage2_vals: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
    stage2_relaxed_vals: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
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


def _merge_generation_args(args: argparse.Namespace, prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    gen = prompt_cfg.get("generation") if isinstance(prompt_cfg.get("generation"), dict) else {}
    return {
        "temperature1": float(gen.get("temperature1", args.temperature1)),
        "temperature2": float(gen.get("temperature2", args.temperature2)),
        "top_p": float(gen.get("top_p", args.top_p)),
        "max_tokens1": int(gen.get("max_tokens1", args.max_tokens1)),
        "max_tokens2": int(gen.get("max_tokens2", args.max_tokens2)),
    }


def _evaluate_outputs_inplace(outputs: List[Dict[str, Any]], args, bailian_key: str):
    jsonl_path: Optional[Path] = None
    raw_jp = getattr(args, "output_jsonl", None)
    if raw_jp is not None:
        jsonl_path = Path(raw_jp) if not isinstance(raw_jp, Path) else raw_jp

    if getattr(args, "no_bertscore", False):
        print("\n开始评测：计算 BLEU/ROUGE（已跳过 BERTScore）与二阶段要点指标...")
    else:
        print("\n开始评测：计算 BLEU/ROUGE/BERTScore 与二阶段要点指标...")
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if getattr(args, "eval_stage1", True):
        s1_preds: List[str] = []
        s1_refs: List[str] = []
        s1_row_indices: List[int] = []

        for i, row in enumerate(outputs):
            s1 = row.get("stage1") or {}
            p1 = str(s1.get("pred") or "")
            s1_gt = s1.get("gt")
            s1_ref = str(s1_gt) if s1_gt is not None else ""

            m = compute_stage1_text_metrics(
                p1,
                s1_ref,
                args.bert_model_path,
                bertscore_device="cpu",
                skip_bertscore=bool(getattr(args, "no_bertscore", False)),
            )
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

        if s1_preds and not bool(getattr(args, "no_bertscore", False)):
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
                    if meta and j == 0:
                        s1m["bertscore_meta"] = meta
            except Exception as e:
                for row_i in s1_row_indices:
                    s1m = (outputs[row_i].get("stage1") or {}).get("metrics") or {}
                    s1m["bertscore_error"] = repr(e)

        if jsonl_path is not None:
            save_outputs_jsonl(jsonl_path, outputs, fsync=True)
            try:
                xlsx_p = getattr(args, "output_xlsx", None)
                if xlsx_p is not None:
                    save_outputs_xlsx(Path(xlsx_p) if not isinstance(xlsx_p, Path) else xlsx_p, outputs)
            except Exception as e:
                print(f"警告：stage1 完成后写 XLSX 失败（JSONL 已保存）：{e}")
            print(f"stage1 测评已落盘：{jsonl_path}")

    if getattr(args, "eval_stage2", True):
        # stage1 后内存峰值（尤其 XLSX）+ stage2 首条两次裁判请求，易触发 OOM（进程表现为 Killed）；先尽量回收
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        relaxed_only = bool(getattr(args, "eval_stage2_relaxed_only", False))
        force_relaxed = bool(getattr(args, "force_relaxed_eval", False))
        it = outputs
        if tqdm is not None:
            it = tqdm(outputs, desc="stage2 eval", unit="sample")
        else:
            print("提示：未安装 tqdm，将不显示进度条。")

        for idx_row, row in enumerate(it, start=1):
            s2 = row.get("stage2") or {}
            p2 = str(s2.get("pred") or "")
            s2_gt = s2.get("gt")
            # 断点续算：已有的 metrics/metrics_relaxed 不应被清空，否则会导致每次都从第 1 条重算
            need_strict = (not relaxed_only) and (not _has_stage2_prf_metrics(s2.get("metrics")))
            need_relaxed = force_relaxed or (not _has_stage2_prf_metrics(s2.get("metrics_relaxed")))

            if not need_strict and not need_relaxed:
                # 本条两种测评都已存在，直接跳过（但仍会走进度条计数）
                if jsonl_path is not None and (idx_row % 25 == 0 or idx_row == len(outputs)):
                    save_outputs_jsonl(jsonl_path, outputs, fsync=True)
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
                        # 理论上不应发生；兜底沿用旧逻辑（会导致 points 粒度不一致，但至少不中断）
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
                                # 裁判输出不可解析 -> relaxed 用严格 points + 手工匹配兜底
                                ev_fb = {
                                    "predict_points": predict_points,
                                    "label_points": label_points,
                                    "correspondences": [],
                                }
                                s2["point_eval_relaxed"] = ev_fb
                                s2["metrics_relaxed"] = calculate_metrics([ev_fb])
                                s2["metrics_relaxed_note"] = "invalid_json_used_manual_match_on_strict_points"
                        else:
                            # 无裁判输出 -> relaxed 用严格 points + 手工匹配兜底
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

            # 仅增量写 JSONL：每条 stage2 后再写 XLSX 会重复构建整张表，超长 instruction/pred 下极易触发内存峰值与 OOM（进程表现为 Killed）
            if jsonl_path is not None:
                save_outputs_jsonl(jsonl_path, outputs, fsync=True)

            if tqdm is None and (idx_row % 25 == 0 or idx_row == len(outputs)):
                print(f"stage2 eval progress: {idx_row}/{len(outputs)}")


def main():
    args = build_parser().parse_args()
    eval_from_jsonl = Path(str(args.eval_from_jsonl)).expanduser() if (args.eval_from_jsonl or "").strip() else None
    resume_from_jsonl = Path(str(args.resume_from_jsonl)).expanduser() if (args.resume_from_jsonl or "").strip() else None
    prompt_cfg = load_prompt_config(str(args.prompt_json or ""))
    gen_cfg = _merge_generation_args(args, prompt_cfg)

    if eval_from_jsonl is not None:
        print("=" * 80)
        print("两阶段测评（从已有回答 JSONL）")
        print("=" * 80)
        print(f"回答文件：{eval_from_jsonl}")

        bailian_key = ""
        infer_key_ej = ""
        try:
            infer_key_ej = _resolve_inference_api_key(
                (args.infer_api_key or "").strip() or None,
                (args.infer_api_key_file or "").strip() or None,
            )
            if infer_key_ej:
                infer_key_ej = _require_ascii_http_token("推理 API Key", infer_key_ej)
        except ValueError as e:
            print(f"错误：{e}")
            infer_key_ej = ""
        explicit_bailian_ej = (args.bailian_api_key or "").strip()
        if args.eval:
            bailian_key = _resolve_bailian_api_key(
                explicit_bailian_ej or None,
                (args.bailian_api_key_file or "").strip() or None,
            )
            if bailian_key:
                try:
                    bailian_key = _require_ascii_http_token("百炼 API Key", bailian_key)
                except ValueError as e:
                    print(f"错误：{e}")
                    bailian_key = ""
            infer_u = str(args.infer_base_url or "").strip().rstrip("/")
            bail_u = str(args.bailian_base_url or "").strip().rstrip("/")
            if infer_key_ej and not explicit_bailian_ej and infer_u == bail_u:
                bailian_key = infer_key_ej
                print(
                    "提示（demo）：二阶段要点评测与推理为同一兼容端点且未指定 --bailian-api-key，已复用推理 Key。"
                    "正式测评请为裁判单独设 STAGE2_EVAL_API_KEY 或 --bailian-api-key，并设置与推理不同的 --bailian-base-url（若跨平台）。"
                )

        outputs = _load_jsonl_rows(eval_from_jsonl)
        args.output_jsonl = eval_from_jsonl
        args.output_xlsx = eval_from_jsonl.with_suffix(".xlsx")

        if args.eval:
            _evaluate_outputs_inplace(outputs, args, bailian_key)

        save_outputs_jsonl(args.output_jsonl, outputs, fsync=True)
        save_outputs_xlsx(args.output_xlsx, outputs)
        if args.eval:
            save_eval_artifacts(args.output_jsonl, outputs)

        print(f"\n处理完成！共 {len(outputs)} 条；JSONL：{args.output_jsonl}")
        return

    if OpenAI is None:
        raise RuntimeError("需要安装 openai 库：pip install openai")

    infer_key = _resolve_inference_api_key(
        (args.infer_api_key or "").strip() or None,
        (args.infer_api_key_file or "").strip() or None,
    )
    if not infer_key:
        raise RuntimeError(
            "未配置推理 API Key：请设置 INFERENCE_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY 或 --infer-api-key"
        )
    infer_key = _require_ascii_http_token("推理 API Key", infer_key)
    base_url = _require_ascii_http_token("推理 base_url", str(args.infer_base_url or "").strip())
    model = _require_ascii_http_token("推理 model", str(args.infer_model or "").strip())

    extra_body: Optional[Dict[str, Any]] = None
    raw_extra = (args.infer_extra_body_json or "").strip()
    if raw_extra:
        extra_body = json.loads(raw_extra)
        if not isinstance(extra_body, dict):
            raise ValueError("--infer-extra-body-json 必须是 JSON 对象")

    client = OpenAI(**_openai_client_kwargs_for_infer(infer_key, base_url))

    stage1_rows = _load_json_list(args.stage1_data)
    stage2_rows = _load_json_list(args.stage2_data)
    stage1_items = _slice_aligned_stage1_rows(stage1_rows, args.start, args.limit)
    if not stage1_items:
        raise ValueError("没有可处理的数据。")
    if stage1_items[-1][0] >= len(stage2_rows):
        raise ValueError(
            f"对齐索引 {stage1_items[-1][0]} 超过 stage2 样本数 {len(stage2_rows)}。"
        )
    stage2_instruction = _pick_stage2_instruction(stage2_rows)
    no_think = not args.think
    custom_nt = prompt_cfg.get("no_think_suffix_template")
    nt_template = str(custom_nt).strip() if isinstance(custom_nt, str) and str(custom_nt).strip() else NO_THINK_SUFFIX
    nt_suffix = nt_template if no_think else ""

    bailian_key = ""
    explicit_bailian_cli = (args.bailian_api_key or "").strip()
    if args.eval:
        bailian_key = _resolve_bailian_api_key(
            explicit_bailian_cli or None,
            (args.bailian_api_key_file or "").strip() or None,
        )
        if bailian_key:
            try:
                bailian_key = _require_ascii_http_token("百炼 API Key", bailian_key)
            except ValueError as e:
                print(f"错误：{e}")
                bailian_key = ""
        infer_u = str(args.infer_base_url or "").strip().rstrip("/")
        bail_u = str(args.bailian_base_url or "").strip().rstrip("/")
        if infer_u == bail_u and infer_key and not explicit_bailian_cli:
            bailian_key = infer_key
            print(
                "提示（demo）：二阶段要点评测与推理为同一兼容端点且未指定 --bailian-api-key，已复用推理 Key。"
                "正式测评请为裁判单独设 STAGE2_EVAL_API_KEY 或 --bailian-api-key，并设置与推理不同的 --bailian-base-url（若跨平台）。"
            )

    print("=" * 80)
    print("两阶段测试（通用 API）")
    print("=" * 80)
    print(f"infer base_url={base_url} model={model}")
    print(f"stage1_data={args.stage1_data} stage2_data={args.stage2_data}")
    print(f"prompt-json: {(args.prompt_json or '(内置默认)')}")
    print(f"处理 {len(stage1_items)} 条；no_think={no_think}；eval={args.eval}")
    print("=" * 80)

    s1_cfg = prompt_cfg.get("stage1") or {}
    s2_cfg = prompt_cfg.get("stage2") or {}
    if not isinstance(s1_cfg, dict) or not isinstance(s2_cfg, dict):
        raise ValueError("prompt 配置中 stage1/stage2 必须为对象")

    fs_root = prompt_cfg.get("few_shot") or {}
    few_shot_block_stage1 = ""
    few_shot_block_stage2 = ""
    if isinstance(fs_root, dict) and fs_root:
        few_shot_block_stage1 = build_few_shot_block_for_stage(fs_root.get("stage1"), stage_name="stage1")
        few_shot_block_stage2 = build_few_shot_block_for_stage(fs_root.get("stage2"), stage_name="stage2")
        if few_shot_block_stage1 or few_shot_block_stage2:
            print(
                f"few_shot：stage1 块长度={len(few_shot_block_stage1)} 字符，"
                f"stage2 块长度={len(few_shot_block_stage2)} 字符"
            )

    few_shot_row_meta = build_few_shot_output_meta(
        fs_root,
        block_len_stage1=len(few_shot_block_stage1),
        block_len_stage2=len(few_shot_block_stage2),
        prompt_json_path=str(args.prompt_json or ""),
    )

    outputs: List[Dict[str, Any]] = []
    existing_ids: set[int] = set()
    if resume_from_jsonl is not None:
        if resume_from_jsonl.is_file():
            outputs = _load_jsonl_rows(resume_from_jsonl)
            for r in outputs:
                try:
                    rid = int(r.get("id")) if isinstance(r, dict) and r.get("id") is not None else None
                except Exception:
                    rid = None
                if isinstance(rid, int):
                    s1 = (r.get("stage1") or {}) if isinstance(r, dict) else {}
                    s2 = (r.get("stage2") or {}) if isinstance(r, dict) else {}
                    # 只有 stage1+stage2 都有 pred 才视为已完成
                    if str((s1.get("pred") or "")).strip() and str((s2.get("pred") or "")).strip():
                        existing_ids.add(rid)
        # 断点恢复时：回写到同一个 JSONL，并把 xlsx 跟随同名
        args.output_jsonl = resume_from_jsonl
        args.output_xlsx = resume_from_jsonl.with_suffix(".xlsx")

    sleep_s = float(args.api_sleep_seconds or 0.0)

    for idx, s1_source_idx, s1 in stage1_items:
        if resume_from_jsonl is not None and idx in existing_ids:
            continue
        s1_instruction = str(s1.get("instruction") or "")
        s1_input = str(s1.get("input") or "")
        if not s1_instruction or not s1_input:
            raise ValueError(f"索引 {idx}（stage1 原始 {s1_source_idx}）缺少 instruction/input。")
        s2_inst_i = _stage2_instruction_for_index(stage2_rows, idx, stage2_instruction)
        s2_gt = stage2_rows[idx].get("output") if 0 <= idx < len(stage2_rows) else None

        ctx1 = {
            "instruction": s1_instruction,
            "input": s1_input,
            "stage1_pred": "",
            "no_think_suffix": "",
            "few_shot_block": few_shot_block_stage1,
        }
        msgs1 = build_messages_from_stage_config(s1_cfg, ctx1)
        t1 = chat_completion_text(
            client,
            model,
            msgs1,
            temperature=gen_cfg["temperature1"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_tokens1"],
            extra_body=extra_body,
        )
        if sleep_s > 0:
            time.sleep(sleep_s)

        p1 = t1
        if no_think:
            p1 = _strip_think_blocks(p1)
        print(f"\n[{idx}] 阶段1 pred 预览：{p1[:500]}{'...' if len(p1) > 500 else ''}")

        ctx2 = {
            "instruction": s2_inst_i,
            "input": s1_input,
            "stage1_pred": p1,
            "no_think_suffix": nt_suffix,
            "few_shot_block": few_shot_block_stage2,
        }
        msgs2 = build_messages_from_stage_config(s2_cfg, ctx2)
        t2 = chat_completion_text(
            client,
            model,
            msgs2,
            temperature=gen_cfg["temperature2"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_tokens2"],
            extra_body=extra_body,
        )
        if sleep_s > 0:
            time.sleep(sleep_s)

        p2 = t2
        if no_think:
            p2 = _strip_think_blocks(p2)
        print(f"\n[{idx}] 阶段2 pred 预览：{p2[:500]}{'...' if len(p2) > 500 else ''}")

        row: Dict[str, Any] = {
            "id": idx,
            "stage1_source_index": s1_source_idx,
            "stage1": {
                "instruction": s1_instruction,
                "input": s1_input,
                "gt": s1.get("output"),
                "pred": p1,
            },
            "stage2": {
                "instruction": s2_inst_i,
                "input": p1,
                "gt": s2_gt,
                "pred": p2,
            },
        }
        if few_shot_row_meta:
            row["few_shot_meta"] = few_shot_row_meta
        outputs.append(row)
        outputs.sort(key=lambda r: int(r.get("id") or 0))
        save_outputs_jsonl(args.output_jsonl, outputs)

    print(f"\n生成阶段已全部结束（{len(outputs)} 条），正在强制落盘 JSONL / XLSX / 生成检查点…")
    persist_generation_outputs(args.output_jsonl, args.output_xlsx, outputs)

    if args.eval:
        _evaluate_outputs_inplace(outputs, args, bailian_key)
        save_outputs_jsonl(args.output_jsonl, outputs, fsync=True)
        save_outputs_xlsx(args.output_xlsx, outputs)
        save_eval_artifacts(args.output_jsonl, outputs)

    print("\n" + "=" * 80)
    print(f"完成，共 {len(outputs)} 条；JSONL：{args.output_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    main()
