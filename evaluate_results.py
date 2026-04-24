"""
评估三个 Expert 的推理结果准确率
- ToolRL:       规则匹配（格式 + tool_call 正确性）
- HotpotQA:     规则匹配（ground_truth 包含检查）
- CodeContests: 沙箱执行代码，对比 test_input/test_output

用法:
    python evaluate_results.py                          # 评估全部
    python evaluate_results.py --task toolrl             # 仅评估某个任务
    python evaluate_results.py --task codecontests --timeout 10
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "inference_results_1"


# ============================================================
# ToolRL 评估（复用原始 reward 逻辑）
# ============================================================

def _match_score(list1, list2):
    """计算两个列表的频率感知相似度（忽略顺序）"""
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    count1 = Counter(list1)
    count2 = Counter(list2)
    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection
    return intersection / max_possible if max_possible > 0 else 0.0


def _compute_tool_call_reward(gt_tools, pd_tools):
    """对比 ground_truth 和 predicted 的 tool_call 列表，返回 0~1 分数"""
    if gt_tools == pd_tools:
        return 1.0

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = _match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        local_max_possible += 1.0 + len(gt_params)

        best_match_score = 0.0
        best_match_index = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            pd_params = pd_tool["parameters"]
            param_score = _match_score(list(gt_params.keys()), list(pd_params.keys()))
            correctness_score = sum(
                1.0 for k, v in gt_params.items()
                if k in pd_params and pd_params[k] == v
            )
            total_score = param_score + correctness_score
            if total_score > best_match_score:
                best_match_score = total_score
                best_match_index = i

        if best_match_index >= 0:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    return score / local_max_possible if local_max_possible > 0 else 0.0


def _check_toolrl_format(response, ground_truth):
    """检查 response 是否符合 ToolRL 要求的格式，返回 True/False

    格式要求（来自原始 rlla.py 的 customize_format_reward_func）：
    - 如果 GT 只有 <response>：需 \n<response>...</response>
    - 如果 GT 只有 <tool_call>：需 \n<tool_call>\n...\n</tool_call>
    - 如果 GT 有两者：需 \n<tool_call>...\n</tool_call>\n<response>...</response>
    实际推理中模型不一定总输出 <think>，因此这里放宽为仅检查关键标签的正确配对。
    """
    has_response_tag = "<response>" in ground_truth
    has_tool_call_tag = "<tool_call>" in ground_truth

    if has_response_tag and not has_tool_call_tag:
        return ("<response>" in response and "</response>" in response
                and response.count("<response>") == 1
                and response.count("</response>") == 1
                and "<tool_call>" not in response)
    elif not has_response_tag and has_tool_call_tag:
        return ("<tool_call>" in response and "</tool_call>" in response
                and response.count("<tool_call>") == 1
                and response.count("</tool_call>") == 1)
    elif has_response_tag and has_tool_call_tag:
        return ("<tool_call>" in response and "</tool_call>" in response
                and "<response>" in response and "</response>" in response
                and response.count("<tool_call>") == 1
                and response.count("</tool_call>") == 1
                and response.count("<response>") == 1
                and response.count("</response>") == 1)
    else:
        # GT 中没有 tool_call 也没有 response，则 response 也不该有
        return "<tool_call>" not in response and "<response>" not in response


def _eval_toolrl_correctness(response, ground_truth):
    """评估 ToolRL 的 tool_call 正确性，返回 0~1 分数"""
    if "<tool_call>" not in ground_truth:
        return 0.0  # 无 tool_call 的样本不参与正确性评分

    try:
        gt_tool_call = ground_truth.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = [json.loads(tool) for tool in gt_tool_call.split("\n")]
    except (IndexError, json.JSONDecodeError):
        return 0.0

    try:
        assert "<tool_call>" in response and "</tool_call>" in response
        pd_tool_call = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        pd_tools = [json.loads(tool) for tool in pd_tool_call.split("\n")]
        return _compute_tool_call_reward(gt_tools, pd_tools)
    except (AssertionError, IndexError, json.JSONDecodeError, ValueError, Exception):
        return 0.0


def evaluate_toolrl(results_path):
    """评估 ToolRL 推理结果"""
    samples = []
    with open(results_path) as f:
        for line in f:
            samples.append(json.loads(line))

    format_correct = 0
    tool_call_scores = []
    exact_match_count = 0
    tool_call_samples = 0
    correct_indices = []
    correct_samples = []

    for sample_idx, sample in enumerate(samples):
        response = sample["response"]
        ground_truth = sample["ground_truth"]

        format_ok = _check_toolrl_format(response, ground_truth)
        if format_ok:
            format_correct += 1

        # tool_call 正确性
        is_correct = False
        if "<tool_call>" in ground_truth:
            tool_call_samples += 1
            score = _eval_toolrl_correctness(response, ground_truth)
            tool_call_scores.append(score)
            if score == 1.0:
                exact_match_count += 1
                is_correct = True
        else:
            # GT 中无 tool_call，格式正确即算正确
            is_correct = format_ok

        if is_correct:
            correct_indices.append(sample["index"])
            correct_samples.append(sample)

    total = len(samples)
    avg_tool_score = sum(tool_call_scores) / len(tool_call_scores) if tool_call_scores else 0.0

    return {
        "task": "toolrl",
        "total_samples": total,
        "format_accuracy": format_correct / total,
        "format_correct": format_correct,
        "tool_call_samples": tool_call_samples,
        "tool_call_exact_match": exact_match_count,
        "tool_call_exact_match_rate": exact_match_count / tool_call_samples if tool_call_samples else 0.0,
        "tool_call_avg_score": avg_tool_score,
        "correct_indices": correct_indices,
        "correct_samples": correct_samples,
    }


# ============================================================
# HotpotQA 评估（MemAgent 官方 verifier 对齐）
# ============================================================

def _normalize_answer(text):
    """归一化答案文本（与 MemAgent 官方 testing verifier 一致）：
    小写 → 去标点 → 去冠词 → 去多余空格"""
    text = text.lower()
    exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    text = "".join(ch for ch in text if ch not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


def _last_boxed_only_string(string):
    """提取最后一个 \\boxed{...} 内容（与 MemAgent 官方 verifier 一致）"""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def _extract_boxed_answer(string):
    """从 \\boxed{...} 中提取答案"""
    boxed_str = _last_boxed_only_string(string)
    if boxed_str is None:
        return None
    if "\\boxed " in boxed_str:
        return boxed_str[len("\\boxed "):]
    left = "\\boxed{"
    if boxed_str.startswith(left) and boxed_str.endswith("}"):
        return boxed_str[len(left):-1]
    return None


def _extract_prediction(response):
    """从 response 中提取预测答案，优先 \\boxed{}，其次 'the answer is'"""
    # 优先从 \boxed{} 提取
    boxed = _extract_boxed_answer(response)
    if boxed is not None:
        return boxed.strip()

    # 其次尝试 "the answer is ..." 模式
    cleaned = response.replace("*", "")
    if "the answer is" in cleaned:
        answer = cleaned.rsplit("the answer is", 1)[-1].strip().strip(".").strip()
        if answer:
            return answer

    # 兜底：返回整个 response（让 F1 等指标仍能计算）
    return response


def _exact_match(prediction, ground_truth):
    """归一化后的精确匹配"""
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _sub_exact_match(prediction, ground_truth):
    """归一化后的子串匹配（双向）"""
    norm_pred = _normalize_answer(prediction)
    norm_gt = _normalize_answer(ground_truth)
    return norm_gt in norm_pred or norm_pred in norm_gt


def _compute_f1(prediction, ground_truth):
    """计算 token 级 F1 分数（与 MemAgent 官方 verifier 一致）"""
    norm_pred = _normalize_answer(prediction)
    norm_gt = _normalize_answer(ground_truth)

    if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return 0.0
    if norm_gt in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return 0.0

    prediction_tokens = norm_pred.split()
    ground_truth_tokens = norm_gt.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_hotpotqa(results_path):
    """评估 HotpotQA 推理结果（对齐 MemAgent 官方 verifier）"""
    samples = []
    with open(results_path) as f:
        for line in f:
            samples.append(json.loads(line))

    boxed_found = 0
    exact_match_count = 0
    sub_exact_match_count = 0
    f1_scores = []
    correct_indices = []
    correct_samples = []

    for sample in samples:
        response = sample["response"]
        ground_truths = sample["ground_truth"]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        prediction = _extract_prediction(response)
        has_boxed = _extract_boxed_answer(response) is not None
        if has_boxed:
            boxed_found += 1

        is_em = any(_exact_match(prediction, gt) for gt in ground_truths)
        is_sub_em = any(_sub_exact_match(prediction, gt) for gt in ground_truths)

        if is_em:
            exact_match_count += 1
        if is_sub_em:
            sub_exact_match_count += 1

        # 以 sub_exact_match 作为"正确"判定
        if is_sub_em:
            correct_indices.append(sample["index"])
            correct_samples.append(sample)

        best_f1 = max(_compute_f1(prediction, gt) for gt in ground_truths)
        f1_scores.append(best_f1)

    total = len(samples)
    return {
        "task": "hotpotqa",
        "total_samples": total,
        "boxed_found": boxed_found,
        "boxed_rate": boxed_found / total,
        "exact_match": exact_match_count,
        "exact_match_rate": exact_match_count / total,
        "sub_exact_match": sub_exact_match_count,
        "sub_exact_match_rate": sub_exact_match_count / total,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "correct_indices": correct_indices,
        "correct_samples": correct_samples,
    }


# ============================================================
# CodeContests 评估（沙箱执行）
# ============================================================

def _extract_python_code(response):
    """从 response 中提取 Python 代码"""
    # 优先尝试提取 ```python ... ``` 代码块
    code_blocks = re.findall(r"```(?:python|Python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        # 取最后一个代码块（通常是最终版本）
        return code_blocks[-1].strip()

    # 如果没有代码块标记，尝试提取看起来像 Python 代码的部分
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "if ", "for ",
                                "while ", "try:", "except", "with ", "print(",
                                "input(", "#", "n =", "t =", "T =")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return None


def _run_code_with_input(code, test_input, timeout=10):
    """在子进程中运行代码，传入 test_input，返回 stdout"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return None, -1
    except Exception as error:
        return None, -2
    finally:
        os.unlink(tmp_path)


def _compare_output(actual, expected):
    """对比实际输出和期望输出（与 CURE 官方 test_if_eq 一致）
    将所有空白字符归一化为单个空格后比较"""
    if actual is None:
        return False
    return " ".join(actual.split()) == " ".join(expected.split())


def evaluate_codecontests(results_path, timeout=10):
    """评估 CodeContests 推理结果"""
    samples = []
    with open(results_path) as f:
        for line in f:
            samples.append(json.loads(line))

    total = len(samples)
    code_extracted = 0
    all_passed = 0
    partial_results = []  # 每个样本的通过率
    correct_indices = []
    correct_samples = []

    for idx, sample in enumerate(samples):
        response = sample["response"]
        test_inputs = sample.get("test_input", [])
        test_outputs = sample.get("test_output", [])

        code = _extract_python_code(response)
        if code is None:
            partial_results.append(0.0)
            continue
        code_extracted += 1

        passed = 0
        total_tests = min(len(test_inputs), len(test_outputs))
        if total_tests == 0:
            partial_results.append(0.0)
            continue

        for test_in, test_out in zip(test_inputs, test_outputs):
            actual_output, returncode = _run_code_with_input(code, test_in, timeout=timeout)
            if _compare_output(actual_output, test_out):
                passed += 1

        pass_rate = passed / total_tests
        partial_results.append(pass_rate)
        if passed == total_tests:
            all_passed += 1
            correct_indices.append(sample.get("index", idx))
            correct_samples.append(sample)

        print(f"  [{idx+1}/{total}] index={sample.get('index','?')} "
              f"passed={passed}/{total_tests} ({pass_rate:.1%})")

    avg_pass_rate = sum(partial_results) / len(partial_results) if partial_results else 0.0
    return {
        "task": "codecontests",
        "total_samples": total,
        "code_extracted": code_extracted,
        "code_extract_rate": code_extracted / total,
        "all_tests_passed": all_passed,
        "all_tests_passed_rate": all_passed / total,
        "avg_test_pass_rate": avg_pass_rate,
        "correct_indices": correct_indices,
        "correct_samples": correct_samples,
    }


# ============================================================
# 主入口
# ============================================================

def print_results(results):
    """格式化打印评估结果"""
    task = results["task"]
    print(f"\n{'='*60}")
    print(f"  {task.upper()} 评估结果")
    print(f"{'='*60}")

    if task == "toolrl":
        print(f"  样本总数:          {results['total_samples']}")
        print(f"  格式正确数:        {results['format_correct']}/{results['total_samples']} "
              f"({results['format_accuracy']:.1%})")
        print(f"  Tool Call 样本数:  {results['tool_call_samples']}")
        print(f"  精确匹配数:        {results['tool_call_exact_match']}/{results['tool_call_samples']} "
              f"({results['tool_call_exact_match_rate']:.1%})")
        print(f"  Tool Call 平均分:  {results['tool_call_avg_score']:.4f}")

    elif task == "hotpotqa":
        print(f"  样本总数:          {results['total_samples']}")
        print(f"  \\boxed 提取成功:   {results['boxed_found']}/{results['total_samples']} "
              f"({results['boxed_rate']:.1%})")
        print(f"  Exact Match:       {results['exact_match']}/{results['total_samples']} "
              f"({results['exact_match_rate']:.1%})")
        print(f"  Sub Exact Match:   {results['sub_exact_match']}/{results['total_samples']} "
              f"({results['sub_exact_match_rate']:.1%})")
        print(f"  平均 F1 分数:      {results['avg_f1']:.4f}")

    elif task == "codecontests":
        print(f"  样本总数:          {results['total_samples']}")
        print(f"  代码提取成功:      {results['code_extracted']}/{results['total_samples']} "
              f"({results['code_extract_rate']:.1%})")
        print(f"  全部用例通过:      {results['all_tests_passed']}/{results['total_samples']} "
              f"({results['all_tests_passed_rate']:.1%})")
        print(f"  平均用例通过率:    {results['avg_test_pass_rate']:.1%}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="评估三个 Expert 的推理结果准确率")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "toolrl", "codecontests", "hotpotqa"],
                        help="要评估的任务（默认全部）")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                        help="推理结果目录路径")
    parser.add_argument("--timeout", type=int, default=10,
                        help="CodeContests 代码执行超时时间（秒）")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_results = {}

    task_map = {
        "toolrl": ("toolrl_inference_results.jsonl", evaluate_toolrl),
        "hotpotqa": ("hotpotqa_inference_results.jsonl", evaluate_hotpotqa),
        "codecontests": ("codecontests_inference_results.jsonl", None),
    }

    tasks_to_eval = list(task_map.keys()) if args.task == "all" else [args.task]

    for task_name in tasks_to_eval:
        filename, eval_fn = task_map[task_name]
        filepath = results_dir / filename

        if not filepath.exists():
            print(f"[跳过] {task_name}: 文件不存在 {filepath}")
            continue

        print(f"\n>>> 正在评估 {task_name} ...")

        if task_name == "codecontests":
            results = evaluate_codecontests(filepath, timeout=args.timeout)
        else:
            results = eval_fn(filepath)

        all_results[task_name] = results
        print_results(results)
    # 保存结果到 JSON（排除 correct_samples 避免文件过大）
    output_path = results_dir / "evaluation_summary.json"
    summary = {}
    for task_name, results in all_results.items():
        summary[task_name] = {
            k: v for k, v in results.items()
            if k not in ("correct_samples",)
        }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存至: {output_path}")

    # 保存正确样本到 datasets 目录
    _save_correct_samples(all_results, args)


# 任务名 → 输出文件名 & task_name 映射（对齐 Qwen/dataset.py 的 TASK_MODEL_ID）
TASK_TO_DATASET = {
    "toolrl": ("ToolCall.json", "ToolCall"),
    "hotpotqa": ("Memory.json", "Memory"),
    "codecontests": ("Code.json", "Code"),
}

DATASETS_DIR = Path(__file__).parent / "datasets"


def _save_correct_samples(all_results, args):
    """将正确样本保存到 datasets 目录，保留原始 messages 格式。
    对于 HotpotQA (Memory) 任务，会展开中间 chunk 轮的 (messages, response)。
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, results in all_results.items():
        correct_samples = results.get("correct_samples", [])
        correct_indices = results.get("correct_indices", [])
        if not correct_samples:
            print(f"[{task_name}] 无正确样本，跳过保存")
            continue

        filename, dataset_task_name = TASK_TO_DATASET.get(task_name, (f"{task_name}.json", task_name))
        output_path = DATASETS_DIR / filename

        dataset_entries = []
        for sample_idx, sample in enumerate(correct_samples):
            question_id = sample.get("index", sample_idx)

            # HotpotQA: 展开中间 chunk 轮 + 最终轮
            if task_name == "hotpotqa" and "chunk_rounds" in sample:
                for round_idx, chunk_round in enumerate(sample["chunk_rounds"]):
                    dataset_entries.append({
                        "messages": chunk_round["messages"],
                        "response": chunk_round["response"],
                        "task_name": dataset_task_name,
                        "question_id": question_id,
                        "round_type": "chunk",
                        "round_idx": round_idx,
                    })

            # 最终轮（所有任务都保存）
            dataset_entries.append({
                "messages": sample.get("messages", []),
                "response": sample.get("response", ""),
                "task_name": dataset_task_name,
                "question_id": question_id,
                "round_type": "final" if task_name == "hotpotqa" else None,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset_entries, f, indent=2, ensure_ascii=False)

        final_count = len(correct_samples)
        total_count = len(dataset_entries)
        print(f"[{task_name}] 保存 {total_count} 条样本到: {output_path}")
        if task_name == "hotpotqa":
            chunk_count = total_count - final_count
            print(f"  其中: {chunk_count} 条 chunk 轮 + {final_count} 条最终轮")
        print(f"  正确样本 indices: {correct_indices}")


if __name__ == "__main__":
    main()