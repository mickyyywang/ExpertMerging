"""
推理脚本：用三个 expert model 分别对三个 agent 数据集（train set）进行推理。

数据集 → 模型 映射：
  ToolRL          → Qwen2.5-7B-Instruct-ToolRL-grpo-cold
  CodeContests    → ReasonFlux-Coder-7B
  HotpotQA        → RL-MemoryAgent-7B

用法:
    python run_inference.py \
        --task toolrl \
        --max_samples 100 \
        --max_new_tokens 2048 \
        --batch_size 4

    python run_inference.py \
        --task codecontests \
        --max_samples 100

    python run_inference.py \
        --task hotpotqa \
        --max_samples 100
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 路径配置
# ============================================================
MODEL_DIR = "/primus_xpfs_workspace_T04/wmq/merge/models"
DATA_DIR = "/primus_xpfs_workspace_T04/wmq/merge/datasets"

TASK_CONFIG = {
    "toolrl": {
        "model_path": f"{MODEL_DIR}/Qwen2.5-7B-Instruct-ToolRL-grpo-cold",
        "data_path": f"{DATA_DIR}/ToolRL/dataset/rlla_4k/train.parquet",
        "data_format": "parquet",
    },
    "codecontests": {
        "model_path": f"{MODEL_DIR}/ReasonFlux-Coder-7B",
        "data_path": f"{DATA_DIR}/CodeContests_train/train/CodeContests_train.json",
        "data_format": "json",
    },
    "hotpotqa": {
        "model_path": f"{MODEL_DIR}/RL-MemoryAgent-7B",
        "data_path": f"{DATA_DIR}/hotpotqa/hotpotqa_train_32k.parquet",
        "data_format": "parquet",
    },
}


# ============================================================
# 数据加载：构建 messages list
# ============================================================

def load_toolrl(data_path, max_samples=None):
    """ToolRL: prompt 已经是 messages list（含 system + user）"""
    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.head(max_samples)

    samples = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in prompt]

        reward_model = row.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "")
        if hasattr(ground_truth, "tolist"):
            ground_truth = ground_truth.tolist()

        samples.append({
            "messages": messages,
            "index": row.get("extra_info", {}).get("index", len(samples)),
            "ground_truth": ground_truth,
            "style": reward_model.get("style", ""),
        })
    return samples


def load_codecontests(data_path, max_samples=None):
    """CodeContests: 使用 CURE 官方 prompt 模板构造 messages（含 system + user）"""
    with open(data_path, "r") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]

    system_content = "You are a helpful assistant help user solve problems."
    user_template = (
        "You need to think first then write python script. "
        "You should use input() to input and print() to output in your script. "
        "Your code should output the results based on the input read in, "
        "rather than generating the given test example.\n"
        "This is the problem:\n{question}"
    )

    samples = []
    for idx, item in enumerate(data):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_template.format(question=item["question"])},
        ]
        samples.append({
            "messages": messages,
            "index": item.get("task_id", idx),
            "test_input": item.get("test_input"),
            "test_output": item.get("test_output"),
            "exe_method": item.get("exe_method"),
        })
    return samples


def load_hotpotqa(data_path, max_samples=None):
    """HotpotQA: 保留 context 和 question 分离，用于 MemAgent 多轮推理"""
    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.head(max_samples)

    samples = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        user_question = prompt[0]["content"]
        context = row["context"]

        reward_model = row.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "")
        if hasattr(ground_truth, "tolist"):
            ground_truth = ground_truth.tolist()

        samples.append({
            "context": context,
            "question": user_question,
            "messages": [],  # 多轮推理时动态构建
            "index": row.get("extra_info", {}).get("index", len(samples)),
            "ground_truth": ground_truth,
            "style": reward_model.get("style", ""),
        })
    return samples


LOADER_MAP = {
    "toolrl": load_toolrl,
    "codecontests": load_codecontests,
    "hotpotqa": load_hotpotqa,
}

# ============================================================
# MemAgent 多轮推理模板（来自 MemAgent quickstart.py）
# ============================================================

MEMAGENT_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

MEMAGENT_TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

MEMAGENT_NO_MEMORY = "No previous memory"
MEMAGENT_CHUNK_SIZE = 5000       # tokens per chunk
MEMAGENT_MAX_MEMORY_TOKENS = 1024  # max tokens for memory update response


def _generate_single(model, tokenizer, messages, max_new_tokens, temperature=0.7, top_p=0.95):
    """对单条消息调用模型生成，返回 response 文本"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32768).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_hotpotqa_inference(model, tokenizer, samples, args):
    """MemAgent 多轮推理：逐 chunk 更新 memory，最后生成答案"""
    results = []
    start_time = time.time()

    for idx, sample in enumerate(tqdm(samples, desc="HotpotQA MemAgent Inference")):
        context = sample["context"]
        question = sample["question"]

        # 1. 将 context 按 token 分 chunk
        context_ids = tokenizer.encode(context)
        chunks = []
        for i in range(0, len(context_ids), MEMAGENT_CHUNK_SIZE):
            chunk_ids = context_ids[i : i + MEMAGENT_CHUNK_SIZE]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))

        # 2. 逐 chunk 更新 memory，同时记录每轮的 (messages, response)
        memory = MEMAGENT_NO_MEMORY
        chunk_rounds = []  # 中间轮记录
        for chunk_text in chunks:
            msg_content = MEMAGENT_TEMPLATE.format(
                prompt=question, memory=memory, chunk=chunk_text
            )
            chunk_messages = [{"role": "user", "content": msg_content}]
            memory = _generate_single(
                model, tokenizer, chunk_messages,
                max_new_tokens=MEMAGENT_MAX_MEMORY_TOKENS,
                temperature=args.temperature if args.do_sample else 0.7,
                top_p=args.top_p if args.do_sample else 0.95,
            )
            chunk_rounds.append({
                "messages": chunk_messages,
                "response": memory,
            })

        # 3. 最终回答
        final_msg = MEMAGENT_TEMPLATE_FINAL.format(prompt=question, memory=memory)
        final_messages = [{"role": "user", "content": final_msg}]
        response = _generate_single(
            model, tokenizer, final_messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else 0.7,
            top_p=args.top_p if args.do_sample else 0.95,
        )

        result = {
            "index": sample["index"],
            "task": "hotpotqa",
            "messages": final_messages,
            "response": response,
            "memory": memory,
            "chunk_rounds": chunk_rounds,  # 中间轮的 (messages, response)
        }
        if "ground_truth" in sample:
            result["ground_truth"] = sample["ground_truth"]
        if "style" in sample:
            result["style"] = sample["style"]
        results.append(result)

        if idx == 0:
            print(f"\n--- Sample 0 preview ---")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Final memory (first 200): {memory[:200]}")
            print(f"  Response (first 200): {response[:200]}")
            print()

    elapsed = time.time() - start_time
    print(f"\nDone! {len(results)} samples in {elapsed:.1f}s")
    return results


# ============================================================
# 推理
# ============================================================

def run_inference(args):
    task_cfg = TASK_CONFIG[args.task]
    model_path = task_cfg["model_path"]
    data_path = task_cfg["data_path"]

    print(f"{'='*60}")
    print(f"Task:  {args.task}")
    print(f"Model: {model_path}")
    print(f"Data:  {data_path}")
    print(f"{'='*60}")

    # 加载数据
    loader = LOADER_MAP[args.task]
    samples = loader(data_path, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    # 加载模型和分词器
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.task}_inference_results.jsonl"

    print(f"Output: {output_file}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Batch size: {args.batch_size}")
    print()

    # HotpotQA 走 MemAgent 多轮推理路径
    if args.task == "hotpotqa":
        results = run_hotpotqa_inference(model, tokenizer, samples, args)
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Results saved to: {output_file}")
        return

    # 其他任务：逐条或批量推理
    results = []
    start_time = time.time()

    for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Inferencing"):
        batch = samples[batch_start : batch_start + args.batch_size]

        # 用 chat template 构造输入
        batch_texts = []
        for sample in batch:
            text = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_texts.append(text)

        # 先单独 tokenize 记录每条的实际长度（不含 padding）
        individual_input_ids = [
            tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_input_length)["input_ids"].shape[1]
            for text in batch_texts
        ]

        # Batch tokenize（带 padding）
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode（用 padded 后的统一长度切分，因为 generate 返回的序列包含完整的 padded input）
        padded_input_len = inputs["input_ids"].shape[1]
        for i, (sample, output_ids) in enumerate(zip(batch, outputs)):
            generated_ids = output_ids[padded_input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            result = {
                "index": sample["index"],
                "task": args.task,
                "messages": sample["messages"],
                "response": response,
            }
            # 保留 ground_truth 等评估字段（如有）
            if "ground_truth" in sample:
                result["ground_truth"] = sample["ground_truth"]
            if "style" in sample:
                result["style"] = sample["style"]
            if "test_input" in sample:
                result["test_input"] = sample["test_input"]
            if "test_output" in sample:
                result["test_output"] = sample["test_output"]
            results.append(result)

    elapsed = time.time() - start_time

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(results)} samples in {elapsed:.1f}s")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Expert model inference on agent datasets")
    parser.add_argument("--task", type=str, required=True,
                        choices=["toolrl", "codecontests", "hotpotqa"],
                        help="Task/dataset to run inference on")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max new tokens to generate")
    parser.add_argument("--max_input_length", type=int, default=4096,
                        help="Max input token length (truncation)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (only with --do_sample)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling (only with --do_sample)")
    parser.add_argument("--output_dir", type=str,
                        default="/primus_xpfs_workspace_T04/wmq/merge/ExpertMerging/inference_results",
                        help="Output directory for results")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
