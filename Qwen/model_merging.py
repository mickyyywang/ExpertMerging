"""
Qwen2.5 Expert Merging — 入口脚本

将多个 Qwen2.5-7B expert 模型合并为一个统一模型。
支持 expert_merging（可学习权重）、task_arithmetic、ties、weight_average 等方法。

用法:
    cd ExpertMerging/Qwen
    python model_merging.py \
        --method expert_merging \
        --base_model /path/to/Qwen2.5-7B-Instruct \
        --expert_models /path/to/Expert1 /path/to/Expert2 /path/to/Expert3 \
        --data_dir ../dataset
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.append("../global_utils")
from config import save_config, setup_logging
from merging_utils import TaskVector, get_param_names_to_merge

logger = logging.getLogger(__name__)

# ============================================================
# Merge methods (non-parametric baselines)
# ============================================================

def task_arithmetic(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 0.5,
):
    """Task Arithmetic: base + scaling * Σ(task_vector_i)"""
    pretrained_param_dict = dict(merged_model.named_parameters())
    param_names_to_merge = get_param_names_to_merge(
        list(pretrained_param_dict.keys()), exclude_param_names_regex,
    )

    task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=m,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for m in models_to_merge
    ]

    merged_params = {}
    with torch.no_grad():
        for param_name in param_names_to_merge:
            base_param = pretrained_param_dict[param_name].data.clone()
            combined = torch.zeros_like(base_param)
            for tv in task_vectors:
                if param_name in tv.task_vector_param_dict:
                    combined += tv.task_vector_param_dict[param_name].to(base_param.device)
            merged_params[param_name] = base_param + scaling_coefficient * combined

    return merged_params


def weight_average(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    scaling_coefficient: float = 1.0,
):
    """Simple weight averaging of all models."""
    pretrained_param_dict = dict(merged_model.named_parameters())
    param_names_to_merge = get_param_names_to_merge(
        list(pretrained_param_dict.keys()), exclude_param_names_regex,
    )

    all_models = [merged_model] + models_to_merge
    merged_params = {}
    with torch.no_grad():
        for param_name in param_names_to_merge:
            stacked = torch.stack([
                dict(m.named_parameters())[param_name].data.to("cpu")
                for m in all_models
            ])
            merged_params[param_name] = stacked.mean(dim=0)

    return merged_params


def copy_params_to_model(params: dict, model: nn.Module):
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name].to(param_value.device))


# ============================================================
# Expert Merging (parametric, learnable weights)
# ============================================================

def expert_merging_method(
    base_model: nn.Module,
    expert_models: list,
    tokenizer,
    exclude_param_names_regex: list,
    data_dir: str,
    samples_per_task=None,
    default_samples_per_task: int = 100,
    num_epochs: int = 3,
    temperature: float = 1.0,
    learning_rate: float = 1e-6,
    log_dir: str = "logs/expert_merging",
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 8,
    seed: int = 42,
    sparsity_reg_weight: float = 0.1,
    hidden_states_layers: List[int] = None,
    hidden_states_weight: float = 0.1,
    weight_coeffs_init_value: float = 0.1,
    loss_alphas: Dict[str, float] = None,
    max_length: int = 2048,
    group_shuffle: bool = False,
):
    """Parametric Expert Merging with learnable weight coefficients."""
    from dataset import ExpertMergingDataset
    from parametric_task_vector_model import ExpertMergingTrainer

    logger.info("Starting Expert Merging (Qwen)...")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Sparsity reg weight: {sparsity_reg_weight}")
    logger.info(f"  Max length: {max_length}")

    # Create dataset
    dataset = ExpertMergingDataset(
        data_dir=data_dir,
        samples_per_task=samples_per_task,
        default_samples_per_task=default_samples_per_task,
    )

    # Create trainer
    trainer = ExpertMergingTrainer(
        base_model=base_model,
        teacher_models=expert_models,
        tokenizer=tokenizer,
        temperature=temperature,
        learning_rate=learning_rate,
        log_dir=log_dir,
        exclude_param_names_regex=exclude_param_names_regex,
        gradient_accumulation_steps=gradient_accumulation_steps,
        cpu_offload_teachers=True,
        mixed_precision=mixed_precision,
        seed=seed,
        sparsity_reg_weight=sparsity_reg_weight,
        hidden_states_layers=hidden_states_layers,
        hidden_states_weight=hidden_states_weight,
        loss_alphas=loss_alphas,
        weight_coeffs_init_value=weight_coeffs_init_value,
        max_length=max_length,
        group_shuffle=group_shuffle,
    )

    # Train
    trained_model = trainer.train(dataset, num_epochs=num_epochs)

    # Extract merged parameters
    merged_params = trained_model.get_final_merged_params()

    # Save weight coefficients
    weight_coeffs_path = os.path.join(log_dir, "weight_coeffs.json")
    os.makedirs(os.path.dirname(weight_coeffs_path), exist_ok=True)
    with open(weight_coeffs_path, "w") as f:
        json.dump(trainer.get_weight_coeffs_dict(mean=False), f, indent=2)
    logger.info(f"Saved weight coefficients → {weight_coeffs_path}")

    return merged_params


# ============================================================
# Model loading
# ============================================================

def load_models(base_model_path: str, expert_model_paths: List[str], dtype=torch.bfloat16):
    """Load base model, tokenizer, and all expert models."""
    logger.info(f"Loading base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, trust_remote_code=True,
    )

    expert_models = []
    for path in expert_model_paths:
        logger.info(f"Loading expert model: {path}")
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True,
        )
        expert_models.append(model)

    return base_model, expert_models, tokenizer


# ============================================================
# Save merged model
# ============================================================

def save_merged_model(base_model, merged_params, tokenizer, output_dir):
    """Apply merged params to base model and save."""
    os.makedirs(output_dir, exist_ok=True)
    copy_params_to_model(merged_params, base_model)
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved merged model → {output_dir}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 Expert Merging")

    parser.add_argument(
        "--method", type=str, default="expert_merging",
        choices=["expert_merging", "task_arithmetic", "weight_average"],
        help="Merging method",
    )
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument(
        "--expert_models", type=str, nargs="+", required=True,
        help="Paths to expert models",
    )
    parser.add_argument("--data_dir", type=str, default="../dataset", help="Training data dir")
    parser.add_argument(
        "--output_path", type=str, default="results/logs",
        help="Base output directory",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name (default: timestamp)")

    # Merge parameters
    parser.add_argument("--scaling_coefficient", type=float, default=0.5, help="Scaling for non-parametric methods")
    parser.add_argument(
        "--exclude_params", type=str, nargs="*",
        default=[".*lm_head.*", ".*norm.*", ".*embed_tokens.*", ".*bias.*"],
        help="Regex for parameters to exclude from merging",
    )

    # ExpertMerging training parameters
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsity_reg_weight", type=float, default=0.1)
    parser.add_argument("--weight_coeffs_init_value", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048, help="Max token length for training")
    parser.add_argument("--samples_per_task", type=int, default=100)
    parser.add_argument("--group_shuffle", action="store_true", default=False)

    # Hidden states alignment
    parser.add_argument("--hidden_states_layers", type=str, default=None, help="Comma-separated layer indices")
    parser.add_argument("--hidden_states_weight", type=float, default=0.1)

    # Task loss weights
    parser.add_argument("--loss_alphas", type=str, default=None, help='JSON dict of task→weight')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        args.run_name = time.strftime("%m%d-%H%M%S")

    log_dir_base = Path(args.output_path) / args.method / args.run_name
    run_logger = setup_logging(args.output_path, args.method, args.run_name)
    save_config(args, args.output_path)

    set_seed(args.seed)

    run_logger.info(f"Method: {args.method}")
    run_logger.info(f"Base model: {args.base_model}")
    run_logger.info(f"Expert models: {args.expert_models}")

    # Load models
    base_model, expert_models, tokenizer = load_models(
        args.base_model, args.expert_models,
    )

    # Parse optional JSON args
    hidden_states_layers = None
    if args.hidden_states_layers:
        hidden_states_layers = [int(x) for x in args.hidden_states_layers.split(",")]

    loss_alphas = json.loads(args.loss_alphas) if args.loss_alphas else None

    # Run merging
    exclude_regex = args.exclude_params

    if args.method == "expert_merging":
        base_model = base_model.cuda()
        merged_params = expert_merging_method(
            base_model=base_model,
            expert_models=expert_models,
            tokenizer=tokenizer,
            exclude_param_names_regex=exclude_regex,
            data_dir=args.data_dir,
            samples_per_task=args.samples_per_task,
            num_epochs=args.num_epochs,
            temperature=args.temperature,
            learning_rate=args.learning_rate,
            log_dir=str(log_dir_base / "board"),
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            sparsity_reg_weight=args.sparsity_reg_weight,
            hidden_states_layers=hidden_states_layers,
            hidden_states_weight=args.hidden_states_weight,
            weight_coeffs_init_value=args.weight_coeffs_init_value,
            loss_alphas=loss_alphas,
            max_length=args.max_length,
            group_shuffle=args.group_shuffle,
        )

    elif args.method == "task_arithmetic":
        base_model = base_model.cuda()
        expert_models = [m.cuda() for m in expert_models]
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=expert_models,
            exclude_param_names_regex=exclude_regex,
            scaling_coefficient=args.scaling_coefficient,
        )

    elif args.method == "weight_average":
        merged_params = weight_average(
            merged_model=base_model,
            models_to_merge=expert_models,
            exclude_param_names_regex=exclude_regex,
            scaling_coefficient=args.scaling_coefficient,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save merged model
    model_output_dir = str(log_dir_base / "model")
    save_merged_model(base_model, merged_params, tokenizer, model_output_dir)

    run_logger.info("Model merging completed!")


if __name__ == "__main__":
    main()
