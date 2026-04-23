import logging
import sys
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("../global_utils")
from expert_merging_base import (
    BaseExpertMergingTrainer,
    BaseParametricTaskVectorModel,
)

logger = logging.getLogger(__name__)


class ParametricTaskVectorModel(BaseParametricTaskVectorModel):
    """Parametric task vector model for pure-text Qwen2.5 models."""

    def __init__(
        self,
        base_model: nn.Module,
        teacher_models: List[nn.Module],
        exclude_param_names_regex: List[str],
        device: torch.device = None,
        weight_coeffs_init_value: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            base_model,
            teacher_models,
            exclude_param_names_regex,
            device,
            weight_coeffs_init_value,
            **kwargs,
        )


class ExpertMergingTrainer(BaseExpertMergingTrainer):
    """ExpertMerging trainer adapted for pure-text Qwen2.5 models."""

    # Qwen2.5 parameters to exclude from merging:
    #   - lm_head: output projection (tied with embed_tokens in some configs)
    #   - norm: all RMSNorm layers
    #   - embed_tokens: input embeddings
    #   - bias: all bias terms
    DEFAULT_EXCLUDE_REGEX = [
        ".*lm_head.*",
        ".*norm.*",
        ".*embed_tokens.*",
        ".*bias.*",
    ]

    def __init__(
        self,
        base_model: nn.Module,
        teacher_models: List[nn.Module],
        tokenizer,
        temperature: float = 3.0,
        learning_rate: float = 1e-4,
        log_dir: str = "logs/expert_merging",
        exclude_param_names_regex: List[str] = None,
        gradient_accumulation_steps: int = 8,
        cpu_offload_teachers: bool = True,
        mixed_precision: str = "fp16",
        seed: int = 42,
        sparsity_reg_weight: float = 0.1,
        max_weight_norm: float = 1.0,
        hidden_states_layers: List[int] = None,
        hidden_states_weight: float = 0.1,
        loss_alphas: Dict[str, float] = None,
        student_device: torch.device = None,
        teacher_device: torch.device = None,
        loss_device: torch.device = None,
        weight_coeffs_init_value: float = 0.1,
        coeffs_size_dict: dict = None,
        task_vector_coeffs_dict: dict = None,
        max_length: int = 2048,
        group_shuffle: bool = False,
    ):
        if exclude_param_names_regex is None:
            exclude_param_names_regex = self.DEFAULT_EXCLUDE_REGEX

        super().__init__(
            base_model=base_model,
            teacher_models=teacher_models,
            temperature=temperature,
            learning_rate=learning_rate,
            log_dir=log_dir,
            exclude_param_names_regex=exclude_param_names_regex,
            gradient_accumulation_steps=gradient_accumulation_steps,
            cpu_offload_teachers=cpu_offload_teachers,
            mixed_precision=mixed_precision,
            seed=seed,
            sparsity_reg_weight=sparsity_reg_weight,
            max_weight_norm=max_weight_norm,
            hidden_states_layers=hidden_states_layers,
            hidden_states_weight=hidden_states_weight,
            loss_alphas=loss_alphas,
            student_device=student_device,
            teacher_device=teacher_device,
            loss_device=loss_device,
            weight_coeffs_init_value=weight_coeffs_init_value,
            group_shuffle=group_shuffle,
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cpu_offload_teachers = cpu_offload_teachers
        self.teacher_models = teacher_models

        # Create parametric student model
        self.student_model = ParametricTaskVectorModel(
            base_model=base_model,
            teacher_models=teacher_models,
            exclude_param_names_regex=exclude_param_names_regex,
            device=student_device,
            weight_coeffs_init_value=weight_coeffs_init_value,
            coeffs_size_dict=coeffs_size_dict or {},
            task_vector_coeffs_dict=task_vector_coeffs_dict or {},
        )

        # Offload teachers to CPU
        if self.cpu_offload_teachers:
            for i, teacher in enumerate(teacher_models):
                self.teacher_models[i] = teacher.cpu()

        # Optimizer — only optimize weight_coeffs
        learnable_params = list(self.student_model.weight_coeffs.parameters())
        self.optimizer = torch.optim.AdamW(
            learnable_params,
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=learning_rate * 0.1
        )

        # Prepare with accelerator
        self.student_model, self.optimizer = self.accelerator.prepare(
            self.student_model, self.optimizer
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("parametric_expert_merging_qwen")

    def _prepare_inputs(self, question: str, response: str, model=None):
        """Prepare tokenized inputs using Qwen chat template.

        Constructs a chat-formatted prompt with ``question`` as user turn and
        ``response`` as assistant turn, tokenizes it and returns tensors ready
        for the model forward pass.
        """
        messages = [{"role": "user", "content": question}]
        if response:
            messages.append({"role": "assistant", "content": response})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not response,
        )

        model_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        device = self.accelerator.device
        return {
            "input_ids": model_inputs["input_ids"].to(device),
            "attention_mask": model_inputs["attention_mask"].to(device),
            "return_dict": True,
        }

    def train_step(self, batch):
        """Single training step: distill one teacher into the merged student."""
        teacher_idx = batch["teacher_model_idx"][0]
        question = batch["question"][0]
        response = batch["response"][0]
        task_name = batch["task_name"][0]

        teacher_model = self._get_teacher_model(teacher_idx)

        # Prepare inputs
        inputs = self._prepare_inputs(question, response, model=teacher_model)

        # Teacher forward (no grad)
        teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = teacher_model(
                **{
                    k: v.to(self.teacher_device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                },
                output_hidden_states=True,
            )

        self._offload_teacher_model(teacher_model, teacher_idx)

        # Student forward
        self.student_model.train()
        student_outputs = self.student_model(
            **inputs, use_cache=False, output_hidden_states=True,
        )

        total_loss, loss_dict = self.compute_loss(
            teacher_outputs, student_outputs, task_name=task_name,
        )

        self.accelerator.backward(total_loss)
        return loss_dict

    def create_balanced_dataloader(self, dataset, batch_size=1):
        """Create a dataloader that balances samples across teachers."""
        teacher_samples = defaultdict(list)
        for i, sample in enumerate(dataset):
            teacher_samples[sample["teacher_model_idx"]].append(i)

        balanced_indices = []
        max_samples = max(len(s) for s in teacher_samples.values())

        for step in range(max_samples):
            for teacher_idx in sorted(teacher_samples):
                samples = teacher_samples[teacher_idx]
                if samples:
                    balanced_indices.append(samples[step % len(samples)])

        class _BalancedDataset:
            def __init__(self, original, indices):
                self.dataset = original
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

        return DataLoader(
            _BalancedDataset(dataset, balanced_indices),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: {
                "teacher_model_idx": [x[0]["teacher_model_idx"]],
                "question": [x[0]["question"]],
                "response": [x[0]["response"]],
                "task_name": [x[0]["task_name"]],
                "question_id": [x[0]["question_id"]],
            },
        )
