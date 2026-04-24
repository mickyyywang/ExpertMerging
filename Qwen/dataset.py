import json
import logging
import random
from pathlib import Path

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Task name -> Teacher expert model index (1-based)
# Expert 1: Tool calling (Qwen2.5-7B-Instruct-ToolRL-grpo-cold)
# Expert 2: Memory agent (RL-MemoryAgent-7B)
# Expert 3: Code reasoning (ReasonFlux-Coder-7B)
TASK_MODEL_ID = {
    "ToolCall": 1,
    "ToolPlan": 1,
    "Memory": 2,
    "MemoryQA": 2,
    "Code": 3,
    "Math": 3,
}


class ExpertMergingDataset(Dataset):
    """Dataset for ExpertMerging training on pure-text Qwen models.

    Expected data layout::

        data_dir/
            ToolCall.json
            Memory.json
            Code.json
            ...

    Each JSON file is a list of objects with chat messages format::

        [
            {
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."}
                ],
                "response": "expected assistant response",
                "task_name": "ToolCall",
                "question_id": 0
            },
            ...
        ]

    Legacy format with a flat "question" field is also supported for
    backward compatibility.
    """

    def __init__(
        self,
        data_dir: str,
        samples_per_task=None,
        default_samples_per_task: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.samples_per_task = samples_per_task
        self.default_samples_per_task = default_samples_per_task
        self.task_to_model_idx = TASK_MODEL_ID

        self.samples = []
        self._load_samples()

    def _normalize_sample(self, sample: dict) -> dict:
        """Normalize a sample to always have a messages field.

        If the sample already contains messages (a list of role/content
        dicts), keep it as-is.  Otherwise fall back to the legacy
        question field and wrap it into a single-user-turn message list.
        """
        if "messages" in sample and isinstance(sample["messages"], list):
            return sample

        # Legacy format: flat question string -> wrap into messages
        question_text = sample.get("question", "")
        sample["messages"] = [{"role": "user", "content": question_text}]
        return sample

    def _load_samples(self):
        for task_name in self.task_to_model_idx:
            json_file = self.data_dir / f"{task_name}.json"
            if not json_file.exists():
                logger.warning(f"{json_file} not found, skipping {task_name}")
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)

            # Determine sample count
            if self.samples_per_task is None:
                sampled_data = task_data
            elif isinstance(self.samples_per_task, int):
                num_samples = min(len(task_data), self.samples_per_task)
                sampled_data = task_data[:num_samples]
            elif isinstance(self.samples_per_task, dict):
                num_samples = min(
                    len(task_data),
                    self.samples_per_task.get(task_name, self.default_samples_per_task),
                )
                sampled_data = task_data[:num_samples]
            else:
                sampled_data = task_data

            logger.info(f"Loaded {len(sampled_data)} samples for {task_name}")

            for idx, sample in enumerate(sampled_data):
                sample = self._normalize_sample(sample)
                sample.setdefault("task_name", task_name)
                sample.setdefault("question_id", idx)
                sample["teacher_model_idx"] = self.task_to_model_idx[task_name]
                self.samples.append(sample)

        logger.info(f"Total samples: {len(self.samples)}")
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "messages": sample["messages"],
            "response": sample.get("response", ""),
            "question_id": sample["question_id"],
            "task_name": sample["task_name"],
            "teacher_model_idx": sample["teacher_model_idx"],
        }
