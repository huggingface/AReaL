from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils import PreTrainedTokenizerBase

    from areal.api.cli_args import _DatasetConfig


def get_hf_text_dataset(
    dataset_config: _DatasetConfig,
    split: str | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Dataset:
    """Load a generic text/chat dataset from Hugging Face Hub."""
    resolved_split = dataset_config.split or split
    if resolved_split is None:
        raise ValueError(
            f"Dataset {dataset_config.path!r} requires a split. "
            "Set dataset_config.split or pass split=..."
        )

    dataset = load_dataset(
        path=dataset_config.path,
        name=dataset_config.config_name,
        split=resolved_split,
    )

    if dataset_config.type == "rl":
        return _build_rl_dataset(dataset, dataset_config, tokenizer)
    if dataset_config.type == "sft":
        return _build_sft_dataset(dataset, dataset_config, tokenizer)

    raise ValueError(
        f"Generic Hugging Face text dataset loading only supports type='rl' or type='sft', got {dataset_config.type!r}."
    )


def _map_dataset(
    dataset: Dataset, process, dataset_config: _DatasetConfig, **kwargs
) -> Dataset:
    return dataset.map(
        process,
        num_proc=_resolve_num_proc(dataset, dataset_config),
        **kwargs,
    )


def _filter_dataset(
    dataset: Dataset, predicate, dataset_config: _DatasetConfig, **kwargs
) -> Dataset:
    return dataset.filter(
        predicate,
        num_proc=_resolve_num_proc(dataset, dataset_config),
        **kwargs,
    )


def _resolve_num_proc(dataset: Dataset, dataset_config: _DatasetConfig) -> int | None:
    if dataset_config.num_proc is None:
        return None

    num_rows = dataset.num_rows
    if num_rows <= 0:
        return None

    num_proc = min(dataset_config.num_proc, num_rows)
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        num_proc = min(num_proc, cpu_count)
    return max(1, num_proc)


def _build_rl_dataset(dataset: Dataset, dataset_config: _DatasetConfig, tokenizer):
    column_names = list(dataset.column_names)
    prompt_completion_columns = _resolve_prompt_completion_columns(
        column_names, dataset_config
    )
    messages_column = _resolve_messages_column(column_names, dataset_config)

    if prompt_completion_columns is not None:
        prompt_column, completion_column = prompt_completion_columns

        def process(sample: dict[str, Any]) -> dict[str, Any]:
            result = {"messages": [{"role": "user", "content": sample[prompt_column]}]}
            if "answer" not in sample:
                result["answer"] = sample[completion_column]
            return result

        dataset = _map_dataset(dataset, process, dataset_config)
    elif messages_column is not None:

        def process(sample: dict[str, Any]) -> dict[str, Any]:
            return {
                "messages": _validate_messages(
                    sample[messages_column], dataset_config.path, messages_column
                )
            }

        dataset = _map_dataset(dataset, process, dataset_config)
    else:
        raise ValueError(
            f"Dataset {dataset_config.path!r} is not handled by a built-in adapter, and generic RL loading requires either a 'messages' column or configured prompt_column/completion_column fields."
        )

    if dataset_config.max_length is not None:
        if tokenizer is None:
            raise ValueError(
                "A tokenizer is required when max_length is set for generic RL datasets."
            )

        dataset = _filter_dataset(
            dataset,
            lambda sample: len(
                _tokenize_messages(
                    tokenizer, sample["messages"], add_generation_prompt=True
                )
            )
            <= dataset_config.max_length,
            dataset_config,
        )

    return dataset


def _build_sft_dataset(dataset: Dataset, dataset_config: _DatasetConfig, tokenizer):
    if tokenizer is None:
        raise ValueError("A tokenizer is required for generic SFT datasets.")

    column_names = list(dataset.column_names)
    prompt_completion_columns = _resolve_prompt_completion_columns(
        column_names, dataset_config
    )
    messages_column = _resolve_messages_column(column_names, dataset_config)

    if prompt_completion_columns is not None:
        prompt_column, completion_column = prompt_completion_columns

        def process(sample: dict[str, Any]) -> dict[str, list[int]]:
            messages = [
                {"role": "user", "content": sample[prompt_column]},
                {"role": "assistant", "content": sample[completion_column]},
            ]
            input_ids = _tokenize_messages(
                tokenizer, messages, add_generation_prompt=False
            )
            prompt_ids = _tokenize_messages(
                tokenizer, messages[:1], add_generation_prompt=True
            )
            return {
                "input_ids": input_ids,
                "loss_mask": _build_suffix_mask(len(input_ids), len(prompt_ids)),
            }

        dataset = _map_dataset(
            dataset,
            process,
            dataset_config,
            remove_columns=column_names,
        )
    elif messages_column is not None:

        def process(sample: dict[str, Any]) -> dict[str, list[int]]:
            messages = _validate_messages(
                sample[messages_column], dataset_config.path, messages_column
            )
            input_ids, loss_mask = _messages_to_sft_example(
                messages, tokenizer, dataset_config.path
            )
            return {"input_ids": input_ids, "loss_mask": loss_mask}

        dataset = _map_dataset(
            dataset,
            process,
            dataset_config,
            remove_columns=column_names,
        )
    else:
        raise ValueError(
            f"Dataset {dataset_config.path!r} is not handled by a built-in adapter, and generic SFT loading requires either a 'messages' column or configured prompt_column/completion_column fields."
        )

    if dataset_config.max_length is not None:
        dataset = _filter_dataset(
            dataset,
            lambda sample: len(sample["input_ids"]) <= dataset_config.max_length,
            dataset_config,
        )

    return dataset


def _resolve_messages_column(
    column_names: list[str], dataset_config: _DatasetConfig
) -> str | None:
    if dataset_config.messages_column is not None:
        _require_column(
            column_names,
            dataset_config.messages_column,
            dataset_config.path,
            "messages",
        )
        return dataset_config.messages_column
    if "messages" in column_names:
        return "messages"
    return None


def _resolve_prompt_completion_columns(
    column_names: list[str], dataset_config: _DatasetConfig
) -> tuple[str, str] | None:
    if dataset_config.prompt_column is None:
        return None
    _require_column(
        column_names, dataset_config.prompt_column, dataset_config.path, "prompt"
    )
    _require_column(
        column_names,
        dataset_config.completion_column,
        dataset_config.path,
        "completion",
    )
    return dataset_config.prompt_column, dataset_config.completion_column


def _require_column(
    column_names: list[str], column_name: str | None, path: str, column_kind: str
) -> None:
    if column_name is None or column_name not in column_names:
        raise ValueError(
            f"Dataset {path!r} is missing the configured {column_kind} column {column_name!r}. Available columns: {column_names}."
        )


def _validate_messages(
    messages: Any, path: str, column_name: str
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or any(
        not isinstance(message, dict)
        or "role" not in message
        or "content" not in message
        for message in messages
    ):
        raise ValueError(
            f"Dataset {path!r} column {column_name!r} must contain a list of chat-template-compatible message dicts with role/content keys."
        )
    return messages


def _messages_to_sft_example(
    messages, tokenizer, path: str
) -> tuple[list[int], list[int]]:
    if not messages:
        raise ValueError(f"Dataset {path!r} contains an empty messages list.")

    input_ids = _tokenize_messages(tokenizer, messages, add_generation_prompt=False)
    loss_mask = [0] * len(input_ids)
    assistant_found = False

    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_found = True
        prefix_messages = messages[:idx]
        prefix_ids = (
            _tokenize_messages(tokenizer, prefix_messages, add_generation_prompt=True)
            if prefix_messages
            else []
        )
        upto_ids = _tokenize_messages(
            tokenizer, messages[: idx + 1], add_generation_prompt=False
        )
        start = len(prefix_ids)
        end = len(upto_ids)
        if start > end or end > len(input_ids):
            raise ValueError(
                f"Failed to build assistant-only loss mask for dataset {path!r}."
            )
        for position in range(start, end):
            loss_mask[position] = 1

    if not assistant_found:
        raise ValueError(
            f"Dataset {path!r} messages-based SFT rows must contain at least one assistant turn."
        )

    return input_ids, loss_mask


def _build_suffix_mask(total_length: int, prefix_length: int) -> list[int]:
    if prefix_length > total_length:
        raise ValueError(
            f"Prompt prefix length ({prefix_length}) cannot exceed total length ({total_length})."
        )
    return [0] * prefix_length + [1] * (total_length - prefix_length)


def _tokenize_messages(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )
