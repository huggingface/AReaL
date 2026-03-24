from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset

import areal.dataset.gsm8k as gsm8k_dataset
import areal.dataset.hf_text as hf_text_dataset
from areal.api.cli_args import TrainDatasetConfig
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


class DummyChatTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> str | list[int]:
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return [ord(char) for char in rendered]
        return rendered


def _patch_load_dataset(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    seen: dict[str, Any] | None = None,
) -> None:
    def fake_load_dataset(path: str, name: str | None, split: str) -> Dataset:
        if seen is not None:
            seen.update({"path": path, "name": name, "split": split})
        return dataset

    monkeypatch.setattr(hf_text_dataset, "load_dataset", fake_load_dataset)


def _tokenize_messages(
    tokenizer: DummyChatTokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


def _build_expected_loss_mask(
    tokenizer: DummyChatTokenizer, messages: list[dict[str, Any]]
) -> list[int]:
    input_ids = _tokenize_messages(tokenizer, messages, add_generation_prompt=False)
    loss_mask = [0] * len(input_ids)
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_messages = messages[:idx]
        prefix_ids = (
            _tokenize_messages(tokenizer, prefix_messages, add_generation_prompt=True)
            if prefix_messages
            else []
        )
        upto_ids = _tokenize_messages(
            tokenizer, messages[: idx + 1], add_generation_prompt=False
        )
        for position in range(len(prefix_ids), len(upto_ids)):
            loss_mask[position] = 1
    return loss_mask


def test_get_custom_dataset_builtin_adapter_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def fake_get_gsm8k_sft_dataset(*args: Any, **kwargs: Any) -> object:
        return sentinel

    monkeypatch.setattr(
        gsm8k_dataset,
        "get_gsm8k_sft_dataset",
        fake_get_gsm8k_sft_dataset,
    )

    dataset_config = TrainDatasetConfig(path="openai/gsm8k", type="sft")

    result = get_custom_dataset(
        split="train",
        dataset_config=dataset_config,
        tokenizer=DummyChatTokenizer(),
    )

    assert result is sentinel


def test_get_custom_dataset_explicit_generic_schema_overrides_builtin_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fail_get_gsm8k_rl_dataset(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("generic schema config should bypass the GSM8K adapter")

    monkeypatch.setattr(
        gsm8k_dataset,
        "get_gsm8k_rl_dataset",
        fail_get_gsm8k_rl_dataset,
    )

    dataset = Dataset.from_list(
        [{"prompt": "What is 2 + 2?", "completion": "4", "topic": "math"}]
    )
    _patch_load_dataset(monkeypatch, dataset, seen)

    dataset_config = TrainDatasetConfig(
        path="openai/gsm8k",
        type="rl",
        config_name="special-subset",
        prompt_column="prompt",
        completion_column="completion",
    )

    result = get_custom_dataset(split="train", dataset_config=dataset_config)

    assert seen == {
        "path": "openai/gsm8k",
        "name": "special-subset",
        "split": "train",
    }
    assert result[0]["messages"] == [{"role": "user", "content": "What is 2 + 2?"}]
    assert result[0]["answer"] == "4"
    assert result[0]["topic"] == "math"


def test_get_custom_dataset_generic_rl_messages_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    dataset = Dataset.from_list(
        [{"messages": messages, "answer": "4", "difficulty": "easy"}]
    )
    _patch_load_dataset(monkeypatch, dataset)

    dataset_config = TrainDatasetConfig(path="acme/math-chat", type="rl")

    result = get_custom_dataset(split="train", dataset_config=dataset_config)

    assert result[0]["messages"] == messages
    assert result[0]["answer"] == "4"
    assert result[0]["difficulty"] == "easy"


def test_get_custom_dataset_generic_rl_prompt_completion_normalizes_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_list(
        [{"prompt": "What is 2 + 2?", "completion": "4", "topic": "math"}]
    )
    _patch_load_dataset(monkeypatch, dataset)

    dataset_config = TrainDatasetConfig(
        path="acme/math-prompts",
        type="rl",
        prompt_column="prompt",
        completion_column="completion",
    )

    result = get_custom_dataset(split="train", dataset_config=dataset_config)

    assert result[0]["messages"] == [{"role": "user", "content": "What is 2 + 2?"}]
    assert result[0]["answer"] == "4"
    assert result[0]["topic"] == "math"


def test_get_custom_dataset_generic_sft_prompt_completion_masks_completion_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = DummyChatTokenizer()
    dataset = Dataset.from_list([{"prompt": "Question", "completion": "Answer"}])
    _patch_load_dataset(monkeypatch, dataset)

    dataset_config = TrainDatasetConfig(
        path="acme/sft-prompts",
        type="sft",
        prompt_column="prompt",
        completion_column="completion",
    )

    result = get_custom_dataset(
        split="train",
        dataset_config=dataset_config,
        tokenizer=tokenizer,
    )

    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    expected_input_ids = _tokenize_messages(
        tokenizer, messages, add_generation_prompt=False
    )
    prompt_ids = _tokenize_messages(tokenizer, messages[:1], add_generation_prompt=True)

    assert result[0]["input_ids"] == expected_input_ids
    assert result[0]["loss_mask"] == [0] * len(prompt_ids) + [1] * (
        len(expected_input_ids) - len(prompt_ids)
    )


def test_get_custom_dataset_generic_sft_messages_masks_assistant_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = DummyChatTokenizer()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Explain"},
        {"role": "assistant", "content": "Sure"},
    ]
    dataset = Dataset.from_list([{"messages": messages}])
    _patch_load_dataset(monkeypatch, dataset)

    dataset_config = TrainDatasetConfig(path="acme/chat-sft", type="sft")

    result = get_custom_dataset(
        split="train",
        dataset_config=dataset_config,
        tokenizer=tokenizer,
    )

    assert result[0]["input_ids"] == _tokenize_messages(
        tokenizer, messages, add_generation_prompt=False
    )
    assert result[0]["loss_mask"] == _build_expected_loss_mask(tokenizer, messages)


def test_get_custom_dataset_generic_loader_forwards_config_name_split_and_messages_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    chat_messages = [{"role": "user", "content": "Ping"}]
    dataset = Dataset.from_list([{"chat": chat_messages, "metadata": 7}])
    _patch_load_dataset(monkeypatch, dataset, seen)

    dataset_config = TrainDatasetConfig(
        path="acme/chat-subset",
        type="rl",
        config_name="subset-a",
        split="validation",
        messages_column="chat",
    )

    result = get_custom_dataset(split="train", dataset_config=dataset_config)

    assert seen == {
        "path": "acme/chat-subset",
        "name": "subset-a",
        "split": "validation",
    }
    assert result[0]["messages"] == chat_messages
    assert result[0]["metadata"] == 7


def test_get_custom_dataset_generic_loader_uses_configured_num_proc_for_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_num_proc: dict[str, list[int | None]] = {"map": [], "filter": []}
    tokenizer = DummyChatTokenizer()
    chat_messages = [{"role": "user", "content": "Ping"}]
    dataset = Dataset.from_list(
        [{"messages": chat_messages, "metadata": idx} for idx in range(30)]
    )
    _patch_load_dataset(monkeypatch, dataset)
    monkeypatch.setattr(hf_text_dataset.os, "cpu_count", lambda: 64)

    original_map = Dataset.map
    original_filter = Dataset.filter

    def fake_map(self, function, *args: Any, **kwargs: Any):
        seen_num_proc["map"].append(kwargs.get("num_proc"))
        kwargs["num_proc"] = 1
        return original_map(self, function=function, *args, **kwargs)

    def fake_filter(self, function, *args: Any, **kwargs: Any):
        seen_num_proc["filter"].append(kwargs.get("num_proc"))
        kwargs["num_proc"] = 1
        return original_filter(self, function=function, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", fake_map)
    monkeypatch.setattr(Dataset, "filter", fake_filter)

    dataset_config = TrainDatasetConfig(
        path="acme/chat-num-proc",
        type="rl",
        max_length=1000,
        num_proc=7,
    )

    result = get_custom_dataset(
        split="train",
        dataset_config=dataset_config,
        tokenizer=tokenizer,
    )

    assert seen_num_proc["map"][0] == 7
    assert seen_num_proc["filter"] == [7]
    assert result[0]["messages"] == chat_messages


@pytest.mark.slow
def test_get_custom_dataset_generic_loader_with_real_hf_assets() -> None:
    model_path = "Qwen/Qwen3-0.6B"
    dataset_path = "lm-provers/FineProofs-SFT"

    tokenizer = load_hf_tokenizer(model_path)
    dataset_config = TrainDatasetConfig(
        path=dataset_path,
        type="sft",
        config_name="default",
        split="train[:2]",
        messages_column="messages",
        batch_size=1,
        max_length=4096,
    )

    dataset = get_custom_dataset(
        split="train[:2]",
        dataset_config=dataset_config,
        tokenizer=tokenizer,
    )

    assert len(dataset) > 0
    row = dataset[0]
    assert sorted(row.keys()) == ["input_ids", "loss_mask"]
    assert len(row["input_ids"]) == len(row["loss_mask"])
    assert sum(row["loss_mask"]) > 0
    assert len(row["input_ids"]) <= 4096


def test_train_dataset_config_requires_prompt_and_completion_columns() -> None:
    with pytest.raises(
        ValueError,
        match="prompt_column and completion_column must be provided together",
    ):
        TrainDatasetConfig(path="acme/bad-config", type="rl", prompt_column="prompt")


def test_train_dataset_config_rejects_mixed_messages_and_prompt_columns() -> None:
    with pytest.raises(
        ValueError,
        match="messages_column cannot be combined with prompt_column/completion_column",
    ):
        TrainDatasetConfig(
            path="acme/bad-config",
            type="rl",
            messages_column="messages",
            prompt_column="prompt",
            completion_column="completion",
        )


def test_train_dataset_config_requires_positive_num_proc() -> None:
    with pytest.raises(
        ValueError,
        match="num_proc must be a positive integer or None",
    ):
        TrainDatasetConfig(path="acme/bad-config", type="rl", num_proc=0)


def test_get_custom_dataset_generic_loader_requires_supported_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_list([{"chat": [{"role": "user", "content": "Hello"}]}])
    _patch_load_dataset(monkeypatch, dataset)

    dataset_config = TrainDatasetConfig(path="acme/unsupported-schema", type="rl")

    with pytest.raises(
        ValueError,
        match="generic RL loading requires either a 'messages' column or configured prompt_column/completion_column fields",
    ):
        get_custom_dataset(split="train", dataset_config=dataset_config)


def test_get_custom_dataset_generic_loader_requires_configured_columns_to_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = DummyChatTokenizer()
    dataset = Dataset.from_list([{"prompt": "Question"}])
    _patch_load_dataset(monkeypatch, dataset)

    dataset_config = TrainDatasetConfig(
        path="acme/missing-columns",
        type="sft",
        prompt_column="prompt",
        completion_column="completion",
    )

    with pytest.raises(ValueError, match="configured completion column"):
        get_custom_dataset(
            split="train",
            dataset_config=dataset_config,
            tokenizer=tokenizer,
        )
