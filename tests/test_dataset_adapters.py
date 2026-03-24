from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from PIL import Image

import areal.dataset.clevr_count_70k as clevr_count_70k_dataset
import areal.dataset.geometry3k as geometry3k_dataset
from areal.api.cli_args import TrainDatasetConfig
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer

TEXT_MODEL_ID = "Qwen/Qwen3-0.6B"
VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TORL_DATA_PATH = "/tmp/areal/torl_data/test.parquet"


def _assert_text_sft_row(row: dict[str, Any]) -> None:
    assert sorted(row.keys()) == ["input_ids", "loss_mask"]
    assert len(row["input_ids"]) == len(row["loss_mask"])
    assert sum(row["loss_mask"]) > 0


def _assert_text_rl_row(row: dict[str, Any]) -> None:
    assert "answer" in row
    assert isinstance(row["messages"], list)
    assert row["messages"]


def _assert_reward_model_row(row: dict[str, Any]) -> None:
    assert sorted(row.keys()) == ["chosen_ids", "rejected_ids"]
    assert row["chosen_ids"]
    assert row["rejected_ids"]


def _assert_multimodal_sft_row(row: dict[str, Any]) -> None:
    assert {"input_ids", "loss_mask", "multi_modal_input"} <= set(row)
    assert len(row["input_ids"]) == len(row["loss_mask"])
    assert sum(row["loss_mask"]) > 0
    assert row["multi_modal_input"]
    assert "pixel_values" in row["multi_modal_input"][0]


def _assert_multimodal_rl_row(
    row: dict[str, Any], *, expect_messages_chat: bool = False
) -> None:
    assert "answer" in row
    assert isinstance(row["messages"], str)
    assert row["messages"]
    assert row["images"]
    if expect_messages_chat:
        assert row["messages_chat"]


@pytest.fixture(scope="module")
def text_tokenizer() -> Any:
    return load_hf_tokenizer(TEXT_MODEL_ID)


@pytest.fixture(scope="module")
def vision_processor() -> Any:
    processor, _ = load_hf_processor_and_tokenizer(VISION_MODEL_ID)
    assert processor is not None
    return processor


def _make_multimodal_fixture_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "images": [Image.new("RGB", (64, 64), color="white")],
                "problem": "<image> Count the objects.",
                "answer": "1",
            }
        ]
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("dataset_config", "validator"),
    [
        pytest.param(
            TrainDatasetConfig(
                path="openai/gsm8k",
                type="sft",
                split="train[:1]",
                max_length=4096,
            ),
            _assert_text_sft_row,
            id="gsm8k-sft",
        ),
        pytest.param(
            TrainDatasetConfig(
                path="openai/gsm8k",
                type="rl",
                split="train[:1]",
                max_length=4096,
            ),
            _assert_text_rl_row,
            id="gsm8k-rl",
        ),
        pytest.param(
            TrainDatasetConfig(
                path="Anthropic/hh-rlhf",
                type="rw",
                split="train[:1]",
                max_length=4096,
            ),
            _assert_reward_model_row,
            id="hh-rlhf-rw",
        ),
        pytest.param(
            TrainDatasetConfig(
                path=TORL_DATA_PATH,
                type="rl",
                split="train",
                max_length=4096,
            ),
            _assert_text_rl_row,
            id="torl-data-rl",
        ),
    ],
)
def test_builtin_text_dataset_adapters_load(
    text_tokenizer: Any,
    dataset_config: TrainDatasetConfig,
    validator,
) -> None:
    dataset = get_custom_dataset(
        split=dataset_config.split,
        dataset_config=dataset_config,
        tokenizer=text_tokenizer,
    )

    assert len(dataset) > 0
    validator(dataset[0])


@pytest.mark.slow
@pytest.mark.parametrize(
    ("dataset_module", "dataset_config", "validator"),
    [
        pytest.param(
            clevr_count_70k_dataset,
            TrainDatasetConfig(
                path="synthetic/clevr_count_70k",
                type="sft",
                split="train",
                max_length=4096,
            ),
            _assert_multimodal_sft_row,
            id="clevr-count-70k-sft",
        ),
        pytest.param(
            clevr_count_70k_dataset,
            TrainDatasetConfig(
                path="synthetic/clevr_count_70k",
                type="rl",
                split="train",
                max_length=4096,
            ),
            lambda row: _assert_multimodal_rl_row(row),
            id="clevr-count-70k-rl",
        ),
        pytest.param(
            geometry3k_dataset,
            TrainDatasetConfig(
                path="synthetic/geometry3k",
                type="sft",
                split="train",
                max_length=4096,
            ),
            _assert_multimodal_sft_row,
            id="geometry3k-sft",
        ),
        pytest.param(
            geometry3k_dataset,
            TrainDatasetConfig(
                path="synthetic/geometry3k",
                type="rl",
                split="train",
                max_length=4096,
            ),
            lambda row: _assert_multimodal_rl_row(row, expect_messages_chat=True),
            id="geometry3k-rl",
        ),
    ],
)
def test_builtin_multimodal_dataset_adapters_load(
    monkeypatch: pytest.MonkeyPatch,
    vision_processor: Any,
    dataset_module,
    dataset_config: TrainDatasetConfig,
    validator,
) -> None:
    monkeypatch.setattr(
        dataset_module,
        "load_dataset",
        lambda path, split: _make_multimodal_fixture_dataset(),
    )
    if dataset_module is clevr_count_70k_dataset:
        monkeypatch.setattr(dataset_module.os, "cpu_count", lambda: 1)

    dataset = get_custom_dataset(
        split=dataset_config.split,
        dataset_config=dataset_config,
        processor=vision_processor,
    )

    assert len(dataset) == 1
    validator(dataset[0])


@pytest.mark.slow
def test_virl39k_rl_dataset_adapter_loads_local_fixture(
    tmp_path: Path,
    vision_processor: Any,
) -> None:
    pytest.importorskip("mathruler.grader")

    root = tmp_path / "virl39k_smoke"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)

    image_path = images_dir / "sample.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    parquet_path = root / "virl39k_sample.parquet"
    Dataset.from_list(
        [
            {
                "question": "Count the objects.",
                "answer": "\\boxed{1}",
                "image": [image_path.name],
                "PassRate_32BTrained": 0.0,
                "PassRate_7BBase": 0.0,
                "category": "counting",
                "source": "synthetic",
                "qid": "q1",
            }
        ]
    ).to_parquet(str(parquet_path))

    dataset_config = TrainDatasetConfig(
        path=str(parquet_path),
        type="rl",
        split="train",
        max_length=4096,
    )
    dataset = get_custom_dataset(
        split=dataset_config.split,
        dataset_config=dataset_config,
        processor=vision_processor,
    )

    assert len(dataset) == 1
    row = dataset[0]
    _assert_multimodal_rl_row(row, expect_messages_chat=True)
    assert row["answer"] == "1"
