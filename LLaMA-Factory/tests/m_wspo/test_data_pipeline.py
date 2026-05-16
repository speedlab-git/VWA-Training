# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Plumbing tests for the m-WSPO data path.

These guard the wiring that's easy to break later (the column flowing all
the way from `data/dataset_info.json` into the per-step batch):

- `DatasetAttr.join` should pick up `images_neg` from `columns: {...}`.
- All three `DatasetConverter`s should produce `_images_neg` in their
  output dict (None when not configured, list when configured).
- `PairwiseDatasetProcessor.preprocess_dataset` should propagate
  `_images_neg` and `_sample_id` into the model_inputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llamafactory.data.parser import DatasetAttr


def _attr() -> DatasetAttr:
    return DatasetAttr(load_from="file", dataset_name="dummy", formatting="sharegpt")


def test_dataset_attr_join_picks_up_images_neg():
    attr = _attr()
    attr.join({"columns": {"images": "img", "images_neg": "img_neg"}})
    assert attr.images == "img"
    assert attr.images_neg == "img_neg"


def test_dataset_attr_join_leaves_images_neg_none_when_absent():
    attr = _attr()
    attr.join({"columns": {"images": "img"}})
    assert attr.images_neg is None


# --- converter ---


def test_alpaca_converter_emits_images_neg(monkeypatch):
    from llamafactory.data.converter import AlpacaDatasetConverter

    attr = DatasetAttr(load_from="file", dataset_name="d", formatting="alpaca")
    attr.images = "image"
    attr.images_neg = "image_neg"
    attr.prompt = "instruction"
    attr.query = "input"
    attr.response = "output"

    data_args = MagicMock(media_dir="/tmp")
    converter = AlpacaDatasetConverter(dataset_attr=attr, data_args=data_args)
    monkeypatch.setattr(converter, "_find_medias", lambda x: x)

    out = converter({"instruction": "hi", "input": "", "output": "yo", "image": ["a.jpg"], "image_neg": ["b.jpg"]})

    assert out["_images"] == ["a.jpg"]
    assert out["_images_neg"] == ["b.jpg"]


def test_alpaca_converter_images_neg_none_when_unconfigured(monkeypatch):
    from llamafactory.data.converter import AlpacaDatasetConverter

    attr = DatasetAttr(load_from="file", dataset_name="d", formatting="alpaca")
    attr.images = "image"
    attr.images_neg = None  # explicit
    attr.prompt = "instruction"
    attr.query = "input"
    attr.response = "output"

    data_args = MagicMock(media_dir="/tmp")
    converter = AlpacaDatasetConverter(dataset_attr=attr, data_args=data_args)
    monkeypatch.setattr(converter, "_find_medias", lambda x: x)

    out = converter({"instruction": "hi", "input": "", "output": "yo", "image": ["a.jpg"]})
    assert out["_images_neg"] is None


# --- pairwise processor ---


@pytest.fixture
def _stub_processor():
    """Build a `PairwiseDatasetProcessor` with `_encode_data_example`
    monkey-patched to a deterministic stub. Avoids needing a tokenizer."""
    from llamafactory.data.processor.pairwise import PairwiseDatasetProcessor

    proc = PairwiseDatasetProcessor.__new__(PairwiseDatasetProcessor)

    def _stub_encode(self, prompt, response, system, tools, images, videos, audios):
        return [1, 2], [-100, 2], [1, 3], [-100, 3]

    proc._encode_data_example = _stub_encode.__get__(proc)
    return proc


def test_pairwise_processor_propagates_images_neg_and_sample_id(_stub_processor):
    examples = {
        "_prompt": [[{"role": "user", "content": "hi"}]],
        "_response": [[{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]],
        "_system": [""],
        "_tools": [""],
        "_images": [["pos.jpg"]],
        "_videos": [None],
        "_audios": [None],
        "_images_neg": [["neg.jpg"]],
        "_sample_id": [42],
    }
    out = _stub_processor.preprocess_dataset(examples)
    assert out["images_neg"] == [["neg.jpg"]]
    assert out["sample_id"] == [42]


def test_pairwise_processor_works_without_optional_columns(_stub_processor):
    examples = {
        "_prompt": [[{"role": "user", "content": "hi"}]],
        "_response": [[{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]],
        "_system": [""],
        "_tools": [""],
        "_images": [["pos.jpg"]],
        "_videos": [None],
        "_audios": [None],
    }
    out = _stub_processor.preprocess_dataset(examples)
    assert "images_neg" not in out
    assert "sample_id" not in out
