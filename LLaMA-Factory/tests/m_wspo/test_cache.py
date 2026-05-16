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

r"""Tests for the m-WSPO weak-logprob cache loader.

These tests cover the round-trip between:
- the parquet schema written by ``scripts/m_wspo/cache_weak_logprobs.py``, and
- the lookup helpers on ``MWSPOTrainer``.

We don't instantiate the trainer (which requires GPU + tokenizers), so the
tests use the bound staticmethod / unbound method via ``__func__`` /
``__get__`` indirection where needed. Skipped if ``pandas`` is not
available locally.
"""

from __future__ import annotations

import os

import pytest

pd = pytest.importorskip("pandas")

import torch  # noqa: E402  (after importorskip)

from llamafactory.train.m_wspo.trainer import _CACHE_COLUMNS, MWSPOTrainer  # noqa: E402


def _write_cache(tmp_path, num_rows: int) -> str:
    path = os.path.join(tmp_path, "weak_lp.parquet")
    rows = []
    for sid in range(num_rows):
        row = {"sample_id": sid}
        # Use a unique value per (sample_id, column) so we can verify ordering.
        for j, col in enumerate(_CACHE_COLUMNS):
            row[col] = float(sid * 100 + j)
        rows.append(row)

    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_load_weak_cache_round_trip(tmp_path):
    path = _write_cache(tmp_path, num_rows=8)
    cache = MWSPOTrainer._load_weak_cache(path)

    assert set(cache.keys()) == set(range(8))
    for sid, row in cache.items():
        for j, col in enumerate(_CACHE_COLUMNS):
            assert row[col] == float(sid * 100 + j), (sid, col, row[col])


def test_load_weak_cache_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Run `python scripts/m_wspo"):
        MWSPOTrainer._load_weak_cache("/does/not/exist.parquet")


def test_load_weak_cache_missing_columns_raises(tmp_path):
    path = os.path.join(tmp_path, "broken.parquet")
    pd.DataFrame([{"sample_id": 0, "weak_aligned_lp_yw_mw": 1.0}]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        MWSPOTrainer._load_weak_cache(path)


class _StubTrainer:
    r"""Just enough surface for ``_lookup_weak_cache`` to run without any of
    the LLaMA-Factory training scaffolding.
    """

    def __init__(self, cache):
        self._weak_cache = cache

    _lookup_weak_cache = MWSPOTrainer._lookup_weak_cache


def test_lookup_returns_tensors_in_sample_id_order(tmp_path):
    path = _write_cache(tmp_path, num_rows=4)
    stub = _StubTrainer(MWSPOTrainer._load_weak_cache(path))

    sample_ids = [3, 0, 2]  # deliberately out of order
    out = stub._lookup_weak_cache(sample_ids, device=torch.device("cpu"))

    assert out is not None
    assert all(t.shape == (3,) for t in out.values())
    assert out["weak_aligned_lp_yw_mw"].tolist() == [3 * 100 + 0, 0 * 100 + 0, 2 * 100 + 0]
    assert out["weak_ref_lp_yw_mw"].tolist() == [3 * 100 + 1, 0 * 100 + 1, 2 * 100 + 1]


def test_lookup_returns_none_on_partial_miss(tmp_path):
    path = _write_cache(tmp_path, num_rows=2)
    stub = _StubTrainer(MWSPOTrainer._load_weak_cache(path))

    out = stub._lookup_weak_cache([0, 99], device=torch.device("cpu"))
    assert out is None


def test_lookup_returns_none_when_cache_unset():
    stub = _StubTrainer(cache=None)
    out = stub._lookup_weak_cache([0, 1], device=torch.device("cpu"))
    assert out is None
