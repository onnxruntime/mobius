# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest


def _evidence() -> dict:
    path = (
        Path(__file__).parents[1]
        / "testdata"
        / "evidence"
        / "gguf_draft_runtime_evidence.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _independent_trace(record: dict) -> dict:
    metadata = record["independent_direct_ort_trace"]
    path = Path(__file__).parents[1] / "testdata" / "evidence" / metadata["filename"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
    return json.loads(payload)


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("architecture", ["dflash", "eagle3"])
def test_real_gguf_draft_pair_direct_ort_acceptance_loop(
    tmp_path: Path,
    architecture: str,
) -> None:
    if os.environ.get("MOBIUS_RUN_GGUF_DRAFT_REAL") != "1":
        pytest.skip("set MOBIUS_RUN_GGUF_DRAFT_REAL=1 for the <=16 GiB real pair probe")

    import onnxruntime as ort
    from gguf_draft_oracle import IndependentDraftOracle, assert_trace_matches
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from mobius.integrations.gguf import (
        DraftPairRunner,
        build_draft_pair_from_gguf,
        write_draft_pair_package,
    )

    record = next(
        item for item in _evidence()["routes"] if item["architecture"] == architecture
    )
    artifacts = tmp_path / "artifacts"
    target_path = hf_hub_download(
        record["target"]["repository"],
        record["target"]["filename"],
        revision=record["target"]["revision"],
        local_dir=artifacts / "target",
    )
    draft_path = hf_hub_download(
        record["draft"]["repository"],
        record["draft"]["filename"],
        revision=record["draft"]["revision"],
        local_dir=artifacts / "draft",
    )
    target_config = artifacts / "target-config"
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(
            record["target_config"]["repository"],
            filename,
            revision=record["target_config"]["revision"],
            local_dir=target_config,
        )

    package = build_draft_pair_from_gguf(
        target_path,
        draft_path,
        target_config=target_config,
        execution_provider="cpu",
    )
    output = tmp_path / "package"
    write_draft_pair_package(package, output, progress_bar=False)

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(os.cpu_count() or 1, 8)
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    tokenizer = Tokenizer.from_file(str(target_config / "tokenizer.json"))
    prompt = "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include"
    input_ids = np.array([tokenizer.encode(prompt).ids], dtype=np.int64)
    oracle = IndependentDraftOracle(
        output,
        Path(draft_path),
        session_options=options,
    )
    trace = oracle.run(input_ids, 32)
    expected_trace = _independent_trace(record)
    assert_trace_matches(trace, expected_trace)
    assert trace["tokens_equal"] is True
    assert trace["counters"]["accepted"] > 0
    assert trace["counters"]["multi_token_rounds"] > 0
    assert trace["counters"]["rejections"] > 0
    for round_trace in trace["rounds"]:
        assert round_trace["target"]["replay_tokens_match"] is True
        assert round_trace["target"]["replay_max_abs"] >= 0
        assert round_trace["draft"]["replay_max_abs"] == 0

    runner = DraftPairRunner(output, session_options=options)
    production = runner.generate(input_ids, max_new_tokens=32)
    assert list(production.tokens) == trace["generated_tokens"]
    assert production.tokens == runner.generate_target_only(input_ids, max_new_tokens=32)
    assert production.stats.rounds == trace["counters"]["rounds"]
    assert production.stats.proposed_tokens == trace["counters"]["proposed"]
    assert production.stats.accepted_tokens == trace["counters"]["accepted"]
    assert production.stats.multi_token_rounds == trace["counters"]["multi_token_rounds"]
    assert len(production.stats.rollback_events) == trace["counters"]["rejections"]

    discriminators = record["independent_direct_ort_trace"]["mutation_discriminators"]
    for mutation, discriminator in discriminators.items():
        with pytest.raises(AssertionError, match=re.escape(discriminator)):
            IndependentDraftOracle(
                output,
                Path(draft_path),
                session_options=options,
                mutation=mutation,
            ).run(input_ids, 32)

    runner.close()
    del oracle, runner, package
    gc.collect()
