# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic GGUF coverage for target-coupled speculative drafts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
_I64_DTYPE = np.dtype(np.int64)


def _tokenizer(tokens: list[str], *, merges: list[str] | None = None) -> dict:
    return {
        "version": "1.0",
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {
            "type": "BPE",
            "vocab": {token: index for index, token in enumerate(tokens)},
            "merges": merges or [],
        },
        "added_tokens": [],
    }


def _target(tokens: list[str], **overrides):
    target = {
        "target_model_id": "example/Qwen3-target@0123456789abcdef",
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": len(tokens),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tokenizer_json": _tokenizer(tokens),
    }
    target.update(overrides)
    return target


def _write_target_dir(
    path: Path,
    tokens: list[str],
    *,
    config_overrides: dict | None = None,
    tokenizer: dict | None = None,
) -> tuple[dict, dict]:
    path.mkdir(parents=True)
    target = _target(tokens)
    default_tokenizer = target.pop("tokenizer_json")
    tokenizer_json = default_tokenizer if tokenizer is None else tokenizer
    if config_overrides:
        target.update(config_overrides)
    (path / "config.json").write_text(
        json.dumps(target, indent=2),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, indent=2),
        encoding="utf-8",
    )
    return target, tokenizer_json


def _directory_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def _write_draft(
    path: Path,
    architecture: str,
    *,
    quantized: bool = False,
    d2t: np.ndarray | None = None,
    extra_tensor: str | None = None,
    own_embedding: bool = False,
    target_hidden_size: int | None = None,
    target_layers: list[int] | None = None,
    d2t_dtype: np.dtype = _I64_DTYPE,
    tokens: list[str] | None = None,
) -> Path:
    hidden, intermediate, heads, kv_heads, layers = 64, 128, 4, 2, 1
    tokens = tokens or [f"token-{index}" for index in range(64)]
    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_context_length(128)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(layers)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_rope_dimension_count(16)
    writer.add_rope_freq_base(1_000_000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_token_list(tokens)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)
    writer.add_array(f"{architecture}.target_layers", target_layers or [1, 3, 5])
    target_hidden = target_hidden_size or hidden
    if architecture == "dflash":
        writer.add_uint32("dflash.block_size", 4)
    else:
        writer.add_uint32("eagle3.target_hidden_size", target_hidden)
        writer.add_bool("eagle3.norm_before_residual", False)
        writer.add_bool("eagle3.norm_before_fc", False)

    rng = np.random.default_rng(42)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, rng.standard_normal(shape, dtype=np.float32))

    def add_matrix(name: str, shape: tuple[int, int]) -> None:
        if not quantized:
            add_float(name, shape)
            return
        n_out, k_in = shape
        assert k_in % 32 == 0
        raw = np.empty((n_out, k_in // 32 * 18), dtype=np.uint8)
        for row in range(n_out):
            for block in range(k_in // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array([0.01], dtype=np.float16).view(
                    np.uint8
                )
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.Q4_0)

    add_matrix("fc.weight", (hidden, 3 * target_hidden))
    add_float("output_norm.weight", (hidden,))
    if architecture == "dflash":
        add_float("enc.output_norm.weight", (hidden,))
    if d2t is not None:
        writer.add_tensor("d2t", d2t.astype(d2t_dtype))
        add_matrix("output.weight", (len(d2t), hidden))
    if own_embedding:
        add_matrix("token_embd.weight", (len(tokens), hidden))

    layer_shapes = {
        "attn_q.weight": (heads * 16, hidden if architecture == "dflash" else 2 * hidden),
        "attn_k.weight": (kv_heads * 16, hidden if architecture == "dflash" else 2 * hidden),
        "attn_v.weight": (kv_heads * 16, hidden if architecture == "dflash" else 2 * hidden),
        "attn_output.weight": (hidden, heads * 16),
        "attn_norm.weight": (hidden,),
        "ffn_norm.weight": (hidden,),
        "ffn_gate.weight": (intermediate, hidden),
        "ffn_up.weight": (intermediate, hidden),
        "ffn_down.weight": (hidden, intermediate),
    }
    if architecture == "dflash":
        layer_shapes["attn_q_norm.weight"] = (16,)
        layer_shapes["attn_k_norm.weight"] = (16,)
    else:
        layer_shapes["attn_norm_2.weight"] = (hidden,)
    for suffix, shape in layer_shapes.items():
        name = f"blk.0.{suffix}"
        if len(shape) == 2:
            add_matrix(name, shape)
        else:
            add_float(name, shape)
    if extra_tensor is not None:
        add_float(extra_tensor, (1,) if extra_tensor.endswith(".scale") else (hidden,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.mark.parametrize("architecture", ["dflash", "eagle3"])
def test_target_config_is_required_before_graph_build(
    tmp_path: Path, architecture: str, monkeypatch
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(tmp_path / f"{architecture}.gguf", architecture)
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="not a standalone language model"):
        build_from_gguf(path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_size": 32}, "hidden size"),
        ({"num_hidden_layers": 4}, "outside target layer count"),
        ({"vocab_size": 63}, "tokenizer size mismatch"),
    ],
)
def test_target_mismatches_fail_before_graph_build(
    tmp_path: Path, monkeypatch, override: dict, message: str
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(tmp_path / "dflash.gguf", "dflash")
    tokens = [f"token-{index}" for index in range(64)]
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match=message):
        build_from_gguf(path, target_config=_target(tokens, **override))


def test_target_resources_reject_symlink_escape(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    external = tmp_path / "external"
    _write_target_dir(external, tokens)
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").symlink_to(external / "config.json")
    (target / "tokenizer.json").write_bytes((external / "tokenizer.json").read_bytes())
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match="could not be opened safely"):
        build_from_gguf(draft, target_config=target)


def test_target_resources_use_safe_path_open_without_dir_fd(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius.integrations.gguf import _draft, build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    target = tmp_path / "target"
    _write_target_dir(target, tokens)
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(_draft, "_supports_secure_dir_fd", lambda: False)

    manifest = build_from_gguf(draft, target_config=target).draft_manifest["target"]

    assert manifest["tokenizer_tokens_sha256"]


def test_safe_path_open_rejects_symlink_without_dir_fd(tmp_path: Path, monkeypatch) -> None:
    from mobius.integrations.gguf import _draft, build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    external = tmp_path / "external"
    _write_target_dir(external, tokens)
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").symlink_to(external / "config.json")
    (target / "tokenizer.json").write_bytes((external / "tokenizer.json").read_bytes())
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(_draft, "_supports_secure_dir_fd", lambda: False)

    with pytest.raises(ValueError, match="could not be opened safely"):
        build_from_gguf(draft, target_config=target)


def test_safe_path_open_rejects_root_swap_between_resources(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius.integrations.gguf import _draft, build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    target = tmp_path / "target"
    _write_target_dir(target, tokens)
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(_draft, "_supports_secure_dir_fd", lambda: False)
    original_read = _draft._read_bounded_json_at

    def swap_root_after_config(*args, **kwargs):
        value = original_read(*args, **kwargs)
        if args[3] == "config.json":
            target.rename(tmp_path / "displaced-target")
            _write_target_dir(target, list(reversed(tokens)))
        return value

    monkeypatch.setattr(_draft, "_read_bounded_json_at", swap_root_after_config)

    with pytest.raises(ValueError, match="target root changed"):
        build_from_gguf(draft, target_config=target)


def test_target_resources_do_not_mix_split_tokenizer_files(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    target = tmp_path / "target"
    config, _ = _write_target_dir(target, tokens)
    (target / "tokenizer.json").unlink()
    (target / "vocab.json").write_text("{}", encoding="utf-8")
    (target / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match="split tokenizer files cannot preserve"):
        build_from_gguf(draft, target_config=target)
    assert config["vocab_size"] == len(tokens)


def test_target_manifest_hashes_full_tokenizer_semantics(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    tokens = ["a", "b", "ab", *[f"token-{index}" for index in range(61)]]
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash", tokens=tokens)
    first = _target(tokens)
    second = _target(tokens)
    second["tokenizer_json"]["model"]["merges"] = ["a b"]

    first_target = build_from_gguf(draft, target_config=first).draft_manifest["target"]
    second_target = build_from_gguf(draft, target_config=second).draft_manifest["target"]

    assert first_target["config_sha256"] == second_target["config_sha256"]
    assert first_target["tokenizer_tokens_sha256"] == second_target["tokenizer_tokens_sha256"]
    assert first_target["tokenizer_sha256"] != second_target["tokenizer_sha256"]


def test_target_tokenizer_accepts_only_exact_trailing_gguf_padding(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    real_tokens = [f"token-{index}" for index in range(60)]
    padded_tokens = [*real_tokens, *[f"[PAD{index}]" for index in range(60, 64)]]
    target = _target(padded_tokens, tokenizer_json=_tokenizer(real_tokens))
    exact = _write_draft(
        tmp_path / "dflash-padded.gguf",
        "dflash",
        tokens=padded_tokens,
    )

    manifest = build_from_gguf(exact, target_config=target).draft_manifest

    assert manifest["target"]["vocab_size"] == 64
    assert manifest["target"]["tokenizer_tokens_sha256"]

    mismatched_tokens = [*padded_tokens[:-1], "not-the-upstream-padding-token"]
    mismatched = _write_draft(
        tmp_path / "dflash-mismatched-padding.gguf",
        "dflash",
        tokens=mismatched_tokens,
    )
    with pytest.raises(ValueError, match="tokenizer is not identical"):
        build_from_gguf(mismatched, target_config=target)


def test_target_tokenizer_rejects_unmapped_interior_ids(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    tokenizer = _tokenizer(tokens)
    del tokenizer["model"]["vocab"]["token-7"]
    target = _target(tokens, tokenizer_json=tokenizer)
    draft = _write_draft(tmp_path / "dflash-interior-gap.gguf", "dflash", tokens=tokens)

    with pytest.raises(ValueError, match="unmapped non-trailing ids"):
        build_from_gguf(draft, target_config=target)


def test_target_manifest_is_stable_across_relocation_and_mapping(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    first_config, tokenizer = _write_target_dir(tmp_path / "first" / "target", tokens)
    _write_target_dir(
        tmp_path / "relocated" / "target",
        tokens,
        tokenizer=tokenizer,
    )
    mapping = {**first_config, "tokenizer_json": tokenizer}

    path_target = build_from_gguf(
        draft,
        target_config=tmp_path / "first" / "target",
    ).draft_manifest["target"]
    relocated_target = build_from_gguf(
        draft,
        target_config=tmp_path / "relocated" / "target",
    ).draft_manifest["target"]
    mapping_target = build_from_gguf(
        draft,
        target_config=mapping,
    ).draft_manifest["target"]

    assert path_target == relocated_target == mapping_target
    assert set(path_target).issuperset(
        {"config_sha256", "tokenizer_sha256", "tokenizer_tokens_sha256"}
    )
    assert not any(str(tmp_path) in str(value) for value in path_target.values())


@pytest.mark.parametrize(
    ("resource", "payload", "message"),
    [
        ("config.json", b"{", "not valid UTF-8 JSON"),
        ("config.json", b"[]", "root must be a JSON object"),
        ("tokenizer.json", b"{", "not valid UTF-8 JSON"),
    ],
)
def test_target_json_schema_errors_fail_before_graph(
    tmp_path: Path,
    monkeypatch,
    resource: str,
    payload: bytes,
    message: str,
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    target = tmp_path / "target"
    _write_target_dir(target, tokens)
    (target / resource).write_bytes(payload)
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match=message):
        build_from_gguf(draft, target_config=target)


def test_target_config_file_size_is_bounded(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf
    from mobius.integrations.gguf._draft import _MAX_CONFIG_JSON_BYTES

    tokens = [f"token-{index}" for index in range(64)]
    target = tmp_path / "target"
    _write_target_dir(target, tokens)
    (target / "config.json").write_bytes(b" " * (_MAX_CONFIG_JSON_BYTES + 1))
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match="file-size limit"):
        build_from_gguf(draft, target_config=target)


def test_sparse_tokenizer_ids_are_bounded_before_allocation(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    target = _target(tokens)
    target["tokenizer_json"]["model"]["vocab"]["sparse"] = 10**12
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match="tokenizer size mismatch"):
        build_from_gguf(draft, target_config=target)


def test_sparse_added_token_ids_are_bounded_before_allocation(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    target = _target(tokens)
    target["tokenizer_json"]["added_tokens"] = [
        {"id": 10**12, "content": "sparse", "special": False}
    ]
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match="tokenizer size mismatch"):
        build_from_gguf(draft, target_config=target)


def test_tokenizer_runtime_schema_is_validated_before_graph_build(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    tokens = [f"token-{index}" for index in range(64)]
    target = _target(tokens)
    target["tokenizer_json"]["normalizer"] = {"type": "not-a-normalizer"}
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match=r"tokenizer\.json has an invalid schema"):
        build_from_gguf(draft, target_config=target)


def test_tokenizer_identity_and_remap_are_fail_closed(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "eagle3.gguf",
        "eagle3",
        d2t=np.array([1, 1, 4], dtype=np.int64),
    )
    tokens = [f"token-{index}" for index in range(64)]
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="duplicate target ids"):
        build_from_gguf(path, target_config=_target(tokens))

    plain = _write_draft(tmp_path / "plain.gguf", "eagle3")
    tokens[9] = "wrong-token"
    with pytest.raises(ValueError, match="first mismatch at id 9"):
        build_from_gguf(plain, target_config=_target(tokens))


def test_d2t_requires_pinned_i64_storage(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "eagle3-i32-remap.gguf",
        "eagle3",
        d2t=np.array([1, 4, 9]),
        d2t_dtype=np.dtype(np.int32),
    )
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="d2t must use I64"):
        build_from_gguf(path, target_config=_target([f"token-{i}" for i in range(64)]))


def test_suffix_closure_rejects_unknown_tensor_before_graph(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "dflash-extra.gguf",
        "dflash",
        extra_tensor="blk.0.attn_magic.weight",
    )
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match=r"pinned llama\.cpp tensor creation sites"):
        build_from_gguf(path, target_config=_target([f"token-{i}" for i in range(64)]))


def test_draft_owned_embedding_is_consumed_by_eagle3_graph(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "eagle3-own-embedding.gguf",
        "eagle3",
        own_embedding=True,
    )
    package = build_from_gguf(
        path,
        target_config=_target([f"token-{i}" for i in range(64)]),
    )

    inputs = {value.name for value in package["model"].graph.inputs}
    initializers = set(package["model"].graph.initializers)
    assert "input_ids" in inputs
    assert "inputs_embeds" not in inputs
    assert "embed_tokens.weight" in initializers
    assert package.draft_manifest["orchestration"]["embedding_source"] == "draft"


def test_dflash_draft_owned_embedding_fails_closed(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "dflash-own-embedding.gguf",
        "dflash",
        own_embedding=True,
    )

    with pytest.raises(ValueError, match="consumes target-provided noise_embedding"):
        build_from_gguf(
            path,
            target_config=_target([f"token-{index}" for index in range(64)]),
        )


def test_eagle3_cross_width_target_sharing_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "eagle3-cross-width.gguf",
        "eagle3",
        target_hidden_size=32,
    )
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="target-shared embedding/head requires"):
        build_from_gguf(
            path,
            target_config=_target(
                [f"token-{i}" for i in range(64)],
                hidden_size=32,
            ),
        )


def test_eagle3_cross_width_owned_embedding_and_head_builds(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "eagle3-cross-width-owned.gguf",
        "eagle3",
        target_hidden_size=32,
        own_embedding=True,
        d2t=np.arange(8, dtype=np.int64),
    )

    package = build_from_gguf(
        path,
        target_config=_target(
            [f"token-{index}" for index in range(64)],
            hidden_size=32,
        ),
    )

    assert package.draft_manifest["target"]["hidden_size"] == 32
    assert package.draft_manifest["draft"]["hidden_size"] == 64
    assert package.draft_manifest["orchestration"]["embedding_source"] == "draft"
    assert package.draft_manifest["orchestration"]["lm_head_source"] == "draft"


def test_eagle3_rope_frequency_shape_is_exact(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "eagle3-rope.gguf",
        "eagle3",
        extra_tensor="blk.0.rope_freqs.weight",
    )
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="invalid tensor shape"):
        build_from_gguf(path, target_config=_target([f"token-{i}" for i in range(64)]))


def test_dflash_sliding_window_is_rejected_before_graph(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf
    from mobius.integrations.gguf._reader import GGUFModel

    path = _write_draft(tmp_path / "dflash-sliding.gguf", "dflash")
    model = GGUFModel(path)
    model.metadata["dflash.attention.sliding_window"] = 1024
    monkeypatch.setattr(
        "mobius.integrations.gguf._shard_set.open_gguf_model",
        lambda _path: model,
    )
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="sliding-window metadata is unsupported"):
        build_from_gguf(path, target_config=_target([f"token-{i}" for i in range(64)]))


def test_dflash_layer_input_indices_are_normalized(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "dflash-final-target-layer.gguf",
        "dflash",
        target_layers=[1, 4, 8],
    )
    package = build_from_gguf(
        path,
        target_config=_target([f"token-{i}" for i in range(64)]),
    )
    assert package.draft_manifest["target"]["target_layers"] == [0, 3, 7]


def test_eagle3_hidden_state_indices_are_normalized(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "eagle3-hidden-state-indices.gguf",
        "eagle3",
        target_layers=[2, 4, 8],
    )
    package = build_from_gguf(
        path,
        target_config=_target([f"token-{index}" for index in range(64)]),
    )

    assert package.draft_manifest["target"]["target_layers"] == [1, 3, 7]


def test_dflash_zero_layer_input_index_is_rejected_before_graph(
    tmp_path: Path, monkeypatch
) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "dflash-zero-target-layer.gguf",
        "dflash",
        target_layers=[0, 3, 5],
    )
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="outside target layer count"):
        build_from_gguf(path, target_config=_target([f"token-{i}" for i in range(64)]))


def test_eagle3_reverses_pinned_llama_qk_permutation() -> None:
    from mobius.integrations.gguf._arch_registry import get_arch_spec

    spec = get_arch_spec("eagle3")
    assert spec.tensor_processor == "llama"
    assert spec.llama_qk_permute is True


def test_auxiliary_scale_is_rejected_instead_of_dropped(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / "dflash-scale.gguf",
        "dflash",
        extra_tensor="fc.scale",
    )
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="cannot represent GGUF scale/input_scale"):
        build_from_gguf(path, target_config=_target([f"token-{i}" for i in range(64)]))


def test_standalone_task_override_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from mobius import _builder as core_builder
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(tmp_path / "dflash.gguf", "dflash")
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )
    with pytest.raises(ValueError, match="only supports task='dflash-draft'"):
        build_from_gguf(
            path,
            target_config=_target([f"token-{i}" for i in range(64)]),
            task="text-generation",
        )


@pytest.mark.parametrize("architecture", ["dflash", "eagle3"])
@pytest.mark.parametrize("quantized", [False, True], ids=["float", "q4"])
def test_synthetic_draft_runs_multiple_speculative_steps(
    tmp_path: Path, architecture: str, quantized: bool
) -> None:
    from mobius._testing.ort_inference import OnnxModelSession
    from mobius.integrations.gguf import build_from_gguf

    path = _write_draft(
        tmp_path / f"{architecture}-{quantized}.gguf", architecture, quantized=quantized
    )
    tokens = [f"token-{index}" for index in range(64)]
    package = build_from_gguf(path, target_config=_target(tokens))
    assert package.draft_manifest["standalone"] is False
    assert package.draft_manifest["runtime"] == "runtime_unvalidated"

    session = OnnxModelSession(package["model"])
    try:
        if architecture == "dflash":
            state = {
                "past_key_values.0.key": np.empty((1, 2, 0, 16), np.float32),
                "past_key_values.0.value": np.empty((1, 2, 0, 16), np.float32),
            }
            for accepted in (2, 1):
                output = session.run(
                    {
                        "noise_embedding": np.ones((1, 2, 64), np.float32),
                        "target_hidden": np.ones((1, accepted, 192), np.float32),
                        "position_ids": np.arange(accepted + 2, dtype=np.int64)[None, :],
                        "q_position_ids": np.arange(accepted, accepted + 2, dtype=np.int64)[
                            None, :
                        ],
                        **state,
                    }
                )
                assert output["draft_hidden"].shape == (1, 2, 64)
                state = {
                    "past_key_values.0.key": output["present.0.key"],
                    "past_key_values.0.value": output["present.0.value"],
                }
        else:
            state = {
                "past_key_values.0.key": np.empty((1, 2, 0, 16), np.float32),
                "past_key_values.0.value": np.empty((1, 2, 0, 16), np.float32),
            }
            recycled = np.zeros((1, 1, 64), np.float32)
            for accepted in (True, False):
                output = session.run(
                    {
                        "inputs_embeds": np.ones((1, 1, 64), np.float32),
                        "fused_hidden": np.ones((1, 1, 192), np.float32),
                        "recycled_hidden": recycled,
                        "attention_mask": np.ones(
                            (1, state["past_key_values.0.key"].shape[2] + 1), np.int64
                        ),
                        "position_ids": np.array([[state["past_key_values.0.key"].shape[2]]]),
                        **state,
                    }
                )
                assert output["draft_hidden"].shape == (1, 1, 64)
                recycled = output["next_hidden"] if accepted else recycled
                state = {
                    "past_key_values.0.key": output["present.0.key"],
                    "past_key_values.0.value": output["present.0.value"],
                }
    finally:
        session.close()


def test_cli_exposes_explicit_target_config() -> None:
    from mobius.__main__ import build_parser

    args = build_parser().parse_args(
        [
            "build-gguf",
            "draft.gguf",
            "--target-config",
            "target/config.json",
            "--target-gguf",
            "target.gguf",
            "--runtime",
            "ort-genai",
            "--runtime-version",
            "0.15.2",
            "--output",
            "out",
        ]
    )
    assert args.target_config == "target/config.json"
    assert args.target_gguf == "target.gguf"


def test_cli_target_pair_uses_hashed_pair_writer(tmp_path: Path, monkeypatch) -> None:
    import onnx_ir as ir

    from mobius.__main__ import main
    from mobius._model_package import ModelPackage

    package = ModelPackage(
        {
            "target": ir.Model(ir.Graph([], [], nodes=[], name="target"), ir_version=11),
            "draft": ir.Model(ir.Graph([], [], nodes=[], name="draft"), ir_version=11),
        }
    )
    package.draft_manifest = {"architecture": "dflash"}
    captured = {}

    monkeypatch.setattr(
        "mobius.integrations.gguf.build_draft_pair_from_gguf",
        lambda *args, **kwargs: package,
    )

    def write_pair(pkg, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir()
        manifest = output / "draft_manifest.json"
        manifest.write_text("{}")
        captured.update(package=pkg, output=output, kwargs=kwargs)
        return {"package": str(output), "manifest": str(manifest)}

    monkeypatch.setattr(
        "mobius.integrations.gguf.write_draft_pair_package",
        write_pair,
    )
    output = tmp_path / "output"
    main(
        [
            "build-gguf",
            "draft.gguf",
            "--target-config",
            "target-config",
            "--target-gguf",
            "target.gguf",
            "--runtime",
            "ort-genai",
            "--runtime-version",
            "0.15.2",
            "--output",
            str(output),
        ]
    )

    assert captured["package"] is package
    assert captured["output"] == output
    assert captured["kwargs"]["external_data"] == "onnx"
    assert captured["kwargs"]["requested_runtime"] == "ort-genai"
    assert captured["kwargs"]["runtime_version"] == "0.15.2"


@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_bad_target_never_mutates_output(tmp_path: Path, existing_output: bool) -> None:
    from mobius.__main__ import main

    tokens = [f"token-{index}" for index in range(64)]
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    target = tmp_path / "target"
    _write_target_dir(target, tokens)
    (target / "config.json").write_text("{", encoding="utf-8")
    output = tmp_path / "output"
    if existing_output:
        output.mkdir()
        (output / "sentinel.bin").write_bytes(b"unchanged")
    before = _directory_bytes(output) if existing_output else None

    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        main(
            [
                "build-gguf",
                str(draft),
                "--target-config",
                str(target),
                "--output",
                str(output),
            ]
        )

    assert output.exists() is existing_output
    if existing_output:
        assert _directory_bytes(output) == before


@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_draft_runtime_export_reaches_advisory_package_writer(
    tmp_path: Path,
    monkeypatch,
    existing_output: bool,
) -> None:
    from mobius.__main__ import main

    tokens = [f"token-{index}" for index in range(64)]
    draft = _write_draft(tmp_path / "dflash.gguf", "dflash")
    target = tmp_path / "target"
    _write_target_dir(target, tokens)
    output = tmp_path / "output"
    if existing_output:
        output.mkdir()
        (output / "sentinel.bin").write_bytes(b"unchanged")
    before = _directory_bytes(output) if existing_output else None
    package = mock.MagicMock()
    package.__iter__.return_value = iter(("model",))
    monkeypatch.setattr(
        "mobius.integrations.gguf.build_from_gguf",
        lambda *args, **kwargs: package,
    )
    writer_calls = []
    monkeypatch.setattr(
        "mobius.integrations.gguf.write_gguf_runtime_package",
        lambda *args, **kwargs: writer_calls.append((args, kwargs)) or {},
    )

    main(
        [
            "build-gguf",
            str(draft),
            "--target-config",
            str(target),
            "--runtime",
            "onnx-genai",
            "--output",
            str(output),
        ]
    )

    assert writer_calls
    assert output.exists() is existing_output
    if existing_output:
        assert _directory_bytes(output) == before


@pytest.mark.parametrize("architecture", ["dflash", "eagle3"])
def test_compact_draft_logits_have_explicit_target_remap(
    tmp_path: Path, architecture: str
) -> None:
    from mobius._testing.ort_inference import OnnxModelSession
    from mobius.integrations.gguf import build_from_gguf

    remap = np.array([3, 7, 11, 15, 19, 23, 27, 31], dtype=np.int64)
    path = _write_draft(
        tmp_path / f"{architecture}-remap.gguf",
        architecture,
        d2t=remap,
    )
    package = build_from_gguf(
        path,
        target_config=_target([f"token-{index}" for index in range(64)]),
    )
    manifest = package.draft_manifest
    assert manifest["draft_to_target"] == remap.tolist()
    assert manifest["orchestration"]["graph_output"] == "draft_logits"
    assert manifest["orchestration"]["lm_head_source"] == "draft"

    session = OnnxModelSession(package["model"])
    try:
        state = {
            "past_key_values.0.key": np.empty((1, 2, 0, 16), np.float32),
            "past_key_values.0.value": np.empty((1, 2, 0, 16), np.float32),
        }
        if architecture == "dflash":
            outputs = session.run(
                {
                    "noise_embedding": np.ones((1, 2, 64), np.float32),
                    "target_hidden": np.ones((1, 1, 192), np.float32),
                    "position_ids": np.arange(3, dtype=np.int64)[None, :],
                    "q_position_ids": np.arange(1, 3, dtype=np.int64)[None, :],
                    **state,
                }
            )
            assert outputs["draft_logits"].shape == (1, 2, len(remap))
        else:
            outputs = session.run(
                {
                    "inputs_embeds": np.ones((1, 1, 64), np.float32),
                    "fused_hidden": np.ones((1, 1, 192), np.float32),
                    "recycled_hidden": np.zeros((1, 1, 64), np.float32),
                    "attention_mask": np.ones((1, 1), np.int64),
                    "position_ids": np.zeros((1, 1), np.int64),
                    **state,
                }
            )
            assert outputs["draft_logits"].shape == (1, 1, len(remap))

        proposed_draft_ids = outputs["draft_logits"].argmax(axis=-1)
        proposed_target_ids = remap[proposed_draft_ids]
        assert np.isin(proposed_target_ids, remap).all()
    finally:
        session.close()
