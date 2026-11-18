# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent CUDA goldens and persisted-package replay for continuous Sigma."""

from __future__ import annotations

import inspect
import json
import tempfile
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import patch

import numpy as np


def generate_golden(case, json_path: Path, device: str) -> None:
    """Record both native samplers and their exact upstream CUDA noise draws."""
    import torch
    import transformers

    from mobius._testing.golden import save_generation_json
    from mobius._testing.nemotron_sigma_reference import MODEL_ID, REVISION, load_reference

    if (case.model_id, case.revision, case.dtype) != (MODEL_ID, REVISION, "float32"):
        raise ValueError(
            "Sigma golden generation requires its pinned dual-head checkpoint "
            f"{MODEL_ID}@{REVISION} in float32; correct the case metadata."
        )
    if not device.startswith("cuda"):
        raise ValueError("Sigma reference goldens require --device cuda; select an idle GPU.")
    model, tokenizer = load_reference()
    prompt = case.prompts[0]
    count = case.generation_params["max_new_tokens"]
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    cache = inspect.getmodule(model).KVCache(model.config.num_hidden_layers)
    with torch.no_grad():
        hidden = model._encode_clean(
            ids,
            0,
            torch.arange(ids.shape[1], device="cuda")[None, :],
            torch.ones_like(ids, dtype=torch.bool),
            cache,
        )
        logits = model.ar_head(hidden)
        if not logits.isfinite().all():
            raise RuntimeError(
                "Reference prefill logits are nonfinite; do not record goldens."
            )
        prefill = logits.float().cpu().numpy()
    arrays = {"prefill_logits": prefill}
    original = torch.randn

    def record(prefix, call):
        draws = []

        def capture(*args, **kwargs):
            tensor = original(*args, **kwargs)
            draws.append(tensor.cpu().numpy().copy())
            return tensor

        with patch("torch.randn", side_effect=capture):
            result = call()
        for index, draw in enumerate(draws):
            arrays[f"{prefix}_{index}"] = draw
        return result, len(draws)

    torch.manual_seed(20261116)
    (spec, stats), spec_draws = record(
        "spec",
        lambda: model.generate_speculative(ids, max_new_tokens=count, return_stats=True),
    )
    torch.manual_seed(20261117)
    (diffusion, nfe), diffusion_draws = record(
        "diffusion",
        lambda: model.generate(
            ids,
            max_new_tokens=count,
            time_schedule="log_snr",
            score_temp=1.0,
            guidance_scale=1.0,
        ),
    )
    installed = json.loads(distribution("transformers").read_text("direct_url.json") or "{}")
    result = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dtype": "float32",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_commit": installed.get("vcs_info", {}).get("commit_id"),
        "tf32": False,
        "reference_loader": "upstream from_pretrained + upstream NLCDRotaryEmbedding constructor",
        "prompt": prompt,
        "input_ids": ids[0].tolist(),
        "prefill_top1": prefill[0].argmax(axis=-1).tolist(),
        "speculative_seed": 20261116,
        "speculative_tokens": spec[0, ids.shape[1] :].tolist(),
        "speculative_stats": stats,
        "speculative_noise_draws": spec_draws,
        "diffusion_seed": 20261117,
        "diffusion_tokens": diffusion[0, ids.shape[1] :].tolist(),
        "diffusion_nfe": nfe,
        "diffusion_noise_draws": diffusion_draws,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(json_path.with_name(json_path.stem + "_inputs.npz"), **arrays)
    save_generation_json(
        json_path.with_name(json_path.stem + "_generation.json"),
        model_id=MODEL_ID,
        prompt=prompt,
        generated_tokens=result["speculative_tokens"],
        generated_text=tokenizer.decode(
            result["speculative_tokens"], skip_special_tokens=True
        ),
        provenance={
            "revision": REVISION,
            "sampler": "dual-head self-speculation",
            "native_diffusion_in_main_golden": True,
        },
    )


def assert_golden(package, json_path: Path, *, case, level: str) -> None:
    """Replay full logits or both full samplers after package save/load on CUDA."""
    import onnxruntime as ort

    from mobius import ModelPackage
    from mobius.generation.nemotron_sigma import NemotronSigmaGenerator

    golden = json.loads(json_path.read_text())
    for field in ("model_id", "revision", "dtype"):
        expected = getattr(case, field)
        if golden.get(field) != expected:
            raise ValueError(
                f"Sigma golden {field}={golden.get(field)!r} does not match "
                f"case {expected!r}; regenerate the pinned golden before replay."
            )
    count = case.generation_params["max_new_tokens"]
    for key in ("speculative", "diffusion"):
        actual = len(golden[f"{key}_tokens"])
        if actual != count:
            raise ValueError(
                f"Sigma {key} golden contains {actual} tokens, but the case "
                f"requires {count}; regenerate the full requested horizon."
            )
    if golden["prompt"] != case.prompts[0] or case.generation_params["do_sample"] is not False:
        raise ValueError(
            "Sigma golden prompt/sampling settings do not match the case; "
            "regenerate with the declared prompt and do_sample=false."
        )
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "Sigma golden replay requires onnxruntime-gpu and CUDAExecutionProvider; "
            "install the GPU runtime in the task environment."
        )
    with (
        np.load(json_path.with_name(json_path.stem + "_inputs.npz")) as arrays,
        tempfile.TemporaryDirectory(prefix="mobius-sigma-golden-") as directory,
    ):
        package.save(directory)
        restored = ModelPackage.load(directory)
        assert restored["model"].metadata_props["mobius.sigma"]
        options = ort.SessionOptions()
        options.add_session_config_entry("session.use_tf32", "0")
        session = ort.InferenceSession(
            str(Path(directory) / "model.onnx"),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
        )
        assert session.get_providers()[0] == "CUDAExecutionProvider"
        runtime = NemotronSigmaGenerator(session)
        ids = np.asarray([golden["input_ids"]], dtype=np.int64)
        if level == "L4":
            valid = np.ones(ids.shape, np.bool_)
            positions = np.arange(ids.shape[1], dtype=np.int64)[None, :]
            result, _ = runtime._run(ids, positions, valid, runtime._empty_cache(1))
            expected = arrays["prefill_logits"]
            assert result["ar_logits"].shape == expected.shape
            np.testing.assert_allclose(result["ar_logits"], expected, rtol=1e-3, atol=1e-4)
            return
        assert level == "L5"
        for mode, prefix, key in (
            ("generate_speculative", "spec", "speculative"),
            ("generate", "diffusion", "diffusion"),
        ):
            draws = iter(
                [arrays[f"{prefix}_{i}"] for i in range(golden[f"{key}_noise_draws"])]
            )

            def sample(shape, draws=draws):
                draw = next(draws)
                assert draw.shape == shape
                return draw.copy()

            kwargs = {"return_stats": True} if mode == "generate_speculative" else {}
            expected = np.asarray([golden[f"{key}_tokens"]], np.int64)
            output, stats = getattr(runtime, mode)(
                ids, max_new_tokens=count, noise_sampler=sample, **kwargs
            )
            assert output.shape == (1, ids.shape[1] + count)
            np.testing.assert_array_equal(output[:, ids.shape[1] :], expected)
            assert next(draws, None) is None
            if mode == "generate_speculative":
                assert stats == golden["speculative_stats"]
            else:
                assert stats == golden["diffusion_nfe"]
