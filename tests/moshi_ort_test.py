from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "personaplex"


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


moshi_ort = _load_example("moshi_ort")


@pytest.mark.parametrize(
    ("initializer_name", "dep_q"),
    [("depformer_in.weight", 8), ("depformer_in.weight_Q4", 16)],
)
def test_infer_dep_q_from_raw_or_quantized_weight(monkeypatch, initializer_name, dep_q):
    model = SimpleNamespace(
        graph=SimpleNamespace(
            initializers={initializer_name: SimpleNamespace(shape=(dep_q, 1024, 4096))}
        )
    )
    monkeypatch.setattr(moshi_ort.ir, "load", lambda _: model)

    assert moshi_ort._infer_dep_q("depformer/model.onnx") == dep_q


@pytest.mark.parametrize(
    "initializers",
    [
        {},
        {"depformer_in.weight": SimpleNamespace(shape=(1, 1024, 4096))},
        {
            "depformer_in.weight": SimpleNamespace(shape=(8, 1024, 4096)),
            "depformer_in.weight_Q4": SimpleNamespace(shape=(16, 1024, 512)),
        },
    ],
)
def test_infer_dep_q_rejects_missing_unsupported_or_ambiguous_width(monkeypatch, initializers):
    model = SimpleNamespace(graph=SimpleNamespace(initializers=initializers))
    monkeypatch.setattr(moshi_ort.ir, "load", lambda _: model)

    with pytest.raises(ValueError, match="depformer width"):
        moshi_ort._infer_dep_q("depformer/model.onnx")


def test_runtime_rejects_explicit_width_mismatch_before_loading_sessions(monkeypatch):
    monkeypatch.setattr(moshi_ort, "_infer_dep_q", lambda _: 8)

    with pytest.raises(ValueError, match=r"Requested dep_q=16.*contains 8"):
        moshi_ort.MoshiORT("model", "cpu", False, dep_q=16)


def test_server_threads_dep_q_to_runtime(monkeypatch):
    class FakeRouter:
        def add_get(self, *_):
            pass

        def add_static(self, *_):
            pass

    class FakeApplication(dict):
        def __init__(self):
            super().__init__()
            self.router = FakeRouter()

    fake_web = SimpleNamespace(Application=FakeApplication)
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(WSMsgType=SimpleNamespace(), web=fake_web),
    )
    server = _load_example("server")
    captured = {}

    class FakeMoshiORT:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(server, "MoshiORT", FakeMoshiORT)
    monkeypatch.setattr(
        server,
        "load_persona_tokenizer",
        lambda _: (_ for _ in ()).throw(RuntimeError("not needed")),
    )

    server.build_app("model", "cpu", False, dep_q=8)

    assert captured == {
        "args": ("model", "cpu", False),
        "kwargs": {"dep_q": 8},
    }
