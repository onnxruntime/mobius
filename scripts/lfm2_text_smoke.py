"""Smoke test: greedy decode LFM2-1.2B text-only via ORT."""
from __future__ import annotations

import sys
sys.path.insert(0, "examples")
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from lfm2_audio_realtime import _init_hybrid_cache, _update_hybrid_cache


def main():
    model_id = "LiquidAI/LFM2-1.2B"
    sess = ort.InferenceSession(
        "/tmp/lfm2-text/model.onnx", providers=["CPUExecutionProvider"]
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    prompt = "Hello, my name is"
    ids = tok(prompt, return_tensors="np")["input_ids"].astype(np.int64)
    print("prompt ids:", ids)

    cache = _init_hybrid_cache(sess, batch=1)

    generated = ids[0].tolist()
    cur_ids = ids
    past_len = 0
    for step in range(30):
        seq_len = cur_ids.shape[1]
        attn_mask = np.ones((1, past_len + seq_len), dtype=np.int64)
        position_ids = np.arange(past_len, past_len + seq_len, dtype=np.int64)[None, :]
        feeds = {
            "input_ids": cur_ids,
            "attention_mask": attn_mask,
            "position_ids": position_ids,
        }
        feeds.update(cache)
        outs = sess.run(None, feeds)
        logits = outs[0]
        next_id = int(np.argmax(logits[0, -1]))
        generated.append(next_id)
        _update_hybrid_cache(cache, outs, sess)
        past_len += seq_len
        cur_ids = np.array([[next_id]], dtype=np.int64)

    text = tok.decode(generated, skip_special_tokens=False)
    print("---generated---")
    print(text)
    print("---ids---")
    print(generated)


if __name__ == "__main__":
    main()
