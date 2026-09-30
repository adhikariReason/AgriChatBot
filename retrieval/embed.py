# -*- coding: utf-8 -*-
"""BGE-M3 dense embeddings (BAAI/bge-m3, MIT licence).

Plain transformers, no FlagEmbedding dependency: BGE-M3's dense vector is the
L2-normalised [CLS] hidden state.  Runs on CPU; on CPUs with AMX/AVX512-bf16 the
model runs under bf16 autocast, which is several times faster than fp32 at
negligible cost in retrieval quality.  Set BGE_DTYPE=fp32 to disable.

The model (~2.3 GB, pytorch_model.bin) is cached by huggingface_hub under
$HF_HOME (default ~/.cache/huggingface).  Nothing is stored in the repo.
"""
from __future__ import annotations

import os
import time

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
MAX_LEN = 512

_state = {}


def _load():
    if "model" in _state:
        return _state["tok"], _state["model"]
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("BGE_THREADS", os.cpu_count() or 4)))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    _state.update(tok=tok, model=model)
    return tok, model


def encode(texts, batch_size=16, max_len=MAX_LEN, progress=False, log=None):
    """texts -> float32 array (n, 1024), rows L2-normalised."""
    import torch
    tok, model = _load()
    use_bf16 = os.environ.get("BGE_DTYPE", "bf16") == "bf16"
    n = len(texts)
    order = np.argsort([-len(t) for t in texts])        # longest first: stable batches
    out = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for bi, start in enumerate(range(0, n, batch_size)):
            idx = order[start:start + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            if use_bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    h = model(**enc).last_hidden_state[:, 0]
            else:
                h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
            out[idx] = h.numpy()
            if progress and (bi % 20 == 0):
                done = min(n, start + batch_size)
                msg = f"  embedded {done}/{n}  {done / (time.time() - t0):.1f}/s"
                print(msg, flush=True)
                if log:
                    log(msg)
    return out


def token_lengths(texts):
    tok, _ = _load()
    return [len(x) for x in tok(list(texts), truncation=False)["input_ids"]]
