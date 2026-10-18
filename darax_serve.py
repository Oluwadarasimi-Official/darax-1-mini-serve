"""DARAX native inference service — serves ONLY our own weights.

- Loads darax-1-mini.pt (17.4M params, trained 2026-10-14) + darax-bpe.json once.
- POST /generate  {prompt, max_new_tokens=100, temperature=0.8, top_k=50}
- GET  /health    {status, model, params, ...}
- Makes ZERO external network calls. No Gemini/Groq/fallbacks anywhere.

Run:  uvicorn darax_serve:app --host 0.0.0.0 --port 8000
"""
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from darax_model import DaraxGPT, DaraxConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("DARAX_CKPT", os.path.join(HERE, "weights.pt"))
TOK = os.environ.get("DARAX_TOK", os.path.join(HERE, "tokenizer.json"))
MODEL_ID = "darax-1-mini"

app = FastAPI(title="DARAX native inference")


class GenRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=100, ge=1, le=200)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)


_model = None
_tok = None
_cfg = None


def get_model():
    global _model, _tok, _cfg
    if _model is None:
        if not os.path.exists(CKPT):
            raise RuntimeError(f"checkpoint missing: {CKPT}")
        if not os.path.exists(TOK):
            raise RuntimeError(f"tokenizer missing: {TOK}")
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        _cfg = DaraxConfig(**ckpt["config"])
        _model = DaraxGPT(_cfg)
        _model.load_state_dict(ckpt["model"], strict=True)
        _model.eval()
        _tok = Tokenizer.from_file(TOK)
    return _model, _tok, _cfg


@app.on_event("startup")
def _startup():
    get_model()  # fail fast if weights are broken


@app.get("/health")
def health():
    try:
        model, tok, cfg = get_model()
        n = sum(p.numel() for p in model.parameters())
        return {"status": "ok", "model": MODEL_ID, "params": n,
                "context": cfg.block_size, "vocab": tok.get_vocab_size(),
                "provider": "darax-native"}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model unavailable: {e}")


@app.post("/generate")
def generate(req: GenRequest):
    t0 = time.time()
    try:
        model, tok, cfg = get_model()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model unavailable: {e}")
    prompt = req.prompt[: cfg.block_size * 4]  # crude guard; tokenizer enforces ctx
    ids = tok.encode(prompt).ids[-cfg.block_size:]
    if not ids:
        raise HTTPException(status_code=400, detail="empty prompt after tokenization")
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]),
                             max_new_tokens=req.max_new_tokens,
                             temperature=req.temperature,
                             top_k=req.top_k)[0].tolist()
    text = tok.decode(out)
    # strip the echoed prompt so the API returns only the continuation
    cont = text[len(tok.decode(ids)):]
    return {"model": MODEL_ID, "provider": "darax-native",
            "text": cont.strip(),
            "tokens": len(out) - len(ids),
            "prompt_tokens": len(ids),
            "latency_s": round(time.time() - t0, 2)}
