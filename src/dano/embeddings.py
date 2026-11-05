"""Pinned frozen RuBERT embeddings; every token is processed in overlapping chunks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MODEL_ID = "cointegrated/rubert-tiny2"


def download_model():
    from huggingface_hub import HfApi, snapshot_download
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / "transformer_provenance.json"
    if provenance_path.exists():
        revision = json.loads(provenance_path.read_text())["revision"]
    else:
        revision = HfApi().model_info(MODEL_ID).sha
    path = snapshot_download(MODEL_ID, revision=revision,
                             allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"],
                             max_workers=2)
    files = {}
    for file in Path(path).iterdir():
        if file.is_file():
            files[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(dict(model_id=MODEL_ID, revision=revision,
        files_sha256=files, pooling="attention-masked token mean; token-count weighted chunk mean; L2 normalized",
        max_length=256, stride=32, trainable_encoder=False), indent=2) + "\n")
    print("Pinned model", MODEL_ID, revision, flush=True)
    return path


def encode_problems():
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(4)
    provenance = json.loads((ROOT / "data/processed/transformer_provenance.json").read_text())
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, revision=provenance["revision"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoder = AutoModel.from_pretrained(path, local_files_only=True).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    records = [json.loads(line) for line in (ROOT / "data/processed/problem_texts.jsonl").read_text().splitlines()]
    outputs = []
    for start in range(0, len(records), 16):
        group = records[start:start+16]
        encoded = tokenizer([r["text"] for r in group], max_length=256, stride=32,
                            truncation=True, padding=True, return_overflowing_tokens=True,
                            return_tensors="pt")
        sample_map = encoded.pop("overflow_to_sample_mapping").numpy()
        vectors, counts = [], []
        for b in range(0, len(sample_map), 16):
            inputs = {k: v[b:b+16] for k, v in encoded.items()}
            with torch.inference_mode():
                hidden = encoder(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                count = mask.sum(dim=1)
                pooled = (hidden * mask).sum(dim=1) / count.clamp_min(1)
            vectors.append(pooled.numpy())
            counts.extend(count[:, 0].numpy().tolist())
        vectors = np.concatenate(vectors)
        counts = np.asarray(counts)
        for j in range(len(group)):
            mask = sample_map == j
            vec = np.average(vectors[mask], axis=0, weights=counts[mask])
            outputs.append(vec / max(float(np.linalg.norm(vec)), 1e-12))
        print("Encoded", min(start+16, len(records)), "/", len(records), flush=True)
    out = ROOT / "data/interim"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rubert_embeddings.npy", np.stack(outputs))
    (out / "embedding_order.json").write_text(json.dumps([dict(problem_id=r["problem_id"], text_sha256=hashlib.sha256(r["text"].encode()).hexdigest()) for r in records], indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        download_model()
    else:
        encode_problems()
