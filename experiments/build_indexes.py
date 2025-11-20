import os, json, yaml, faiss, numpy as np
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_dir):
    docs, metas = [], []
    for p in Path(chunks_dir).glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)  # list of dict or list of strings
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                meta = {**item, "source_file": p.name, "chunk_id": f"{p.stem}#{i}"}
            else:
                text = str(item)
                meta = {"source_file": p.name, "chunk_id": f"{p.stem}#{i}"}
            if text.strip():
                docs.append(text)
                metas.append(meta)
    return docs, metas

def tokenize_vn(s: str):
    # baseline: tách theo whitespace, chữ thường
    return s.lower().split()

def build_bm25(docs):
    tokenized = [tokenize_vn(d) for d in docs]
    return BM25Okapi(tokenized)

def build_faiss(docs, model_name, nlist=4096):
    model = SentenceTransformer(model_name)
    X = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    dim = X.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(X.astype(np.float32))
    index.add(X.astype(np.float32))
    return index, X

if __name__ == "__main__":
    cfg = yaml.safe_load(open("experiments/config.yaml", "r", encoding="utf-8"))
    chunks_dir = cfg["paths"]["chunks_dir"]
    arts_dir = Path(cfg["paths"]["artifacts_dir"])
    arts_dir.mkdir(parents=True, exist_ok=True)

    docs, metas = load_chunks(chunks_dir)
    print(f"Loaded {len(docs)} chunks")

    # BM25
    bm25 = build_bm25(docs)
    faiss.write_index = getattr(faiss, "write_index")
    # Lưu BM25 tokenized (đơn giản) + metas
    import pickle, json
    with open(arts_dir/"bm25.pkl", "wb") as f: pickle.dump(bm25, f)
    with open(arts_dir/"docs.json", "w", encoding="utf-8") as f: json.dump(docs, f, ensure_ascii=False)
    with open(arts_dir/"metas.json", "w", encoding="utf-8") as f: json.dump(metas, f, ensure_ascii=False)

    # FAISS
    idx, _ = build_faiss(docs, cfg["index"]["embedding_model"], cfg["index"]["faiss_nlist"])
    faiss.write_index(idx, str(arts_dir/"faiss.index"))
    print("Indexes saved to experiments/artifacts/")
