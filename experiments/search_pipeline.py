import json, yaml, faiss, numpy as np, pickle
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

def rrf_fuse(ranked_lists, K=60, topk=10):
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, idx in enumerate(lst, start=1):
            scores[idx] += 1.0 / (K + rank)
    return [i for i,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:topk]

class HybridSearcher:
    def __init__(self, cfg):
        arts = Path(cfg["paths"]["artifacts_dir"])
        self.docs = json.load(open(arts/"docs.json","r",encoding="utf-8"))
        self.metas = json.load(open(arts/"metas.json","r",encoding="utf-8"))
        self.bm25 = pickle.load(open(arts/"bm25.pkl","rb"))
        self.faiss = faiss.read_index(str(arts/"faiss.index"))
        self.emb = SentenceTransformer(cfg["index"]["embedding_model"])

        self.bm25_topk = cfg["retrieval"]["bm25_topk"]
        self.dense_topk = cfg["retrieval"]["dense_topk"]
        self.rrf_K = cfg["retrieval"]["rrf_K"]
        self.final_topk = cfg["retrieval"]["final_topk"]

    def tokenized(self, s): return s.lower().split()

    def search(self, query):
        # BM25
        tokenized_q = self.tokenized(query)
        bm25_scores = self.bm25.get_scores(tokenized_q)
        bm25_rank = np.argsort(-bm25_scores)[:self.bm25_topk].tolist()

        # FAISS
        qv = self.emb.encode([query], normalize_embeddings=True)
        D, I = self.faiss.search(qv.astype(np.float32), self.dense_topk)
        dense_rank = I[0].tolist()

        fused = rrf_fuse([bm25_rank, dense_rank], K=self.rrf_K, topk=self.final_topk)
        return [{"rank": r+1, "doc": self.docs[i], "meta": self.metas[i]} for r,i in enumerate(fused)]
