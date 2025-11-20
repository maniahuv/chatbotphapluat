from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def rerank(self, query, candidates, keep_topk=10, batch_size=32):
        pairs = [(query, c["doc"]) for c in candidates]
        scores = []
        for i in range(0, len(pairs), batch_size):
            q, d = zip(*pairs[i:i+batch_size])
            enc = self.tok(list(q), list(d), padding=True, truncation=True, return_tensors="pt").to(self.device)
            out = self.model(**enc).logits.squeeze(-1)
            scores.extend(out.detach().cpu().tolist())
        order = np.argsort(-np.array(scores))[:keep_topk]
        reranked = [candidates[i] | {"rerank_score": float(scores[i])} for i in order]
        return reranked, float(np.max(scores)) if scores else 0.0
