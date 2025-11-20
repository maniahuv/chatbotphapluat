import json, yaml
from pathlib import Path
from tqdm import tqdm
from search_pipeline import HybridSearcher
from eval_metrics import recall_at_k, mrr, ndcg_at_k

cfg = yaml.safe_load(open("experiments/config.yaml","r",encoding="utf-8"))
dev = [json.loads(l) for l in open(cfg["paths"]["devset_path"],"r",encoding="utf-8")]
hs = HybridSearcher(cfg)

def meta_to_id(m):  # tuỳ bạn chuẩn hoá
    return m.get("stable_id") or m.get("chunk_id")

R10 = []; MRR = []; N10 = []
for ex in tqdm(dev):
    q = ex["question"]
    gold = ex["gold"]
    hits = hs.search(q)
    retrieved_ids = [meta_to_id(h["meta"]) for h in hits]
    R10.append(recall_at_k(retrieved_ids, gold, k=10))
    MRR.append(mrr(retrieved_ids, gold))
    N10.append(ndcg_at_k(retrieved_ids, gold, k=10))

print(f"Hybrid: Recall@10={sum(R10)/len(R10):.3f}  MRR={sum(MRR)/len(MRR):.3f}  nDCG@10={sum(N10)/len(N10):.3f}")
