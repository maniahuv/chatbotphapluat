import json, yaml
from tqdm import tqdm
from search_pipeline import HybridSearcher
from rerank import CrossEncoderReranker

cfg = yaml.safe_load(open("experiments/config.yaml","r",encoding="utf-8"))
dev = [json.loads(l) for l in open(cfg["paths"]["devset_path"],"r",encoding="utf-8")]
hs = HybridSearcher(cfg)
rr = CrossEncoderReranker(cfg["reranker"]["model_name"])
keep_topk = cfg["reranker"]["keep_topk"]
tau = cfg["thresholds"]["answerability_min_score"]

def meta_to_id(m): return m.get("stable_id") or m.get("chunk_id")

n_refuse, n_need_refuse, total = 0, 0, 0
for ex in tqdm(dev):
    total += 1
    cands = hs.search(ex["question"])
    reranked, smax = rr.rerank(ex["question"], cands, keep_topk=keep_topk)
    if smax < tau:
        n_refuse += 1
        # nếu gold tồn tại nhưng smax thấp => cần từ chối (đếm riêng optional)
        # n_need_refuse += ...
print(f"Refusal rate (threshold={tau}): {n_refuse/total:.3f} over {total} queries")
