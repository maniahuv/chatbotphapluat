# experiments/quick_try.py
import yaml
from experiments.search_pipeline import HybridSearcher
from experiments.rerank import CrossEncoderReranker

TEST_QUERIES = [
    "Điều 8 Luật Hôn nhân và Gia đình 2014 quy định gì?",
    "Các hành vi bị cấm trong hôn nhân là gì?",
    "Nghị định 117/2024/NĐ-CP phần xử phạt hành vi X?",
    "Thẩm quyền giải quyết ly hôn đơn phương?",
    "Hồ sơ cần nộp để đăng ký kết hôn?",
    "Mức phạt không đăng ký kết hôn là bao nhiêu?",
]

if __name__ == "__main__":
    cfg = yaml.safe_load(open("experiments/config.yaml","r",encoding="utf-8"))
    hs = HybridSearcher(cfg)
    rr = CrossEncoderReranker(cfg["reranker"]["model_name"])
    keep_topk = cfg["reranker"]["keep_topk"]
    for q in TEST_QUERIES:
        print("\n==============================")
        print("Q:", q)
        hits = hs.search(q)  # E2: hybrid top-k
        print(f"[E2] Top-{len(hits)} (tiêu đề rút gọn):")
        for i,h in enumerate(hits,1):
            title = h["meta"].get("title") or h["meta"].get("section_title") or h["meta"].get("chunk_id")
            print(f"{i:>2}. {title}  | {h['meta'].get('source_file')}")
        # E3: rerank
        reranked, smax = rr.rerank(q, hits, keep_topk=keep_topk)
        print(f"[E3] Rerank top-{len(reranked)} (smax={smax:.3f}):")
        for i,h in enumerate(reranked,1):
            title = h["meta"].get("title") or h["meta"].get("section_title") or h["meta"].get("chunk_id")
            print(f"{i:>2}. {title}  | score={h['rerank_score']:.3f}")
