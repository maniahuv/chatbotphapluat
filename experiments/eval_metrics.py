import numpy as np
def recall_at_k(retrieved_ids, gold_ids, k=10):
    got = set(retrieved_ids[:k]) & set(gold_ids)
    return 1.0 if got else 0.0

def mrr(retrieved_ids, gold_ids):
    gold = set(gold_ids)
    for i, did in enumerate(retrieved_ids, start=1):
        if did in gold: return 1.0/i
    return 0.0

def ndcg_at_k(retrieved_ids, gold_ids, k=10):
    # binary gain
    dcg, idcg = 0.0, 1.0  # one relevant
    for i, did in enumerate(retrieved_ids[:k], start=1):
        if did in set(gold_ids):
            dcg = 1.0/np.log2(i+1)
            break
    return dcg/idcg
