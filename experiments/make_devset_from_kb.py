import json, yaml
from pathlib import Path
from random import shuffle, seed
seed(42)

cfg = yaml.safe_load(open("experiments/config.yaml","r",encoding="utf-8"))
kb_dir = Path(cfg["paths"]["kb_dir"])
outp = Path(cfg["paths"]["devset_path"])
outp.parent.mkdir(parents=True, exist_ok=True)

examples = []
for p in kb_dir.glob("*.json"):
    kb = json.load(open(p,"r",encoding="utf-8"))
    for it in kb[:]:
        title = (it.get("title") or "").strip()
        text  = (it.get("text") or "").strip()
        sid   = it.get("stable_id") or f"{p.stem}:{title}"
        if not title or not text: continue
        # template cực đơn giản
        if "điều kiện" in title.lower():
            q = f"Những điều kiện nào áp dụng theo {title}?"
        elif "cấm" in title.lower():
            q = f"Theo {title}, các hành vi bị cấm là gì?"
        else:
            q = f"{title} quy định vấn đề gì?"
        examples.append({"qid": sid, "question": q, "gold":[sid]})

shuffle(examples)
examples = examples[:100]
with open(outp,"w",encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False)+"\n")
print(f"Wrote {len(examples)} items to {outp}")
