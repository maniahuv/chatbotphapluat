import os, json, re
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
META_FILE = os.path.join(BASE_DIR, "data", "metadata", "documents.json")

os.makedirs(KB_DIR, exist_ok=True)

# Đọc metadata danh sách văn bản
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata_docs = {doc["file"]: doc for doc in json.load(f)}

import re

def extract_knowledge_units(text):
    """
    Tách tri thức từ văn bản pháp luật.
    - Nếu có 'Điều', sẽ tách theo Điều 1., Điều 2., ...
    - Nếu không có 'Điều', sẽ tách theo Mục I., II., III., ...
    """

    # Loại bỏ khoảng trắng dư, dòng trống
    text = re.sub(r'\n+', '\n', text.strip())

    # Nếu văn bản có 'Điều', thì tách theo điều luật
    if re.search(r"Điều\s+\d+", text):
        pattern = r"(Điều\s+\d+[\.\:\-]?\s*.*?)(?=(Điều\s+\d+[\.\:\-]?\s)|$)"
        parts = re.findall(pattern, text, flags=re.DOTALL)
        parts = [p[0].strip() for p in parts]
        unit_type = "Điều"

    # Ngược lại, tách theo Mục I., II., III. (thường thấy ở Nghị quyết, Thông tư)
    else:
        pattern = r"([IVXLC]+\.\s.*?)(?=[IVXLC]+\.\s|$)"
        parts = re.findall(pattern, text, flags=re.DOTALL)
        parts = [p.strip() for p in parts if len(p.strip()) > 30]
        unit_type = "Mục"

    return parts, unit_type


for filename in tqdm(os.listdir(CLEAN_DIR), desc="Đang xử lý các văn bản..."):
    if filename.endswith(".txt"):
        file_path = os.path.join(CLEAN_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_meta = metadata_docs.get(filename, {})
        parts, unit_type = extract_knowledge_units(text)
        knowledge = []

    for i, part in enumerate(parts, 1):
        if unit_type == "Điều":
            match = re.match(r"(Điều\s+\d+)", part)
            title = match.group(1) if match else f"Điều {i}"
        else:
            match = re.match(r"([IVXLC]+)\.", part)
            title = f"Mục {match.group(1)}" if match else f"Mục {i}"

        knowledge.append({
            "law_name": doc_meta.get("title", "Không rõ"),
            "type": doc_meta.get("type", "Không rõ"),
            "year": doc_meta.get("year", ""),
            "section_type": unit_type,
            "title": title,
            "text": part.strip(),
            "source_file": filename
        })

        # Lưu mỗi văn bản thành file JSON tri thức
        out_file = os.path.join(KB_DIR, filename.replace("_clean.txt", "_knowledge.json"))
        with open(out_file, "w", encoding="utf-8") as out:
            json.dump(knowledge, out, ensure_ascii=False, indent=2)

print(" Hoàn tất! Tri thức đã lưu trong thư mục /data/knowledge_base/")
