import os
import json
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

#  Đường dẫn thư mục dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
CHUNK_DIR = os.path.join(BASE_DIR, "data", "chunks")

#  Tạo thư mục đầu ra nếu chưa có
os.makedirs(CHUNK_DIR, exist_ok=True)

#  Cấu hình chia nhỏ văn bản
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

#  Duyệt qua tất cả file .txt trong cleaned/
for filename in tqdm(os.listdir(CLEAN_DIR), desc="Đang chia nhỏ văn bản..."):
    if filename.endswith(".txt"):
        file_path = os.path.join(CLEAN_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f" File trống: {filename}")
            continue

        # Chia nhỏ thành danh sách các đoạn
        chunks = splitter.split_text(text)

        # Lưu kết quả vào file JSON
        chunk_file = filename.replace("_clean.txt", "_chunks.json")
        out_path = os.path.join(CHUNK_DIR, chunk_file)

        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(chunks, out, ensure_ascii=False, indent=2)

        print(f" Đã chia nhỏ: {filename} → {chunk_file}")

print("\n🎉 Hoàn tất! Các file chunks đã được lưu trong /data/chunks/")
