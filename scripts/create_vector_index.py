import os
import json
from tqdm import tqdm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

#  Đường dẫn thư mục dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DIR = os.path.join(BASE_DIR, "data", "chunks")
VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_db")

#  Tạo thư mục nếu chưa có
os.makedirs(VECTOR_DIR, exist_ok=True)


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("⚠️ Chưa có GOOGLE_API_KEY. Vui lòng thêm vào file .env (lấy từ https://aistudio.google.com/app/apikey).")
    exit(1)

#  Khởi tạo model embedding của Gemini
embedding = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",  # model embedding miễn phí
    google_api_key=api_key
)

#  Gom tất cả các đoạn (chunks) từ các file JSON
all_texts = []
metadata_list = []

for filename in tqdm(os.listdir(CHUNK_DIR), desc="Đang tải các file chunks..."):
    if filename.endswith(".json"):
        with open(os.path.join(CHUNK_DIR, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk in chunks:
                all_texts.append(chunk)
                metadata_list.append({"source": filename})

if not all_texts:
    print(" Không tìm thấy dữ liệu trong thư mục chunks/. Hãy chạy split_text.py trước.")
    exit(1)

print(f" Tổng số đoạn văn cần nhúng: {len(all_texts)}")

# ⚡ Tạo vector database FAISS
print(" Đang tạo FAISS index bằng Gemini embeddings, vui lòng chờ...")
vector_db = FAISS.from_texts(all_texts, embedding, metadatas=metadata_list)

#  Lưu lại index vào thư mục vector_db
vector_db.save_local(VECTOR_DIR)

print(f" Hoàn tất! Vector index đã lưu tại: {VECTOR_DIR}")
