from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# 🔑 Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("⚠️ Chưa có GOOGLE_API_KEY trong file .env")
    exit(1)

#  Đường dẫn
VECTOR_DIR = "data/vector_db"

#  Khởi tạo model embedding Gemini
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

#  Load FAISS index
vector_db = FAISS.load_local(
    VECTOR_DIR,
    embedding,
    allow_dangerous_deserialization=True
)

# 🕵️ Truy vấn thử
query = "Khi ly hôn, ai có quyền nuôi con?"
results = vector_db.similarity_search(query, k=3)

print(f"🔍 Kết quả cho câu hỏi: {query}\n")
for i, r in enumerate(results, 1):
    print(f"{i}. 📜 {r.page_content[:250]}...")
    print(f"   📎 Nguồn: {r.metadata}\n")
