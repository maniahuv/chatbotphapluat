# scripts/chatbot_legal.py
import os, time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# --- Init ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_db")

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY
)
vector_db = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
model = genai.GenerativeModel("gemini-2.0-flash")

def query_vector(query: str, k: int = 5):
    """Vector-only RAG. Trả về (answer, sources, latency_seconds)."""
    t0 = time.perf_counter()
    results = vector_db.similarity_search(query, k=k)
    context = "\n\n".join(r.page_content for r in results)
    prompt = f"""
Bạn là trợ lý pháp lý am hiểu luật Việt Nam.
Hãy trả lời ngắn gọn, chính xác và có dẫn Điều luật liên quan.

Câu hỏi: {query}

Các đoạn luật tham khảo:
{context}
"""
    resp = model.generate_content(prompt)
    latency = time.perf_counter() - t0
    sources = [r.metadata.get("source") for r in results]
    return resp.text, sources, latency

if __name__ == "__main__":
    q = input(" Nhập câu hỏi pháp luật của bạn: ")
    ans, srcs, t = query_vector(q)
    print("\n Trả lời:\n", ans)
    print("\n Dẫn nguồn:")
    for s in srcs: print("→", s)
    print(f"\n⚡ Latency: {t:.2f}s")
