# scripts/chatbot_graph_rag.py
import os, time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- ENV ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_db")

# --- Init Vector DB ---
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)
vector_db = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)

# --- Init LLM & Neo4j ---
model = genai.GenerativeModel("gemini-2.0-flash")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def _search_graph_by_concepts(concepts):
    if not concepts:
        return []
    with driver.session() as sess:
        res = sess.run("""
            MATCH (a:Article)-[r:RELATED]->(b:Article)
            WHERE any(c IN $concepts WHERE toLower(a.topic) CONTAINS toLower(c))
            RETURN a.id AS from_id, b.id AS to_id, coalesce(r.relation,'RELATED') AS rel, b.topic AS topic
            LIMIT 25
        """, concepts=[c.lower() for c in concepts])
        return [dict(r) for r in res]

def query_graph_rag(query: str, k: int = 5):
    import re, time
    t0 = time.perf_counter()

    # 1) Vector retrieve
    hits = vector_db.similarity_search(query, k=k)
    ctx_vec = "\n\n".join(h.page_content for h in hits)
    vec_sources = [h.metadata.get("source") for h in hits]

    # 2) Concepts từ câu hỏi (giữ nguyên, nhưng chỉ là nguồn phụ)
    extract = model.generate_content(
        f"Từ câu hỏi sau, liệt kê tối đa 5 khái niệm pháp lý cốt lõi (mỗi dòng 1 mục, không giải thích):\n{query}"
    )
    concepts = [x.strip("-• \n") for x in extract.text.splitlines() if x.strip()][:5]

    # 3) Fallback quan trọng: bắt 'Điều \d+' từ context vector
    article_ids = re.findall(r"Điều\s+\d+", ctx_vec, flags=re.IGNORECASE)
    article_ids = list({a.strip() for a in article_ids})[:10]  # unique & limit

    edges = []
    with driver.session() as sess:
        # 3a) mở rộng 1-hop từ các Article tìm được trong context
        if article_ids:
            res1 = sess.run("""
                MATCH (a:Article)-[r:RELATED]-(b:Article)
                WHERE a.id IN $ids
                RETURN a.id AS from_id, b.id AS to_id, coalesce(r.relation,'RELATED') AS rel, b.topic AS topic
                LIMIT 50
            """, ids=article_ids)
            edges += [dict(r) for r in res1]

        # 3b) nếu vẫn trống, thử khớp theo topic chứa concepts
        if not edges and concepts:
            res2 = sess.run("""
                MATCH (a:Article)-[r:RELATED]->(b:Article)
                WHERE any(c IN $concepts WHERE toLower(a.topic) CONTAINS toLower(c))
                   OR any(c IN $concepts WHERE toLower(b.topic) CONTAINS toLower(c))
                RETURN a.id AS from_id, b.id AS to_id, coalesce(r.relation,'RELATED') AS rel, b.topic AS topic
                LIMIT 50
            """, concepts=[c.lower() for c in concepts])
            edges += [dict(r) for r in res2]

    ctx_graph = "\n".join(f"{e['from_id']} {e['rel']} {e['to_id']} ({e.get('topic','')})" for e in edges) or "Không có."

    # 4) Tổng hợp trả lời
    prompt = f"""
Bạn là trợ lý pháp lý Việt Nam. Dựa vào ngữ cảnh dưới đây, trả lời chính xác, có dẫn Điều/khoản nếu có.

[Câu hỏi]
{query}

[Đoạn văn pháp luật gần nhất (Vector)]
{ctx_vec}

[Các quan hệ pháp lý từ đồ thị (Graph)]
{ctx_graph}
"""
    resp = model.generate_content(prompt)
    latency = time.perf_counter() - t0

    meta = {"concepts": concepts, "vector_sources": vec_sources, "graph_edges": edges, "article_ids_from_vector": article_ids}
    return resp.text, meta, latency

# Cho phép chạy lẻ để test nhanh
if __name__ == "__main__":
    q = input("❓ Nhập câu hỏi pháp luật: ")
    ans, meta, t = query_graph_rag(q)
    print("\n=== HYBRID (GraphRAG) ===")
    print(ans)
    print("\nMeta:", meta)
    print(f"\n⚡ Latency: {t:.2f}s")
