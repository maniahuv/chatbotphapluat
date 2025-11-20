# scripts/evaluate_rag_vs_graphrag.py
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

from chatbot_legal import query_vector
try:
    from chatbot_graph_rag import query_graph_rag  # hàm này bạn đã/đang thêm
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False

def main():
    q = input("Nhập câu hỏi pháp luật của bạn: ")

    v_ans, v_srcs, v_t = query_vector(q)
    print("\n=== VECTOR-ONLY ===")
    print(v_ans)
    print("Nguồn:", v_srcs)
    print(f"Latency: {v_t:.2f}s")

    if HAS_GRAPH:
        h_ans, h_meta, h_t = query_graph_rag(q)
        print("\n=== HYBRID (GraphRAG) ===")
        print(h_ans)
        print("Meta:", h_meta)
        print(f"Latency: {h_t:.2f}s")
    else:
        print("\n(GraphRAG chưa sẵn sàng — chưa import được query_graph_rag)")

if __name__ == "__main__":
    main()
