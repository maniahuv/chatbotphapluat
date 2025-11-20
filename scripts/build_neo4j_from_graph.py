import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(BASE_DIR, "data", "knowledge_graph.json")

# Nạp thông tin đăng nhập
load_dotenv()
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"  # ⚠️ nếu bạn đã đổi mật khẩu, sửa ở đây

# Kết nối driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Đọc file JSON graph
with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

with driver.session() as sess:
    print("🔹 Đang nạp các node...")
    for node in data.get("nodes", []):
        sess.run("""
            MERGE (n:Article {id:$id})
            SET n.topic = $topic
        """, id=node["id"], topic=node.get("topic", ""))

    print("🔹 Đang nạp các quan hệ...")
    for edge in data.get("edges", []):  # <- Vòng lặp khai báo biến edge
        sess.run("""
            MATCH (a:Article {id:$from_id}), (b:Article {id:$to_id})
            MERGE (a)-[:RELATED {relation:$relation}]->(b)
        """, from_id=edge["from"], to_id=edge["to"], relation=edge.get("relation", "liên quan đến"))

print("✅ Đã nạp Knowledge Graph vào Neo4j thành công!")
