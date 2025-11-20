import os
import json
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
GRAPH_PATH = os.path.join(BASE_DIR, "data", "knowledge_graph.json")


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

for m in genai.list_models():
    print(m.name)

model = genai.GenerativeModel("gemini-2.5-flash")




all_sections = []
for filename in tqdm(os.listdir(KB_DIR), desc="Đọc file knowledge base..."):
    if filename.endswith(".json"):
        with open(os.path.join(KB_DIR, filename), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                all_sections.extend(data)
            except:
                pass

print(f" Đã tải {len(all_sections)} mục tri thức.")

#  Chuẩn bị prompt yêu cầu mô hình tạo graph
prompt = f"""
Bạn là chuyên gia pháp lý.
Phân tích các điều luật và tạo mạng tri thức (knowledge graph) dạng JSON như sau:

{{
  "nodes": [
    {{"id": "Điều 33", "topic": "Tài sản chung"}},
    {{"id": "Điều 59", "topic": "Chia tài sản khi ly hôn"}}
  ],
  "edges": [
    {{"from": "Điều 33", "to": "Điều 59", "relation": "liên quan đến"}}
  ]
}}

Dữ liệu đầu vào gồm các đoạn luật sau (rút gọn mỗi đoạn, chỉ cần trích tiêu đề và tóm tắt nội dung):

{[{"title": s.get("title"), "text": s.get("text")[:400]} for s in all_sections[:30]]}
"""

#  Gọi API Gemini
print(" Đang phân tích bằng Gemini...")
response = model.generate_content(prompt)
# Lưu output vào file JSON
try:
    content = response.text
    start = content.find("{")
    end = content.rfind("}") + 1
    json_str = content[start:end]
    graph_data = json.loads(json_str)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f" Đã tạo graph thành công: {GRAPH_PATH}")
except Exception as e:
    print(" Lỗi khi phân tích kết quả:", e)
    print("Kết quả trả về:")
    print(response.text)
