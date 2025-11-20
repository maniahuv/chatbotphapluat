import os
import PyPDF2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text):
    import re, unidecode
    # Xóa ký tự đặc biệt, tiêu đề trang, mục lục
    text = re.sub(r'(\n\s*){2,}', '\n', text)
    text = re.sub(r'Page \d+', '', text)
    text = text.strip()
    return text

if __name__ == "__main__":
    os.makedirs(CLEAN_DIR, exist_ok=True)
    for filename in tqdm(os.listdir(RAW_DIR)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, filename)
            txt_name = filename.replace(".pdf", "_clean.txt")
            txt_path = os.path.join(CLEAN_DIR, txt_name)

            raw_text = extract_text_from_pdf(pdf_path)
            clean = clean_text(raw_text)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(clean)
    print(" Đã xử lý xong tất cả file PDF!")

