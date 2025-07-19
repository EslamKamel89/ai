import json
import re
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✏️ إعدادات التقطيع
CHUNK_SIZE = 600
CHUNK_OVERLAP = 75

# 🧹 تنظيف HTML و CSS و JavaScript
def clean_html(raw_html):
    if not isinstance(raw_html, str):
        raw_html = ""
    raw_html = re.sub(r'<script.*?>.*?</script>', '', raw_html, flags=re.DOTALL)
    raw_html = re.sub(r'<style.*?>.*?</style>', '', raw_html, flags=re.DOTALL)
    raw_html = re.sub(r'<[^>]+>', '', raw_html)
    raw_html = re.sub(r'\s+', ' ', raw_html).strip()
    return raw_html

# 🧠 إعداد Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

collection = chroma_client.get_or_create_collection(
    name="quran",
    embedding_function=embedding_fn
)

# 📥 تحميل ملف JSON
with open("data/quran.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🪓 أداة التقطيع
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# ⛓️ تجهيز البيانات
ids = []
documents = []
metadatas = []

for item in data:
    explain_raw = item.get("explain_ar", "")
    if not explain_raw:
        continue

    cleaned_text = clean_html(explain_raw)
    chunks = text_splitter.create_documents([cleaned_text])

    sura_id = item.get("sura_id", "unknown")
    ayah = item.get("ayah", "unknown")

    for i, chunk in enumerate(chunks):
        unique_id = uuid.uuid4().hex[:8] if "unknown" in (str(sura_id), str(ayah)) else str(i)
        doc_id = f"{sura_id}_{ayah}_{unique_id}"

        ids.append(doc_id)
        documents.append(chunk.page_content)
        metadatas.append({
            "sura_id": sura_id,
            "ayah": ayah,
            "text_ar": item.get("text_ar", ""),
            "sura_ar": item.get("sura_ar", "")
        })

        print(f"📥 تم إدخال المقطع {i+1} من الآية {ayah} (السورة {sura_id}) | كلمات: {len(chunk.page_content.split())}")

# 🧾 إدخال في Chroma
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"\n✅ تم إدخال {len(documents)} مقطعًا في Chroma DB بنجاح.")
