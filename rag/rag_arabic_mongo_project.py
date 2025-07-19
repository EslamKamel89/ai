from flask import Flask, request, jsonify
from pymongo import MongoClient
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import numpy as np
import requests
import os
import gc
import collections.abc

app = Flask(__name__)
OLLAMA_URL = "http://192.168.1.149:11434/api/chat"

client = MongoClient("mongodb://localhost:27017/")
db = client["quran_rag"]
collection = db["docs_chunks"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n", "۔", ".", "؟", "،", " "]  # للغة العربية
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def flatten_embedding(embedding):
    flat = []
    for x in embedding:
        if isinstance(x, (list, np.ndarray, collections.abc.Iterable)) and not isinstance(x, (str, bytes)):
            flat.extend(float(i) for i in x)
        else:
            flat.append(float(x))
    return flat

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def insert_chunks_to_mongo(text, topic="عام", source="وثيقة"):
    chunks = text_splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        embedding_raw = embedding_fn(chunk)[0]
        embedding = flatten_embedding(embedding_raw)
        if len(embedding) < 100 or len(embedding) > 2000:
            print("⚠️ تم تجاهل مقطع بحجم:", len(embedding))
            continue
        docs.append({
            "id": f"chunk_{i}",
            "content": chunk,
            "topic": topic,
            "source": source,
            "embedding": embedding
        })
    if docs:
        collection.insert_many(docs)
        gc.collect()
    return len(docs)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_topic_from_question(question):
    known_topics = db["docs_chunks"].distinct("topic")

    question_emb = flatten_embedding(embedding_fn(question)[0])
    best_topic = None
    best_score = -1
    for topic in known_topics:
        topic_emb = flatten_embedding(embedding_fn(topic)[0])
        sim = cosine_similarity(question_emb, topic_emb)
        print(f"🔍 مقارنة السؤال مع الموضوع '{topic}': {sim:.4f}")
        if sim > best_score:
            best_score = sim
            best_topic = topic
    return best_topic if best_score > 0.3 else None

def get_relevant_chunks(question, k=48, top_final=10, topic=None):
    embedding_raw = embedding_fn(question)[0]
    q_embedding = flatten_embedding(embedding_raw)

    query_filter = {}
    if topic:
        query_filter["topic"] = topic

    regex_matches = list(collection.find(
        {"$and": [query_filter, {"content": {"$regex": question, "$options": "i"}}]} if topic else
        {"content": {"$regex": question, "$options": "i"}},
        {"content": 1, "embedding": 1}
    ))

    all_docs = list(collection.find(query_filter, {"content": 1, "embedding": 1}))
    scored = []
    for doc in all_docs:
        if len(doc["embedding"]) != len(q_embedding):
            continue
        sim = cosine_similarity(q_embedding, doc["embedding"])
        scored.append((sim, doc["content"]))

    top_by_embedding = [x[1] for x in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]
    combined = list(dict.fromkeys([doc["content"] for doc in regex_matches] + top_by_embedding))

    question_keywords = question.split()
    filtered_chunks = []
    for chunk in combined:
        if any(kw in chunk for kw in question_keywords):
            filtered_chunks.append(chunk)

    return filtered_chunks[:top_final]

def build_prompt(chunks, rules, question):
    context = "\n".join(chunks)
    return f"""# القواعد:
{rules}

# السؤال:
{question}

# السياق المستخرج من التفسير:
{context}

⛔️ ملاحظات:
- لا تُجب إلا عن السؤال فقط.
- لا تُكرر مفاهيم غير مطلوبة.
- لا تُضيف تعريفات أو شروح زائدة من السياق.

✅ أجب من النص فقط وبدقة.
"""

def query_gemma(prompt):
    try:
        payload = {
            "model": "gemma3:12b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9
        }
        res = requests.post(OLLAMA_URL, json=payload)
        print("🔹 Ollama response:", res)
        res.raise_for_status()
        return res.json()["message"]["content"]
    except Exception as e:
        print("❌ Error during Gemma call:", str(e))
        return "⚠️ حدث خطأ أثناء الاتصال بالنموذج."

def load_rules():
    try:
        with open("rules.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return "أجب من النص فقط. لا تستنتج ولا تخترع."

@app.route("/upload", methods=["POST"])
def upload_doc():
    files = request.files.getlist("files")
    topic = request.form.get("topic", "عام")
    source = request.form.get("source", "وثيقة")

    total_chunks = 0
    for file in files:
        file_path = f"./temp_{file.filename}"
        file.save(file_path)
        text = extract_text_from_docx(file_path)
        count = insert_chunks_to_mongo(text, topic=topic, source=source)
        total_chunks += count
        os.remove(file_path)

    return jsonify({
        "message": f"تم إدخال {len(files)} ملف.",
        "total_chunks_inserted": total_chunks
    })

@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.json.get("question")
    top_k = int(request.json.get("top_k", 48))
    final_k = int(request.json.get("final_k", 10))
    topic = request.json.get("topic")

    if not topic:
        topic = detect_topic_from_question(question)
        print(f"🔍 تم استنتاج الموضوع تلقائيًا: {topic}")

    chunks = get_relevant_chunks(question, k=top_k, top_final=final_k, topic=topic)
    rules = load_rules()
    prompt = build_prompt(chunks, rules, question)
    answer = query_gemma(prompt)

    db["qa_log"].insert_one({
        "question": question,
        "topic": topic,
        "chunks": chunks,
        "prompt": prompt,
        "answer": answer
    })

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
