from flask import Flask, request, jsonify
import requests
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gc
import uuid
import re
import sqlite3
from bs4 import BeautifulSoup
import traceback

app = Flask(__name__)
OLLAMA_URL = "http://192.168.1.149:11434/api/chat"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "live_context"
SQLITE_DB_PATH = "./chroma.sqlite3"  # ✅ قاعدة البيانات الخارجية المراد تفريغها

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ✅ دالة تفريغ جميع جداول قاعدة بيانات SQLite
def truncate_all_sqlite_tables(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table,) in tables:
            if table != "sqlite_sequence":
                print(f"🗑️ Truncating table: {table}")
                cursor.execute(f'DELETE FROM "{table}";')

        conn.commit()
        conn.close()
        print("✅ All tables truncated.")
    except Exception as e:
        print(f"❌ Error truncating SQLite tables: {e}")

def reset_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # حذف المجموعة القديمة
        for col in chroma_client.list_collections():
            if col.name == COLLECTION_NAME:
                chroma_client.delete_collection(COLLECTION_NAME)
                print(f"🗑️ Collection '{COLLECTION_NAME}' deleted.")

        return chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except Exception as e:
        print(f"❌ Error during reset_collection: {e}")
        raise e

def load_rules_from_file(file_path="quranic_rules_summary.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "⚠️ ملف القواعد غير موجود!"

def build_prompt(chunks, rules, question):
    context = "\n".join(chunks)
    final_prompt = f"""
# القواعد:
{rules}

# السياق:
{context}

# السؤال:
{question}
""".strip()

    # 🧾 حفظ prompt للمراجعة
    with open("debug_prompt.txt", "w", encoding="utf-8") as f:
        f.write(final_prompt)

    return final_prompt

def remove_english_but_keep_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    english_pattern = re.compile(r'[a-zA-Z0-9.,;:!?()\'"\\+=@#$%^&*<>_/|-]+')
    for element in soup.find_all(string=True):
        if element.parent.name not in ['script', 'style']:
            element.replace_with(re.sub(english_pattern, '', element))
    return str(soup)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # ✅ تفريغ جميع جداول SQLite الخارجية
        truncate_all_sqlite_tables(SQLITE_DB_PATH)

        data = request.json
        full_text = data.get("text", "").strip()
        question = data.get("question", "").strip()
        model = data.get("model", "gemma3:12b")
        rules = load_rules_from_file()

        print("📥 Received text length:", len(full_text))
        print("🧠 Received question:", question)
        print("📝 Received context:", full_text)
        if len(full_text) < 10:
            print("⚠️ full_text is too short or missing!")
            return jsonify({"error": "⚠️ نص السياق غير كافٍ"}), 400

        collection = reset_collection()

        splitter = RecursiveCharacterTextSplitter(chunk_size=5120, chunk_overlap=640)
        chunks = splitter.create_documents([full_text])
        chunk_texts = [doc.page_content for doc in chunks]
        print(f"📦 Total chunks created: {len(chunk_texts)}")

        if not chunk_texts:
            return jsonify({"error": "⚠️ لم يتمكن النظام من تقسيم النص إلى مقاطع."}), 400

        collection.add(
            documents=chunk_texts,
            metadatas=[{"index": i} for i in range(len(chunk_texts))],
            ids=[f"chunk-{i}" for i in range(len(chunk_texts))]
        )

        results = collection.query(query_texts=[question], n_results=100)
        top_chunks = results["documents"][0]
        print(f"🔍 Retrieved {len(top_chunks)} top chunks for the question.")

        if not top_chunks:
            return jsonify({"error": "⚠️ لم يتم العثور على مقاطع مناسبة للإجابة."}), 400

        prompt = build_prompt(top_chunks, rules, question)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "أجب حسب القواعد بدقة ولا تخترع"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "top_p": 0.0,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.ok:
            result = response.json()
            message = result.get("message", {})
            content = message.get("content", "")
            cleaned_answer = content.strip()
            print("📤 Answer length:", len(cleaned_answer))
            print("✅ Answer successfully generated.")
            return jsonify({"answer": cleaned_answer})
        else:
            return jsonify({"error": f"Ollama Error {response.status_code}: {response.text}"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000, debug=False, use_reloader=False)
