from flask import Flask, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client
import os

app = Flask(__name__)

# 📡 إعداد الاتصال بـ Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.1.149:11434")
ollama_client = Client(host=OLLAMA_HOST)

# إعداد قاعدة البيانات
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_collection(name="quran", embedding_function=embedding_fn)

# 🧠 استرجاع الآيات
def retrieve_ayat(query, n=5):
    results = collection.query(query_texts=[query], n_results=n)
    return [
        {
            "text": results["documents"][0][i],
            "sura_ar": results["metadatas"][0][i].get("sura_ar", ""),
            "sura_id": results["metadatas"][0][i].get("sura_id", ""),
            "ayah": results["metadatas"][0][i].get("ayah", ""),
            "text_ar": results["metadatas"][0][i].get("text_ar", "")
        }
        for i in range(len(results["documents"][0]))
    ]

# 🗣️ إرسال السياق إلى النموذج
def ask_llm(query, ayat):
    context = "\n".join([f"{a['sura_id']}:{a['ayah']} - {a['text_ar']}" for a in ayat])
    prompt = f"""
أعد فقط الآيات التي تتعلق بالسؤال التالي دون تفسير أو شرح، مستخدمًا الترتيب ورقم السورة والآية كما هو:

السؤال: {query}

الآيات:
{context}
""".strip()

    response = ollama_client.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# 🚀 نقطة النهاية API
@app.route("/search", methods=["GET"])
def search_quran():
    query = request.args.get("q", "")
    print(f"🔍 استعلام البحث: {query}")
    limit = int(request.args.get("limit", 5))

    if not query:
        return jsonify({"error": "يرجى إدخال استعلام."}), 400

    ayat = retrieve_ayat(query, n=limit)
    result = ask_llm(query, ayat)
    return jsonify({
        "query": query,
        "matched_verses": ayat,
        "llm_output": result
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7000)
