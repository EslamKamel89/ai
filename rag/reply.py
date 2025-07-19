from flask import Flask, request, jsonify
import gc
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
OLLAMA_URL = "http://192.168.1.149:11434/api/chat"

def load_rules_from_file(file_path="quranic_rules_summary.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            rules = file.read().strip()
        return rules
    except FileNotFoundError:
        return "⚠️ ملف القواعد غير موجود!"

def build_prompt(chunks, rules, question):
    context = "\n".join(chunks)
    return f"""
# القواعد:
أجب من السياق فقط ولا تخترع إجابات. اتبع القواعد التالية بدقة:
النص المرفق يحتوى question  و answer_idوid
أرجع بسؤال واحد فقط و نسبة التقارب  و answer_id ,id فقط ولا ترد على الأسئلة التي لا تتعلق بالقواعد القرآنية.
in json array

# السياق:
{context}

# السؤال:
{question}
""".strip()

@app.route('/ask', methods=['POST'])
def ask():
    try:
        gc.collect()  # Collect garbage to free memory
        print("🔹 Garbage collection triggered")

        data = request.json
        full_text = data.get("text", "")
        print("🔹 Received text:", full_text)
        question = data.get("question", "")
        print("🔹 Question:", question)
        rules = load_rules_from_file()
        # print("🔹 Rules:", rules)
        model = data.get("model", "gemma3:12b")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.create_documents([full_text])
        chunk_texts = [doc.page_content for doc in chunks]

        prompt = build_prompt(chunk_texts[:1000], rules, question)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "أجب من النص المرفق فقط أجب حسب القواعد بدقة ولا تخترع"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "top_p": 0.0,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        if response.ok:
            result = response.json()
            return jsonify({"answer": result["message"]["content"].strip()})
        else:
            return jsonify({"error": f"Ollama Error {response.status_code}: {response.text}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
