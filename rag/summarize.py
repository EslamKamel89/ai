from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://192.168.1.149:11434/api/chat"  # استخدم /api/chat مع gemma:12b

@app.route('/summarize', methods=['POST'])
def summarize_and_organize():
    data = request.get_json()
    user_text = data.get("text", "")

    system_prompt = (
        "أنت مساعد لغوي ذكي متخصص في النصوص الدينية.\n"
        "- لخص النص التالي بدقة .\n"
        "- أعِد تنظيم الفقرات لتكون واضحة وسهلة القراءة.\n"
        "- لا تضف أي رأي، فقط نظم وقدّم تلخيصًا مبنيًا على النص نفسه.\n"
        "- استخدم تنسيق جيد مثل النقاط والفقرات إن لزم.\n"
    )

    payload = {
        "model": "deepseek-r1:14b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)

        if response.ok:
            result = response.json()
            return jsonify({"summary": result["message"]["content"]})
        else:
            return jsonify({"error": f"Gemma returned {response.status_code}: {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/answer', methods=['POST'])
def answer_from_context():
    data = request.get_json()
    print("🔹 Received data:", data)
    question = data.get("question", "")
    context = data.get("context", "")

    system_prompt = (
        
    )

    print("🔹 System prompt:", system_prompt)
    payload = {
        "model": "gemma3:12b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"السؤال: {question}\n\nالسياق:\n{context}"}
        ],
        "temperature": 0.1,
        "top_p": 0.2,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        print("🔹 Ollama response:", response)
        if response.ok:
            result = response.json()
            return jsonify({"answer": result["message"]["content"].strip()})
        else:
            return jsonify({"error": f"Ollama Error {response.status_code}: {response.text}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
