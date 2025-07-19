# file: main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_URL = "http://192.168.1.149:11434/api/generate"  # عدل إذا كانت Ollama تعمل على IP مختلف

class RewriteRequest(BaseModel):
    text: str

@app.post("/rewrite")
def rewrite_text(req: RewriteRequest):
    prompt = f"""
أنت كاتب لغوي ماهر متخصص في تنسيق النصوص القرآنية واللغوية بطريقة منظمة دون تغيير المعنى الأصلي.

- لا تحذف أي فقرة.
- لا تغيّر مضمون النص.
- فقط نظّم الشكل والتسلسل بحيث يسهل قراءته.
- استخدم عناوين فرعية عند الحاجة (مثل: الزنا، الفاحشة، الإفك...).
- افصل بين الآيات والشرح بوضوح.
- اجعل كل وحدة موضوعية مرتبة تحت عنوان مناسب.

ابدأ الآن في إعادة صياغة هذا النص مع الحفاظ الكامل على المعنى:\n\n{req.text}
"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "gemma:12b",
            "prompt": prompt,
            "temperature": 0.3,
            "top_p": 0.9,
            "stream": False
        })

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama Error: {response.text}")

        return {"rewritten_text": response.json()["response"]}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))
