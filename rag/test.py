import requests

# عنوان FastAPI
API_URL = "http://192.168.1.149:9000/nl-to-sql"  # تأكد أن هذا هو عنوان الجهاز الذي يشغل gemma_api.py

# السؤال التجريبي
question = "Who are the users from Egypt?"

# إرسال الطلب
response = requests.post(API_URL, json={"question": question})

# عرض النتيجة
print("\n=== Raw Response ===")
print(response.text)

if response.ok:
    data = response.json()
    print("\n✅ SQL Output:")
    print(data['sql'])
else:
    print("\n❌ Error occurred:")
    print(response.status_code)
