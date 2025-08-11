import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ Cheia OpenAI a fost încărcată cu succes.")
else:
    print("❌ Cheia OpenAI NU a fost găsită.")
