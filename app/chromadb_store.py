import os
import openai
from dotenv import load_dotenv
from chromadb import PersistentClient

# Încarcă cheia OpenAI din fișierul .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Calea unde vor fi salvați vectorii
CHROMA_PATH = "chroma"

# Trimite textul la OpenAI și primește embedding
def get_embedding(text: str) -> list:
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Încarcă rezumatele din fișierul text
def load_book_summaries(file_path=None) -> list:
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "book_summaries.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    books = content.strip().split("## Title: ")[1:]
    book_data = []
    for book in books:
        lines = book.strip().split("\n", 1)
        title = lines[0].strip()
        summary = lines[1].strip().replace("\n", " ")
        book_data.append((title, summary))
    return book_data

# Creează colecția și încarcă datele în Chroma
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def initialize_chroma():
    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    client = PersistentClient(path=CHROMA_PATH)

    # Șterge colecția existentă dacă e cazul
    if "books" in [c.name for c in client.list_collections()]:
        client.delete_collection("books")

    # Creează colecția cu funcția corectă de embedding
    collection = client.create_collection(
        name="books",
        embedding_function=embedding_function
    )

    for idx, (title, summary) in enumerate(load_book_summaries()):
        embedding = get_embedding(summary)
        collection.add(
            ids=[f"book_{idx}"],
            documents=[summary],
            metadatas=[{"title": title}],
            embeddings=[embedding]
        )
        print(f"Added: {title}")

    print("ChromaDB data loaded.")

# Punct de intrare
if __name__ == "__main__":
    initialize_chroma()
