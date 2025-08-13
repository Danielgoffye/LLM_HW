import os
import re
import openai
from dotenv import load_dotenv
from langdetect import detect
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from utils import get_summary_by_title
from chromadb_store import load_book_summaries

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
MODEL = "gpt-4-1106-preview"


def search_books(query: str, top_k: int = 3):
    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="books",
        embedding_function=embedding_function
    )
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    matches = []
    for i in range(len(results["ids"][0])):
        title = results["metadatas"][0][i]["title"]
        summary = results["documents"][0][i]
        matches.append((title, summary))
    return matches


def extract_first_title_from_response(response_text: str, titles: list) -> str:
    """
    Attempts to find the first title from the known list that appears in the GPT response.
    It tolerates punctuation and quotation marks around the title.
    """
    for title in titles:
        # Escape any regex characters in the title
        pattern = re.compile(rf"[\"']?{re.escape(title)}[\"']?", re.IGNORECASE)
        if pattern.search(response_text):
            return title
    return None

def is_book_related_question(user_query: str) -> bool:
    # Basic keyword filter to avoid unnecessary GPT calls
    keywords = [
    "carte", "roman", "recomand", "lectură", "literatură", "poveste",
    "book", "novel", "read", "recommend", "story", "adventure", "magic",
    "autor", "scriitor", "literary", "poezie", "ficțiune", "fantasy", "istorie"
]

    if any(word in user_query.lower() for word in keywords):
        return True

    # GPT fallback classifier
    check_prompt = (
        "You are a strict classifier. Your job is to determine if the user's question "
        "is related to books, literature, reading interests, or book recommendations. "
        "The user may speak Romanian or English. "
        "Only respond with 'yes' or 'no'."
    )

    messages = [
        {"role": "system", "content": check_prompt},
        {"role": "user", "content": user_query}
    ]

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip().lower() == "yes"



def translate_text(text: str, target_language: str) -> str:
    if target_language == "en":
        return text
    messages = [
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": f"Please translate the following text into {target_language}:\n\n{text}"}
    ]
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def generate_response(user_query: str, matches: list):
    context = "\n\n".join([f"Title: {title}\nSummary: {summary}" for title, summary in matches])
    language = detect(user_query)
    if language == "ro":
        system_prompt = (
            "Ești un asistent AI care recomandă cărți în funcție de interesele utilizatorului. "
            "Ți se oferă mai jos o listă de una sau mai multe cărți relevante, pe baza întrebării. "
            "Fă o recomandare doar din aceste opțiuni. "
            "Nu presupune că sunt mai multe decât cele listate. "
            "Răspunde în limba română, într-un mod conversațional."
        )
    else:
        system_prompt = (
            "You are an AI assistant that recommends books based on user preferences. "
            "You are given a list of one or more relevant books. "
            "Recommend only from the list. "
            "Do not assume there are more options than provided. "
            "Respond in English, conversationally."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"My question: {user_query}\n\nRelevant books:\n{context}"}
    ]
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def run_chatbot():
    print("Smart Librarian: CLI Mode")
    print("Type 'exit' to quit.\n")
    all_titles = [title for title, _ in load_book_summaries()]
    while True:
        query = input("Ask me for a book recommendation: ").strip()
        lang = detect(query)
        if contains_offensive_language(query):
            print("\n--------------------------------------------------")
            print("Te rog să folosești un limbaj adecvat." if lang == "ro" else "Please use respectful language.")
            print("--------------------------------------------------\n")
            continue
        if query.lower() in {"exit", "quit"}:
            break
        elif not is_book_related_question(query):
            lang = detect(query)
            fallback_msg = (
                "Îți pot răspunde doar la întrebări legate de cărți, literatură sau recomandări de lectură. Te rog reformulează întrebarea."
                if lang == "ro" else
                "I can only respond to questions related to books, literature, or reading recommendations. Please rephrase your question."
            )
            print("\n--------------------------------------------------")
            print(fallback_msg)
            print("--------------------------------------------------\n")
            continue

        matches = search_books(query)
        answer = generate_response(query, matches)
        print("\n" + "-" * 50)
        print(answer)
        recommended_title = extract_first_title_from_response(answer, all_titles)
        if recommended_title:
            full_summary = get_summary_by_title(recommended_title)
            lang = detect(query)
            translated = translate_text(full_summary, target_language="ro" if lang == "ro" else "en")
            print(f"\nDetailed summary for: {recommended_title}\n{translated}")
        else:
            print("\n(No exact title identified for detailed summary.)")
        print("-" * 50 + "\n")


def contains_offensive_language(text: str) -> bool:
    offensive_words = [
        # Romanian
        "prost", "idiot", "bou", "tâmpit", "handicapat", "fraier",
        # English
        "stupid", "idiot", "dumb", "moron", "retard", "loser", "bastard"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in offensive_words)


if __name__ == "__main__":
    run_chatbot()
