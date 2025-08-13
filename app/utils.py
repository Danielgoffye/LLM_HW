import os

def get_summary_by_title(title: str, file_path=None) -> str:
    """
    Searches for an exact title in the book_summaries.txt file and returns the full summary.
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "book_summaries.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"File '{file_path}' not found."
    sections = content.strip().split("## Title: ")[1:]
    for section in sections:
        header, summary = section.strip().split("\n", 1)
        if header.strip().lower() == title.strip().lower():
            return summary.strip()
    return f"No summary found for '{title}'."
