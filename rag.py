import os

from models import RetrievedChunk
from summarize import build_rag_context, build_extractive_answer


def generate_answer(question: str, results: list[RetrievedChunk]) -> str:
    if not results:
        return "I could not find relevant information in the knowledge base."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return build_extractive_answer(results)

    from openai import OpenAI

    response = OpenAI(api_key=api_key).chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied evidence. If it does not "
                    "answer the question, say so. Cite evidence IDs in brackets."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved evidence:\n{build_rag_context(results)}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The configured model returned an empty answer.")
    return content
