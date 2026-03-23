import os
import requests


def get_answer(question: str, context_chunks: list[str]) -> str:
    """
    Send the question + retrieved context to an LLM and return the answer.
    Uses OpenAI's API — swap for any other provider easily.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ No GROQ_API_KEY found in .env file. Please add it and restart."

    context = "\n\n---\n\n".join([c[:500] for c in context_chunks])

    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.

If the answer is not in the context, say "I couldn't find that in the document."
Be concise and direct.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        return f"⚠️ API call failed: {str(e)} | Response: {e.response.text if hasattr(e, 'response') else 'no response'}"