import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI
from notes_store import save_card



def read_input_text() -> str:
    parser = argparse.ArgumentParser(
        description="Turn text into a structured knowledge card JSON."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to summarize. If omitted, the script reads from standard input.",
    )
    args = parser.parse_args()

    if args.text:
        return args.text.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("Enter text: ").strip()


def build_knowledge_card(text: str) -> dict[str, Any]:
    client = OpenAI(
        base_url="https://api.deepseek.com",
    )
    model = "deepseek-chat"

    system_prompt = """

You are an expert in AI agent systems, especially LLM-based agents.

Your task is to convert user input into a structured knowledge card.

IMPORTANT:
- Interpret all concepts in the context of modern AI systems and LLM agents
- Avoid generic textbook definitions if a more specific meaning exists
- Preserve the original intent of the user
- Do not hallucinate

Return ONLY valid json.
The output json must follow this format exactly:
{
  "topic": "string",
  "summary": "string",
  "keywords": ["string", "string"],
  "suggested_category": "string",
  "deepen": ["string", "string"]
}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        max_tokens=512,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The LLM returned an empty response.")

    return json.loads(content)




def main() -> None:
    text = read_input_text()
    if not text:
        raise ValueError("Input text cannot be empty.")

    card = build_knowledge_card(text)
    save_card(card)
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
