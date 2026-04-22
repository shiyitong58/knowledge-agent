import argparse
import json
import sys
import uuid
import os
import re
from datetime import datetime
from typing import Any

from openai import OpenAI
from notes_store import save_card, load_cards
from embedding_utils import get_embedding, cosine_similarity

def read_input_text() -> str:
    parser = argparse.ArgumentParser(
        description="Turn text into a structured learning card JSON."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze. If omitted, the script reads from standard input.",
    )
    args = parser.parse_args()

    if args.text:
        return args.text.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("Enter text: ").strip()


def extract_json(text: str):
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def normalize_card(card: dict[str, Any]) -> dict[str, Any]:
    fixed = []
    for item in card.get("connects_to", []):
        if isinstance(item, dict) and "topic" in item:
            fixed.append(item)

    card["connects_to"] = fixed

    card.setdefault("mental_model", "")
    card.setdefault("boundary", [])
    card.setdefault("likely_confusion", [])
    card.setdefault("next_best_question", [])

    return card


def normalize_topic(topic: str) -> str:
    topic = topic.lower().strip()
    topic = topic.replace("（", "(").replace("）", ")")
    topic = re.sub(r"\s+", " ", topic)

    # 去掉括号内容，保留主干概念
    topic_main = re.sub(r"\(.*?\)", "", topic).strip()

    return topic_main

def build_text(card):
    return f"""
主题: {card.get("topic", "")}
核心: {card.get("core_takeaway", "")}
意义: {card.get("why_it_matters", "")}
"""



def build_knowledge_card(text: str) -> dict[str, Any]:
    # client = OpenAI(base_url="https://api.deepseek.com")
    client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
    model = "deepseek-chat"

    system_prompt = """
你是一个帮助用户构建 AI / Agent 知识体系的中文学习助手。

你的任务不是给出泛泛的百科定义，而是帮助用户真正理解一个概念在整个知识体系中的位置，并推动用户继续思考。

要求：
1. 所有输出必须使用简体中文。
2. 不要只解释“它是什么”，而要帮助用户形成理解和连接。
3. 输出要尽量贴近 AI / LLM / Agent 语境，不要跑到无关领域。
4. 不要泛泛而谈，要尽量有启发感。
5. mental_model 要像一句“帮助理解的类比或直觉解释”。
6. boundary 要写清这个概念不是什么，或不该和什么混淆。
7. 只返回合法 JSON，不要输出任何额外说明。

输出格式：
{
  "topic": "主题",
  "core_takeaway": "一句话抓住这个概念最重要的点",
  "mental_model": "一个帮助理解的类比、直觉解释或口语化理解",
  "why_it_matters": "这个概念为什么值得理解，它在整个知识体系里为什么重要",
  "boundary": [
    "这个概念不是什么",
    "不要和什么混淆"
  ],
  "connects_to": [
    {
      "topic": "相关主题",
      "relation": "补充/对比/纠偏/上下位",
      "why": "为什么和这个主题相关"
    }
  ],
  "likely_confusion": [
    "用户最容易搞混的点1",
    "用户最容易搞混的点2"
  ],
  "next_best_question": [
    "用户下一步最值得追问的问题1",
    "用户下一步最值得追问的问题2"
  ]
}
"""
    

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        max_tokens=700,
    )


    content = response.choices[0].message.content
    json_str = extract_json(content)

    if not json_str:
        raise RuntimeError("No valid JSON found in LLM response.")

    return json.loads(json_str)



def find_related_cards(new_card, existing_cards, top_k=3):
    if not existing_cards:
        return []

    new_topic = normalize_topic(new_card.get("topic", ""))

    new_text = build_text(new_card)
    new_emb = get_embedding(new_text)

    scored = []

    for c in existing_cards:
        topic_norm = normalize_topic(c.get("topic", ""))

        if topic_norm == new_topic:
            continue

        text = build_text(c)
        emb = get_embedding(text)

        score = cosine_similarity(new_emb, emb)

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen = set()
    new_topic_norm = normalize_topic(new_card.get("topic", ""))

    for score, c in scored:
        topic = c.get("topic", "")
        topic_norm = normalize_topic(topic)

    # 1. 去掉和当前新卡片本身重复的概念
        if topic_norm == new_topic_norm:
            continue

    # 2. 去掉召回结果里彼此重复的概念
        if topic_norm in seen:
            continue

        seen.add(topic_norm)

        results.append({
            "topic": topic,
            "relation_type": "语义相似",
            "reason": f"similarity={score:.3f}"
        })

        if len(results) >= top_k:
            break

    return results

    





def main() -> None:
    text = read_input_text()
    if not text:
        raise ValueError("Input text cannot be empty.")

    card = build_knowledge_card(text)
    card = normalize_card(card)

    card["id"] = str(uuid.uuid4())
    card["created_at"] = datetime.now().isoformat(timespec="seconds")

    existing_cards = load_cards()
    related_cards = find_related_cards(card, existing_cards)
    card["related_cards"] = related_cards

    print("\n推荐关联：")
    if related_cards:
        for item in related_cards:
            print(f"- {item['topic']}（{item['relation_type']}）")
            print(f"  原因：{item['reason']}")
    else:
        print("- 暂无")

    save_card(card)
    print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()