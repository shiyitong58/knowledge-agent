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
    # 基础字段兜底
    card.setdefault("topic", "")
    card.setdefault("core_takeaway", "")
    card.setdefault("mental_model", "")
    card.setdefault("why_it_matters", "")
    card.setdefault("boundary", [])
    card.setdefault("connects_to", [])
    card.setdefault("likely_confusion", [])
    card.setdefault("next_best_question", [])
    card.setdefault("related_cards", [])

    # boundary：只保留字符串
    if not isinstance(card["boundary"], list):
        card["boundary"] = []
    card["boundary"] = [x for x in card["boundary"] if isinstance(x, str)]

    # connects_to：只保留合法结构
    fixed_connects = []
    if isinstance(card["connects_to"], list):
        for item in card["connects_to"]:
            if not isinstance(item, dict):
                continue

            topic = item.get("topic")
            relation = item.get("relation", "补充")
            why = item.get("why", "")

            if not isinstance(topic, str) or not topic.strip():
                continue
            if not isinstance(relation, str):
                relation = "补充"
            if not isinstance(why, str):
                why = ""

            fixed_connects.append({
                "topic": topic.strip(),
                "relation": relation.strip() or "补充",
                "why": why.strip()
            })
    card["connects_to"] = fixed_connects

    # likely_confusion：只保留字符串
    if not isinstance(card["likely_confusion"], list):
        card["likely_confusion"] = []
    card["likely_confusion"] = [
        x for x in card["likely_confusion"] if isinstance(x, str)
    ]

    # next_best_question：只保留字符串
    if not isinstance(card["next_best_question"], list):
        card["next_best_question"] = []
    card["next_best_question"] = [
        x for x in card["next_best_question"] if isinstance(x, str)
    ]

    # related_cards：只保留合法结构
    fixed_related = []
    if isinstance(card["related_cards"], list):
        for item in card["related_cards"]:
            if not isinstance(item, dict):
                continue

            topic = item.get("topic")
            relation_type = item.get("relation_type", "语义相似")
            reason = item.get("reason", "")

            if not isinstance(topic, str) or not topic.strip():
                continue
            if not isinstance(relation_type, str):
                relation_type = "语义相似"
            if not isinstance(reason, str):
                reason = ""

            fixed_related.append({
                "topic": topic.strip(),
                "relation_type": relation_type.strip() or "语义相似",
                "reason": reason.strip()
            })
    card["related_cards"] = fixed_related

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

    new_topic_norm = normalize_topic(new_card.get("topic", ""))

    # 🔥 用缓存
    new_emb = new_card.get("embedding")
    if new_emb is None:
        new_emb = get_embedding(build_text(new_card))

    scored = []

    for c in existing_cards:
        topic = c.get("topic", "")
        topic_norm = normalize_topic(topic)

        if topic_norm == new_topic_norm:
            continue

        # 🔥 用缓存
        emb = c.get("embedding")
        if emb is None:
            emb = get_embedding(build_text(c))

        score = cosine_similarity(new_emb, emb)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen = set()

    for score, c in scored:
        topic = c.get("topic", "")
        topic_norm = normalize_topic(topic)

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

    card["embedding"] = get_embedding(build_text(card))
    card["id"] = str(uuid.uuid4())
    card["created_at"] = datetime.now().isoformat(timespec="seconds")

    existing_cards = load_cards()
    related_cards = find_related_cards(card, existing_cards)
    card["related_cards"] = related_cards

    # 🔥 再过一遍，修 related_cards
    card = normalize_card(card)


    print("\n推荐关联：")
    if related_cards:
        for item in related_cards:
            print(f"- {item['topic']}（{item['relation_type']}）")
            print(f"  原因：{item['reason']}")
    else:
        print("- 暂无")

    save_card(card)
    card_to_print = dict(card)
    card_to_print.pop("embedding", None)

    print(json.dumps(card_to_print, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()