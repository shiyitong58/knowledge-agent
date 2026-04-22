import argparse
import json
import sys
import uuid
from datetime import datetime
from typing import Any

from openai import OpenAI
from notes_store import save_card, load_cards


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
    return topic.lower().replace("（", "(").replace("）", ")").strip()

def build_knowledge_card(text: str) -> dict[str, Any]:
    client = OpenAI(base_url="https://api.deepseek.com")
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


def find_related_cards(new_card: dict[str, Any], existing_cards: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not existing_cards:
        return []
    
    # 🔥 过滤掉“同 topic 的卡片”（避免自己推荐自己）
    existing_cards = [
        c for c in existing_cards
        if normalize_topic(c.get("topic", "")) != normalize_topic(new_card.get("topic", ""))
    ]
    sample_cards = [
        {
            "topic": c.get("topic", ""),
            "core_takeaway": c.get("core_takeaway", ""),
            "why_it_matters": c.get("why_it_matters", "")
        }
        for c in existing_cards[-5:]
    ]

    new_card_simple = {
        "topic": new_card.get("topic", ""),
        "core_takeaway": new_card.get("core_takeaway", ""),
        "why_it_matters": new_card.get("why_it_matters", "")
    }

    client = OpenAI(base_url="https://api.deepseek.com")

    prompt = f"""
你是一个知识结构分析助手。

任务：
找出最相关的 1~3 个已有卡片，并说明为什么相关。

强制要求：
1. 只返回 JSON
2. 不要输出任何解释文字
3. 不要使用 markdown
4. 如果没有合适的，返回：{{"related_cards": []}}

输出格式：
{{
  "related_cards": [
    {{
      "topic": "卡片主题",
      "reason": "为什么相关",
      "relation_type": "补充/对比/上下位关系"
    }}
  ]
}}

新卡片：
{new_card_simple}

已有卡片：
{sample_cards}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一个严格输出 JSON 的助手，不允许输出任何解释或多余文本。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=250,
    )

    content = response.choices[0].message.content
    if not content:
        return []

    json_str = extract_json(content)
    if not json_str:
        return []

    try:
        parsed = json.loads(json_str)
        results = parsed.get("related_cards", [])

# 🔥 第1步：过滤掉和新卡片相同的 topic
        filtered = [
            item for item in results
            if normalize_topic(item.get("topic", "")) != normalize_topic(new_card.get("topic", ""))
        ]

# 🔥 第2步：去重（防止 MCP 两种写法）
        seen = set()
        unique = []

        for item in filtered:
            key = normalize_topic(item.get("topic", ""))
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique
    except json.JSONDecodeError:
        return [
            {
                "topic": "解析失败",
                "reason": f"无法解析 JSON：{content[:100]}",
                "relation_type": "error"
            }
        ]


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