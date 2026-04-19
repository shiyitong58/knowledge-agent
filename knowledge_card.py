import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI
from notes_store import save_card
from notes_store import load_cards


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


def extract_json(text: str):
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def build_knowledge_card(text: str) -> dict[str, Any]:
    client = OpenAI(
        base_url="https://api.deepseek.com",
    )
    model = "deepseek-chat"
    
    system_prompt = """
    你是一个帮助用户构建 AI / Agent 知识体系的中文学习助手。

    你的任务不是给出泛泛的百科定义，而是帮助用户真正理解一个概念在整个知识体系中的位置。

    要求：
1. 所有输出必须使用简体中文。
2. 不要只回答“它是什么”，还要回答：
   - 为什么会有它
   - 它在系统里起什么作用
   - 它和哪些概念有关
   - 它最容易和什么搞混
3. 解释必须贴近 AI / LLM / Agent 语境，不要跑到无关领域。
4. 输出要有“帮助继续学习”的价值，而不是只下定义。
5. 只返回合法 JSON，不要输出任何额外说明。

输出格式：
{
  "topic": "主题",
  "what_it_is": "它是什么",
  "why_it_exists": "为什么会有这个东西",
  "what_it_does": "它在系统里起什么作用",
  "related_to": ["相关概念1", "相关概念2"],
  "common_confusions": ["容易混淆点1", "容易混淆点2"],
  "my_next_question": ["建议继续追问的问题1", "建议继续追问的问题2"]
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



def find_related_card(new_card, existing_cards):
    if not existing_cards:
        return {"related": []}


    sample_cards = [
    {
        "topic": c.get("topic", ""),
        "keywords": c.get("keywords", [])
    }
    for c in existing_cards[-5:]
]
    new_card_simple = {
    "topic": new_card.get("topic"),
    "keywords": new_card.get("keywords", [])
}
    client = OpenAI(base_url="https://api.deepseek.com")

    prompt = f"""
你是一个知识结构分析助手。

任务：
找出最相关的 1~3 个已有卡片，并说明为什么相关。

⚠️ 强制要求：
1. 只返回 JSON
2. 不要输出任何解释文字
3. 不要使用 markdown
4. 如果没有合适的，返回：{{"related": []}}

输出格式：
{{
  "related": [
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
        "content": "你是一个严格输出JSON的助手，不允许输出任何解释或多余文本。"
    },
    {"role": "user", "content": prompt}
],
        response_format={"type": "json_object"},
        max_tokens=200
    )

    content = response.choices[0].message.content

    if not content:
        return {"related": []}

    json_str = extract_json(content)
    if not json_str:
        return{"related": []}
    # 🔥 核心：容错解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "related": [
                {
                    "topic": "解析失败",
                    "reason": f"无法解析JSON：{content[:100]}",
                    "relation_type": "error"
                }
            ]
        }


def main() -> None:
    text = read_input_text()
    if not text:
        raise ValueError("Input text cannot be empty.")


    card = build_knowledge_card(text)

    existing_cards = load_cards()
    related = find_related_card(card, existing_cards)
    card["related"] = related
    
    print("\n推荐关联：")

    if related and "related" in related:
        for item in related["related"]:
            print(f"- {item['topic']}（{item['relation_type']}）")
            print(f"  原因：{item['reason']}")
    save_card(card)
    print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
