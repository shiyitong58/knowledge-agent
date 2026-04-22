import json
from collections import OrderedDict

INPUT_FILE = "outputs/cards.json"
OUTPUT_FILE = "outputs/cards_clean.json"


def normalize_topic(topic: str) -> str:
    return topic.lower().replace("（", "(").replace("）", ")").strip()


def load_cards():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cards(cards):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)


def clean_cards(cards):
    dedup = OrderedDict()

    for card in cards:
        topic = card.get("topic", "")
        key = normalize_topic(topic)

        # 👉 保留“最新的”（后出现的覆盖前面的）
        dedup[key] = card

    cleaned = list(dedup.values())

    # 🔥 修字段结构
    for c in cleaned:
        c.setdefault("mental_model", "")
        c.setdefault("boundary", [])
        c.setdefault("connects_to", [])
        c.setdefault("likely_confusion", [])
        c.setdefault("next_best_question", [])
        c.setdefault("related_cards", [])

        # 清 connects_to 脏数据
        c["connects_to"] = [
            item for item in c["connects_to"]
            if isinstance(item, dict) and "topic" in item
        ]

    return cleaned


def main():
    cards = load_cards()
    cleaned = clean_cards(cards)

    print(f"原始卡片数: {len(cards)}")
    print(f"清洗后卡片数: {len(cleaned)}")

    save_cards(cleaned)
    print(f"已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    print("running clean_cards.py")
    main()