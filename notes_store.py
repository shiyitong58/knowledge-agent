import json
import os

FILE_PATH = "outputs/cards.json"


def load_cards():
    if not os.path.exists(FILE_PATH):
        return []

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return []


def save_card(card):
    data = load_cards()
    data.append(card)

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)