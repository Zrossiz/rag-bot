import json
import os

with open("terms_map.json", "r", encoding="utf-8") as f:
    terms = json.load(f)

os.makedirs("knowledge_base", exist_ok=True)

for file in os.listdir("raw_pages"):
    with open(f"raw_pages/{file}", "r", encoding="utf-8") as f:
        text = f.read()
    for old, new in terms.items():
        text = text.replace(old, new)
    with open(f"knowledge_base/{file}", "w", encoding="utf-8") as f:
        f.write(text)
