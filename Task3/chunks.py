from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, json

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

docs = []
for file in os.listdir("knowledge_base"):
    with open(f"knowledge_base/{file}", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        docs.append({
            "content": chunk,
            "source": file,
            "chunk_id": i
        })

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)
