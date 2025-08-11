from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

texts = [d["content"] for d in docs]
metadatas = [{"source": d["source"], "chunk_id": d["chunk_id"]} for d in docs]

db = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory="chroma_db")
db.persist()
