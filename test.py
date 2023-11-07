from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import faiss
embedding_size = 384
print("begining...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2",
    cache_folder="/home/cc/workspace/ximu/workspace/yh/Models/all-mpnet-base-v2")
#embeddings_model = TensorflowHubEmbeddings()

text = "This is a test document."
print("input: ",text)
query_result = embeddings_model.embed_query(text)
doc_result = embeddings_model.embed_documents([text])

print(query_result)
print(len(query_result))

index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})