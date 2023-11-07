from langchain.embeddings import HuggingFaceEmbeddings
import faiss
embedding_size = 384
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings_model = TensorflowHubEmbeddings()

text = "This is a test document."
query_result = embeddings_model.embed_query(text)
doc_result = embeddings_model.embed_documents([text])

print(query_result)
print(doc_result)

index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})