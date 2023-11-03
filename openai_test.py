from langchain.embeddings import OpenAIEmbeddings
import os

openai_api_key = os.environ.get('OPENAI_API_KEY', None)
print("embedding begin.. ")
embed = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
doc = embed.embed_documents(["hello, this is a weatehr tool."])
print(doc)
