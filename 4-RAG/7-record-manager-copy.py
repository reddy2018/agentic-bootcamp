'''
This script builds a simple RAG-style indexing pipeline that loads text files, 
embeds them with OpenAI embeddings, stores those embeddings in a Postgres PGVector database, 
and uses LangChain’s record-tracking system to decide when a document should be added, updated,
or ignored. First, the code loads environment variables and connects to Postgres, then initializes a 
PGVector store to hold embeddings and an InMemoryRecordManager to track which documents have already been indexed. 
Three text files (cats.txt, dogs.txt, and new_source.txt) are loaded as LangChain Document objects. 
The index() function is then called multiple times to demonstrate how LangChain’s incremental indexing works: 
the first call inserts new cat documents, the second inserts dog documents, and the third sees that the cat documents
are unchanged and performs no update. A fourth call indexes the documents from new_source.txt. Afterward, the script 
modifies cats.txt, reloads it, and runs indexing again; this time the record manager detects that the file’s content 
changed and updates the corresponding embeddings in the vector store. In essence, the script shows how LangChain coordinates
document loading, embedding, vector storage, and change detection so only new or modified documents are re-indexed.
'''


from langchain_core.indexing.base import InMemoryRecordManager
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.indexing import index
from langchain_community.document_loaders import TextLoader
from langchain_classic.indexes import SQLRecordManager

load_dotenv()
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
connection_name = "my_docs"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
namespace = "langchain_rag"

vector_store = PGVector(
    connection=connection,
    embeddings=embedding_model,
    collection_name=connection_name,
    use_jsonb=True
)

record_manager = InMemoryRecordManager(namespace=namespace)

record_manager.create_schema()

cats_loader = TextLoader("cats.txt")
dogs_loader = TextLoader("dogs.txt")
new_source = TextLoader("new_source.txt")

cat_docs = cats_loader.load()
dog_docs = dogs_loader.load()
new_source_docs = new_source.load()

# display output of documents loaded
print(cat_docs)
print(dog_docs)
print(new_source_docs)


# Index initial documents
index_1 = index(cat_docs, record_manager=record_manager, vector_store=vector_store, cleanup="incremental", source_id_key="source")
# 1st iteration will add new documents to the vector store
print("Indexed cat documents:", index_1)
index_2 = index(dog_docs, record_manager=record_manager, vector_store=vector_store,  source_id_key="source")
# 2nd iteration will update the existing documents to the vector store
print("Indexed dog documents:", index_2)
index_3 = index(cat_docs, record_manager=record_manager, vector_store=vector_store, source_id_key="source")
# 3rd iteration will not add any documents as they already exist in the vector store
print("Re-indexed cat documents (should be no changes):", index_3)

index_4 = index(new_source_docs, record_manager=record_manager, vector_store=vector_store, source_id_key="source")
# 4th iteration will add new documents from new_source.txt to the vector store
print("Indexed new source documents:", index_4)


# modify the context of cats.txt 
with open("cats.txt", "at") as f:
    f.write("\nCats are known for their agility and independence.\n")
    
# Reload modified documents
cats_loader = TextLoader("cats.txt")
cat_docs = cats_loader.load()

# Re-index modified documents
index_5 = index(cat_docs, record_manager=record_manager, vector_store=vector_store, source_id_key="source")
# 5th iteration will update the existing cat documents in the vector store
print("Re-indexed modified cat documents:", index_5)
