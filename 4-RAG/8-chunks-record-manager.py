from langchain_core.indexing.base import InMemoryRecordManager
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.indexing import index
from langchain_community.document_loaders import TextLoader
from langchain_classic.indexes import SQLRecordManager
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

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
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
record_manager.create_schema()

cats_loader = TextLoader("cats.txt")
dogs_loader = TextLoader("dogs.txt")
new_source = TextLoader("new_source.txt")
cat_docs = cats_loader.load()
dog_docs = dogs_loader.load()
new_source_docs = new_source.load()

# chunk the documents before indexing
cat_docs = splitter.split_documents(cat_docs)
dog_docs = splitter.split_documents(dog_docs)
new_source_docs = splitter.split_documents(new_source_docs)

# display output of documents loaded
print(cat_docs)
print(dog_docs)
print(new_source_docs)

# index the documents
index_1 = index(cat_docs, record_manager=record_manager, vector_store=vector_store, cleanup="incremental", source_id_key="source")
# 1st iteration will add new documents to the vector store
print("Indexing cats documents...", index_1)

index_2 = index(dog_docs, record_manager=record_manager, vector_store=vector_store, source_id_key="source")
# 2nd iteration will add new documents to the vector store
print("Indexing dogs documents...", index_2)

index_3 = index(cat_docs, record_manager=record_manager, vector_store=vector_store, source_id_key="source")
# 3rd iteration will not add any documents as they already exist in the vector store
print("Re-indexing cats documents (should be no changes)...", index_3)

index_4 = index(new_source_docs, record_manager=record_manager, vector_store=vector_store, source_id_key="source")
# 4th iteration will add new documents to the vector store
print("Indexing new source documents...", index_4)

# modify a document in new_source_docs
new_source_docs[0].page_content = "This is a modified document about turtles."
index_5 = index(new_source_docs, record_manager=record_manager, vector_store=vector_store, source_id_key="source")
# 5th iteration will update the modified document in the vector store
print("Re-indexing modified new source documents...", index_5)
