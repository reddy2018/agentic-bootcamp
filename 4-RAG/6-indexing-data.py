from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

load_dotenv()
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
loader = PyPDFLoader("ansible.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
split_docs = text_splitter.split_documents(docs)
print(f"Total Chunks: {len(split_docs)}")
print (split_docs[0].page_content)

for s, n in zip(split_docs, range(len(split_docs))):
    print(f"chunk {n}: {s.page_content} --> len is {len(s.page_content)}")
    print("\n---\n")
    
# create vector store
vector_store =PGVector.from_documents(
    documents=split_docs, # chunked documents
    embedding=embeddings,
    connection = connection,
)
print("Vector store created successfully.")

# query vector store
query = "What is Ansible?"
results = vector_store.similarity_search(
    query=query,
    k=3  # number of similar documents to retrieve
)
print(f"Top {len(results)} results for query: '{query}'")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    
query = "How to install Ansible?"
results = vector_store.similarity_search(
    query=query,
    k=3  # number of similar documents to retrieve
)
print(f"Top {len(results)} results for query: '{query}'")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)

