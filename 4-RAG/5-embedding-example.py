from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
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
    
embeddings = embeddings.embed_documents([chunk.page_content for chunk in split_docs])
for e, i in zip(embeddings, range(len(embeddings))):
    print(f"Embedding {i+1}: {e[:10]} --> len is {len(e)}")
    print("\n---\n")