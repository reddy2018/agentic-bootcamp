from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

loader = PyPDFLoader("ansible.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=20,
    length_function=len,
)
split_docs = text_splitter.split_documents(docs)
for i, doc in enumerate(split_docs):
    print(f"--- Document Chunk {i+1} ---")
    print(doc.page_content)