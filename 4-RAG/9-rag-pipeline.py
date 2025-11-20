from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

loader = PyPDFLoader("book_manuscript.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
chunked_docs = splitter.split_documents(docs)

# create vector store
vector_store = PGVector.from_documents(
    documents =chunked_docs, # chunks of documents
    embedding = embedding_model,
    connection = connection
)

# create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # k is the number of chunks to retrieve
# create LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."),
    ("user", "summarize the following book on the mentioned query: {input}")
])

output = prompt.invoke({"input": "Summarize the book on the query: what is the main idea of the book? summarize in less than 200 words."})
retrieve = retriever.invoke("what does it mean by monitoring in the book")
print("\nRetrieved Chunks: ", retrieve)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes books."),
    ("user", "Summarize the following book on the mentioned query: {input} and the following context: {context}")
])

response = prompt.invoke({"input":"What does it mean by monitoring in the book?", "context": retrieve})
print("Prompt Output:")
print(response)

# invoke the llm with the prompt
llm_response = llm.invoke(output)
print("LLM Response: ", llm_response)   