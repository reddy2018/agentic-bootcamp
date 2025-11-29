import time
import config
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()



REDIS_URL = config.REDIS_URL
embedding_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

# add documents to the vector store
def add_documents(documents: list[str]):
    docs = [Document(page_content=doc) for doc in documents] # converting documents to langchain Document format
    # store documents in Redis vector store
    RedisVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        redis_url=REDIS_URL,
        index_name=config.INDEX_NAME
    )
    
# return the vector store instance
def get_vector_store():
    try:
        return RedisVectorStore(
            embeddings=embedding_model,
            redis_url=REDIS_URL,
            index_name=config.INDEX_NAME
        )
    except Exception as e:
        print(f"Error: {e}")
        return None

# retrieve documents similar to the query 
def retrieve_documents(query: str, k: int = 5):
    # retry logic can be added here if needed
    for i in range(3):
        try:
            vector_store = get_vector_store() # fetch the instance of vector store
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2) # wait before retrying
        
    #vector_store = get_vector_store() # fetch the instance of vector store
    results = vector_store.similarity_search(query, k=k)
    return [d.page_content for d in results] # return only the text content of the documents

# retrieve documents similar to the query along with similarity scores
def retrieve_documents_with_scores(query: str, k: int = 5):
    vector_store = get_vector_store() # fetch the instance of vector store
    results = vector_store.similarity_search_with_score(query, k=k)
    return [(d.page_content, score) for d, score in results] # return text content along with similarity scores