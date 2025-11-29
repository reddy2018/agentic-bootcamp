from vector_store import retrieve_documents
def retrieve_context(query: str, k: int = 5) -> list[str]:
    context = retrieve_documents(query, k)
    # You can add additional processing or filtering of the context here if needed
    # for example, removing duplicates or sorting based on relevance
    # context = list[str]  each element is a chunk from vector store
    return "\n".join(context) # return as a single string