from vector_store import retrieve_documents_with_scores

query = "What is  Agentic AI?"
results = retrieve_documents_with_scores(query, k=3, )
print("Top 3 relevant documents with similarity scores:")
for result in results:
    print(f"Document: {result[0]}\nScore: {result[1]}\n")
    
# results filtering based on a similarity score threshold
# if similarity score is less than 0.75, we can ignore that document
# MMR
