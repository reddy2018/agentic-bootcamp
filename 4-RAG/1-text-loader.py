from langchain_community.document_loaders import TextLoader

loader = TextLoader("example.txt")
docs = loader.load()
print(docs)
