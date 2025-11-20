docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16

# pgvector Docker Container
This Docker container runs a PostgreSQL database with the pgvector extension installed. pgvector is useful for storing and querying vector embeddings, making it ideal for applications involving machine learning and natural language processing.

# https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector