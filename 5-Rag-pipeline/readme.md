# Setting up Redis for the RAG Pipeline
docker run -d  --name genai-redis  -p 6379:6379 redis/redis-stack-server:latest


# how to execute this project
python main.py --question What is agentic ai?
