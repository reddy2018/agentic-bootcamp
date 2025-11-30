# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles

# from main import run_rag_pipeline

# app = FastAPI()

# # Serve static files folder (contains index.html)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def serve_ui():
#     """Serve HTML chat UI instead of JSON."""
#     with open("static/index.html", "r") as f:
#         return HTMLResponse(f.read())

# @app.post("/rag")
# async def rag_endpoint(request: Request):
#     data = await request.json()
#     question = data.get("question", "")

#     if not question:
#         return JSONResponse({"error": "Question is required"}, status_code=400)

#     answer = run_rag_pipeline(question)
#     return {"answer": answer}
