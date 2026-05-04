import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator

from ollama._client import AsyncClient

app = FastAPI(title="Ollama Rotating Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We use the AsyncClient we already built, pointing it to the cloud.
# The rotating key logic is already built into the AsyncClient!
client = AsyncClient(host="https://ollama.com")

async def _stream_generator(stream_iterator) -> AsyncGenerator[str, None]:
    async for chunk in stream_iterator:
        # Handle both dicts and Pydantic models
        if hasattr(chunk, 'model_dump'):
            data = chunk.model_dump(exclude_none=True)
        elif hasattr(chunk, 'dict'):
            data = chunk.dict(exclude_none=True)
        else:
            data = dict(chunk)
        yield json.dumps(data) + "\n"

@app.post("/api/chat")
async def proxy_chat(request: Request):
    body = await request.json()

    # We must explicitly handle the stream flag
    stream = body.get("stream", False)

    # We remove 'stream' from body since we handle it separately
    kwargs = body.copy()
    if 'stream' in kwargs:
        del kwargs['stream']

    try:
        response = await client.chat(**kwargs, stream=stream)

        if stream:
            return StreamingResponse(_stream_generator(response), media_type="application/x-ndjson")
        else:
            if hasattr(response, 'model_dump'):
                return response.model_dump(exclude_none=True)
            elif hasattr(response, 'dict'):
                return response.dict(exclude_none=True)
            return dict(response)

    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/generate")
async def proxy_generate(request: Request):
    body = await request.json()

    stream = body.get("stream", False)
    kwargs = body.copy()
    if 'stream' in kwargs:
        del kwargs['stream']

    try:
        response = await client.generate(**kwargs, stream=stream)

        if stream:
            return StreamingResponse(_stream_generator(response), media_type="application/x-ndjson")
        else:
            if hasattr(response, 'model_dump'):
                return response.model_dump(exclude_none=True)
            elif hasattr(response, 'dict'):
                return response.dict(exclude_none=True)
            return dict(response)

    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/tags")
async def proxy_tags():
    try:
        response = await client.list()
        if hasattr(response, 'model_dump'):
            return response.model_dump(exclude_none=True)
        elif hasattr(response, 'dict'):
            return response.dict(exclude_none=True)
        return dict(response)
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})

def serve(host: str = "127.0.0.1", port: int = 11434):
    """
    Starts the local proxy server.
    """
    print(f"Starting rotating-key proxy server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
