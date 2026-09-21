FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi duckdb==0.10.3 uvicorn pydantic httpx

COPY api/ ./api/

EXPOSE 7860

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
