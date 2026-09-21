FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install DuckDB extensions
RUN python -c "import duckdb; c=duckdb.connect(); c.execute('INSTALL parquet'); c.execute('INSTALL httpfs'); print('Extensions installed!')"

# Copy app
COPY api/index.py ./api/index.py

EXPOSE 7860

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
