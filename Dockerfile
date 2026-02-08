# Dockerfile for Ramana API Server
# Uses OpenRouter for LLM (no local vLLM needed)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for sentence-transformers and faiss
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Only runtime dependencies (no training libraries)
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Copy application code
COPY ramana_api.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

# Copy data files
COPY ramana/nan-yar.txt ./ramana/
COPY ramana/Ulladu_Narpadu.txt ./ramana/
COPY ramana/Upadesa_Undiyar.txt ./ramana/
COPY ramana/src/ramana_qa_training.jsonl ./ramana/src/

# Copy filtered passages corpus
COPY filtered_guten/filtered_passages/corpus.jsonl ./filtered_guten/filtered_passages/

# Create sessions directory (runtime data, persisted via volume)
RUN mkdir -p sessions

# Expose port
EXPOSE 5001

# Environment variables (can be overridden at runtime)
ENV FLASK_APP=ramana_api.py
ENV PYTHONUNBUFFERED=1

# Default command: run with OpenRouter backend
# Override with docker run to customize
CMD ["python", "ramana_api.py", \
     "--llm-backend", "openrouter", \
     "--llm-url", "https://openrouter.ai/api/v1", \
     "--llm-model", "qwen/qwen3-vl-235b-a22b-instruct", \
     "--host", "0.0.0.0", \
     "--port", "5001"]
