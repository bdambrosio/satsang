# Docker Deployment Guide

This guide explains how to build and run the Ramana API server using Docker.

## Prerequisites

- Docker installed
- OpenRouter API key (get one at https://openrouter.ai)

## Quick Start

### Option 1: Pull Pre-built Image from GitHub Container Registry

If the image has been built and pushed to GitHub Container Registry:

```bash
# Pull the image (replace OWNER/REPO with your GitHub username/repo)
docker pull ghcr.io/OWNER/REPO/ramana_api:latest

# Run the container
docker run -d \
  --name ramana-api \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY="your-api-key-here" \
  -v $(pwd)/sessions:/app/sessions \
  ghcr.io/OWNER/REPO/ramana_api:latest
```

### Option 2: Build Locally

1. **Build the Docker image:**
   ```bash
   docker build -t ramana-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name ramana-api \
     -p 5001:5001 \
     -e OPENROUTER_API_KEY="your-api-key-here" \
     -v $(pwd)/sessions:/app/sessions \
     ramana-api
   ```

The container defaults to:
- Model: `qwen/qwen3-vl-235b-a22b-instruct`
- Port: `5001`
- Backend: `openrouter`

3. **Access the website:**
   Open http://localhost:5001 in your browser.

## Overriding Defaults

To use a different model or port, override the CMD:

```bash
docker run -d \
  --name ramana-api \
  -p 8080:8080 \
  -e OPENROUTER_API_KEY="your-api-key-here" \
  -v $(pwd)/sessions:/app/sessions \
  ramana-api \
  python ramana_api.py \
    --llm-backend openrouter \
    --llm-url https://openrouter.ai/api/v1 \
    --llm-model "your-model-name" \
    --host 0.0.0.0 \
    --port 8080
```

## What's Included

The Docker image includes:
- `ramana_api.py` - Main Flask application
- `src/` - Required source modules (aliveness_critic, contemplative_rag, filtered_passages_rag)
- `templates/index.html` - Frontend HTML
- `static/style.css` - Stylesheet
- `ramana/nan-yar.txt` - Nan_Yar passages
- `ramana/Commentaries_qa_excert.txt` - Commentaries Q&A corpus
- `filtered_guten/filtered_passages/corpus.jsonl` - Filtered passages corpus

The `sessions/` directory is mounted as a volume to persist conversation history across container restarts.

## Managing the Container

### Viewing Logs

```bash
docker logs -f ramana-api
```

### Stopping the Container

```bash
docker stop ramana-api
docker rm ramana-api
```

### Restarting the Container

```bash
docker restart ramana-api
```

## Production Considerations

1. **HTTPS/SSL**: The container serves HTTP only. For HTTPS in production:
   - **Recommended**: Use your cloud provider's load balancer or a reverse proxy (nginx, Caddy, Traefik) to handle SSL/TLS termination
   - The container runs on HTTP internally (port 5001), and the proxy forwards HTTPS traffic to it
   - This is the standard production pattern and avoids managing certificates in the container

2. **Environment Variables**: Set `OPENROUTER_API_KEY` via your cloud provider's secrets management (environment variables).

3. **Session Persistence**: The `sessions/` volume ensures conversation history persists. Consider backing this up or using a database for production.

4. **Resource Limits**: The container uses sentence-transformers and FAISS, which load models into memory. Ensure adequate RAM (recommend at least 2GB).

## Troubleshooting

**Container fails to start:**
- Check that `OPENROUTER_API_KEY` is set correctly
- Verify `corpus.jsonl` exists in `filtered_guten/filtered_passages/`
- Check logs: `docker logs ramana-api`

**"No passages loaded" warning:**
- Ensure `filtered_guten/filtered_passages/corpus.jsonl` exists and is included in the build
- Run `filter_passages.py` locally first to generate the corpus

**OpenRouter API errors:**
- Verify your API key is valid and has credits
- Check the model name is correct (default: `qwen/qwen3-vl-235b-a22b-instruct`)
