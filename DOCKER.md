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
docker pull ghcr.io/OWNER/REPO/ramana_web:latest

# Run the container
docker run -d \
  --name ramana-api \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY="your-api-key-here" \
  -v $(pwd)/sessions:/app/sessions \
  ghcr.io/OWNER/REPO/ramana_web:latest
```

### Option 2: Build and Push to GitHub Container Registry Manually

1. **Build the Docker image:**
   ```bash
   docker build -t ramana-web .
   ```

2. **Tag the image for GitHub Container Registry:**
   ```bash
   # Replace YOUR_USERNAME and YOUR_REPO with your GitHub username and repository name
   docker tag ramana-web ghcr.io/YOUR_USERNAME/YOUR_REPO/ramana_web:latest
   ```

3. **Authenticate with GitHub Container Registry:**
   
   **Option A: Using Personal Access Token (if available):**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Give it a name (e.g., "Docker Push")
   - Look for **"Packages"** permission (may be under Account or Repository permissions)
   - If you can't find it, try selecting **"repo"** scope (full repository access)
   - Click "Generate token" and copy it
   
   Then login:
   ```bash
   docker login ghcr.io -u YOUR_USERNAME
   # When prompted for password, paste your token
   ```
   
   **Option B: Using GitHub username/password (simpler):**
   ```bash
   docker login ghcr.io -u YOUR_USERNAME
   # When prompted for password, enter your GitHub password
   # If you have 2FA enabled, you'll need to create a token (see Option A)
   ```
   
   **Note**: If you have 2FA enabled, you MUST use a Personal Access Token (Option A).

4. **Push the image:**
   ```bash
   docker push ghcr.io/YOUR_USERNAME/YOUR_REPO/ramana_web:latest
   ```

5. **Make the package public (optional):**
   - Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/pkgs/container/ramana_web
   - Click "Package settings" → "Change visibility" → "Public"

### Option 3: Build Locally (No Push)

1. **Build the Docker image:**
   ```bash
   docker build -t ramana-web .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name ramana-api \
     -p 5001:5001 \
     -e OPENROUTER_API_KEY="your-api-key-here" \
     -v $(pwd)/sessions:/app/sessions \
     ramana-web
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
docker logs -f ramana-web
```

### Stopping the Container

```bash
docker stop ramana-web
docker rm ramana-web
```

### Restarting the Container

```bash
docker restart ramana-web
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
- Check logs: `docker logs ramana-web`

**"No passages loaded" warning:**
- Ensure `filtered_guten/filtered_passages/corpus.jsonl` exists and is included in the build
- Run `filter_passages.py` locally first to generate the corpus

**OpenRouter API errors:**
- Verify your API key is valid and has credits
- Check the model name is correct (default: `qwen/qwen3-vl-235b-a22b-instruct`)
