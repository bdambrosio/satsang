# Cloud Server Deployment Guide

Quick guide for deploying the Ramana API container on your cloud server.

## Prerequisites

- Docker installed on your cloud server
- `OPENROUTER_API_KEY` set as an environment variable or secret

## Steps

### 1. Pull the Docker Image

```bash
docker pull ghcr.io/bdambrosio/satsang/ramana_web:latest
```

**Note:** If the package is private, you'll need to authenticate first:
```bash
docker login ghcr.io -u bdambrosio
# Enter your GitHub Personal Access Token when prompted
```

### 2. Run the Container

**Basic command:**
```bash
docker run -d \
  --name ramana-web \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY="your-openrouter-api-key-here" \
  -v ramana-sessions:/app/sessions \
  --restart unless-stopped \
  ghcr.io/bdambrosio/satsang/ramana_web:latest
```

**Using environment variable (recommended):**
```bash
# Set your OpenRouter API key as an environment variable first
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Then run
docker run -d \
  --name ramana-web \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY \
  -v ramana-sessions:/app/sessions \
  --restart unless-stopped \
  ghcr.io/bdambrosio/satsang/ramana_web:latest
```

**Using Docker secrets (if your cloud provider supports it):**
```bash
docker run -d \
  --name ramana-web \
  -p 5001:5001 \
  --secret source=openrouter_api_key,target=/run/secrets/openrouter_api_key \
  -e OPENROUTER_API_KEY_FILE=/run/secrets/openrouter_api_key \
  -v ramana-sessions:/app/sessions \
  --restart unless-stopped \
  ghcr.io/bdambrosio/satsang/ramana_web:latest
```

### 3. Verify It's Running

```bash
# Check container status
docker ps

# View logs
docker logs -f ramana-web

# Check if the API is responding
curl http://localhost:5001/api/nan-yar
```

### 4. Access the Website

- **Local (on server):** http://localhost:5001
- **External:** http://YOUR_SERVER_IP:5001 (if firewall allows)
- **With reverse proxy:** Configure nginx/Caddy to proxy HTTPS → http://localhost:5001

## Container Options Explained

- `-d`: Run in detached mode (background)
- `--name ramana-web`: Name the container for easy reference
- `-p 5001:5001`: Map host port 5001 to container port 5001
- `-e OPENROUTER_API_KEY`: Pass the OpenRouter API key as environment variable
- `-v ramana-sessions:/app/sessions`: Create a Docker volume for session persistence
- `--restart unless-stopped`: Automatically restart container if it crashes or server reboots

## Managing the Container

```bash
# Stop the container
docker stop ramana-api

# Start the container
docker start ramana-web

# Restart the container
docker restart ramana-web

# View logs
docker logs -f ramana-web

# Remove the container (stops it first)
docker rm -f ramana-web

# Update to latest image
docker pull ghcr.io/bdambrosio/satsang/ramana_web:latest
docker stop ramana-web
docker rm ramana-web
# Then run the docker run command again
```

## HTTPS Setup (Recommended for Production)

The container serves HTTP only. Set up a reverse proxy:

**Example nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then use Let's Encrypt/Certbot for SSL certificates.

## Troubleshooting

**Container won't start:**
- Check logs: `docker logs ramana-web`
- Verify `OPENROUTER_API_KEY` is set correctly
- Check if port 5001 is already in use: `netstat -tulpn | grep 5001`

**"No passages loaded" warning:**
- This is normal if `corpus.jsonl` wasn't included in the build
- The app will still work, but sidebar passages won't be available

**OpenRouter API errors:**
- Verify your API key is valid and has credits
- Check the model name is correct (default: `qwen/qwen3-vl-235b-a22b-instruct`)
