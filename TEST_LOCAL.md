# Testing Docker Image Locally

Quick guide for testing the Ramana web container on your local machine.

## Prerequisites

- Docker installed
- `OPENROUTER_API_KEY` environment variable or value ready

## Option 1: Build and Run Locally

**1. Build the image:**
```bash
docker build -t ramana-web .
```

**2. Run the container:**
```bash
docker run -d \
  --name ramana-web \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY="your-openrouter-api-key-here" \
  -v ramana-sessions:/app/sessions \
  ramana-web
```

**3. Test it:**
```bash
# Check if container is running
docker ps

# View logs
docker logs -f ramana-web

# Test the API endpoint
curl http://localhost:5001/api/nan-yar

# Open in browser
# http://localhost:5001
```

## Option 2: Pull from GitHub Container Registry

**1. Pull the image (if already pushed):**
```bash
docker pull ghcr.io/bdambrosio/satsang/ramana_web:latest
```

**2. Run the container:**
```bash
docker run -d \
  --name ramana-web \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY="your-openrouter-api-key-here" \
  -v ramana-sessions:/app/sessions \
  ghcr.io/bdambrosio/satsang/ramana_web:latest
```

**3. Test it (same as Option 1):**
```bash
docker ps
docker logs -f ramana-web
curl http://localhost:5001/api/nan-yar
# Open http://localhost:5001 in browser
```

## Using Environment Variable

Instead of hardcoding the API key, you can use an environment variable:

```bash
# Set it first
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Then run (note: no quotes around $OPENROUTER_API_KEY)
docker run -d \
  --name ramana-web \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY \
  -v ramana-sessions:/app/sessions \
  ramana-web
```

## Testing the Website

Once running, test these endpoints:

**1. Homepage:**
```bash
curl http://localhost:5001/
# Or open http://localhost:5001 in your browser
```

**2. Nan_Yar passage:**
```bash
curl http://localhost:5001/api/nan-yar
```

**3. Query endpoint (POST):**
```bash
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Self?"}'
```

## Viewing Logs

```bash
# Follow logs in real-time
docker logs -f ramana-web

# View last 50 lines
docker logs --tail 50 ramana-web

# View logs with timestamps
docker logs -t ramana-web
```

## Stopping and Cleaning Up

```bash
# Stop the container
docker stop ramana-web

# Remove the container
docker rm ramana-web

# Remove the image (if you want to start fresh)
docker rmi ramana-web

# Remove the volume (if you want to clear session data)
docker volume rm ramana-sessions
```

## Troubleshooting

**Container won't start:**
```bash
# Check what went wrong
docker logs ramana-web

# Try running in foreground to see errors
docker run --rm \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY="your-key" \
  ramana-web
```

**Port already in use:**
```bash
# Check what's using port 5001
lsof -i :5001
# or
netstat -tulpn | grep 5001

# Use a different port
docker run -d \
  --name ramana-web \
  -p 8080:5001 \
  -e OPENROUTER_API_KEY="your-key" \
  ramana-web
# Then access at http://localhost:8080
```

**API key not working:**
- Verify your `OPENROUTER_API_KEY` is correct
- Check logs for authentication errors
- Make sure you have credits in your OpenRouter account

**"No passages loaded" warning:**
- This is normal if `corpus.jsonl` wasn't included in the build
- The app will still work, but sidebar passages won't be available
- Check logs to confirm the app started successfully

## Quick Test Script

Save this as `test-local.sh`:

```bash
#!/bin/bash

# Build the image
echo "Building Docker image..."
docker build -t ramana-web .

# Stop and remove existing container if it exists
docker stop ramana-web 2>/dev/null
docker rm ramana-web 2>/dev/null

# Run the container
echo "Starting container..."
docker run -d \
  --name ramana-web \
  -p 5001:5001 \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -v ramana-sessions:/app/sessions \
  ramana-web

# Wait a few seconds for startup
sleep 3

# Test the API
echo "Testing API..."
curl -s http://localhost:5001/api/nan-yar | head -20

echo ""
echo "Container is running!"
echo "View logs: docker logs -f ramana-web"
echo "Open browser: http://localhost:5001"
echo "Stop container: docker stop ramana-web"
```

Make it executable and run:
```bash
chmod +x test-local.sh
export OPENROUTER_API_KEY="your-key-here"
./test-local.sh
```
