# ProtIntel Production Deployment Guide

This guide details the exact steps required to deploy the ProtIntel application stack (FastAPI backend + Redis cache + Nginx frontend) onto a fresh Linux Virtual Private Server (VPS) or cloud VM (Ubuntu 22.04 LTS recommended) from scratch.

---

## Prerequisites

Before starting, ensure your VPS has:
- A minimum of **4 GB RAM** (required for ESM-2 650M CPU-only tokenization and forward passes).
- Port **3000** (frontend) and **8000** (API, optional) open in your firewall (UFW, Security Group).

---

## Step 1: Install Docker and Docker Compose

Log in to your VPS via SSH and install Docker:

```bash
# 1. Update package list
sudo apt-get update

# 2. Install Docker dependencies
sudo apt-get install -y ca-certificates curl gnupg

# 3. Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 4. Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Install Docker Engine and Docker Compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 6. Verify installation
sudo docker --version
sudo docker compose version
```

---

## Step 2: Clone the Repository & Configure Environment

Clone the codebase onto the server:

```bash
# 1. Clone repo
git clone https://github.com/RamaVenkataCharan/ProtIntel.git
cd ProtIntel

# 2. Create environment file from template
cp .env.example .env

# 3. Edit config using nano or vim
nano .env
```

Ensure the following variables are set in your `.env`:
```env
API_HOST=0.0.0.0
API_PORT=8000
DEVICE=cpu
MODEL_PATH=models/best_checkpoint_pruned.pt
REDIS_HOST=redis
REDIS_PORT=6379
CORS_ORIGINS=http://<YOUR_SERVER_IP>:3000
```

---

## Step 3: Place the Pruned Model Checkpoint

The pruned checkpoint is required for inference and must be placed in the `models/` directory (which is mounted as a volume by Docker Compose).

```bash
# Create models directory
mkdir -p models

# Copy/download your pruned checkpoint (79.7 MB) into the folder:
# wget -O models/best_checkpoint_pruned.pt <URL_TO_CHECKPOINT_STORE>
# Or copy from local development machine:
# scp models/best_checkpoint_pruned.pt user@server:/path/to/ProtIntel/models/
```

*Note: Ensure the filename matches the `MODEL_PATH` setting inside `.env`.*

---

## Step 4: Run the Docker Compose Stack

Build and start the microservices in detached mode:

```bash
# Start containers
sudo docker compose up -d --build
```

This will spin up:
1. `protintel-redis`: Redis v7 caching instance.
2. `protintel-backend`: FastAPI application server (port 8000).
3. `protintel-frontend`: Nginx server hosting the React app (port 3000) and proxying `/api/*` to the backend.

Verify the status of the running containers:
```bash
sudo docker compose ps
```

---

## Step 5: Verification & Smoke Test

### 1. Check Container Health
Wait ~10 seconds for the model weights to load, then run a curl check against the health endpoint:

```bash
curl -i http://localhost:8000/health
```

Expected response (`HTTP/1.1 200 OK`):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "redis_connected": true
}
```

*Note: If the model is not loaded or Redis is down, this endpoint will return `503 Service Unavailable`.*

### 2. Inspect Prometheus Metrics
Verify Prometheus metrics are active and showing request counters and queue depth:

```bash
curl http://localhost:8000/prometheus
```

### 3. Verify Frontend Access
Open your web browser and navigate to `http://<YOUR_SERVER_IP>:3000`. 
Submit a single sequence (e.g. `MKFLILLFN`) on the Predict tab and verify that:
- A spinner overlay appears.
- Predictions and explainability maps load successfully.

---

## Step 6: GPU Acceleration Setup (Optional but Recommended)

Because CPU ESM-2 inference takes ~4.75s, deploying on an Nvidia GPU-enabled server will reduce latency to <200ms.

1. **Install NVIDIA Container Toolkit**:
   Follow instructions at [docs.nvidia.com](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to enable GPU support inside Docker.

2. **Update `.env`**:
   Set `DEVICE=cuda` inside `.env`.

3. **Update `docker-compose.yml`**:
   Uncomment/add GPU reservations to the backend service:
   ```yaml
     backend:
       ...
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: all
                 capabilities: [gpu]
   ```

4. **Restart Containers**:
   ```bash
   sudo docker compose down && sudo docker compose up -d
   ```
