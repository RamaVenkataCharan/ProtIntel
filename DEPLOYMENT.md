# ProtIntel Persistent Production VPS Deployment Guide

This guide details the exact steps to deploy and persistently run ProtIntel (FastAPI Backend + Redis Job/Cache Layer + React/Nginx Frontend) on a cloud VPS (Ubuntu 22.04/24.04 LTS, 4 GB RAM, 2 vCPUs minimum).

---

## 1. System Requirements & Security Configuration

### Prerequisites
- **OS**: Ubuntu 22.04 LTS / 24.04 LTS (x86_64)
- **CPU**: $\ge 2$ vCPUs
- **RAM**: $\ge 4\text{ GB}$
- **Storage**: $\ge 40\text{ GB}$ NVMe/SSD

### UFW Firewall Configuration
Expose only HTTP/HTTPS and SSH to the public internet:
```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP Frontend
sudo ufw allow 443/tcp   # HTTPS (if domain configured)
sudo ufw allow 3000/tcp  # Direct Frontend Port
sudo ufw enable
```

---

## 2. Docker & Environment Setup

```bash
# 1. Install Docker & Docker Compose Plugin
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 2. Clone Repository
git clone https://github.com/RamaVenkataCharan/ProtIntel.git
cd ProtIntel

# 3. Configure .env File
cp .env.example .env
```

Ensure `.env` contains real server settings (replace `<SERVER_PUBLIC_IP>`):
```env
API_HOST=0.0.0.0
API_PORT=8000
DEVICE=cpu
MODEL_PATH=models/best_checkpoint_pruned.pt
REDIS_HOST=redis
REDIS_PORT=6379
CORS_ORIGINS=http://<SERVER_PUBLIC_IP>:3000,http://<SERVER_PUBLIC_IP>
```

---

## 3. Checkpoint & Persistent Deployment

```bash
# 1. Ensure Model Checkpoint Exists
mkdir -p models
# Copy pruned model checkpoint (79.7 MB) into models/ directory

# 2. Launch Stack in Detached Mode
sudo docker compose up -d --build
```

### Persistence Verification (`restart: unless-stopped`)
All containers in `docker-compose.yml` are configured with `restart: unless-stopped`. To verify persistence across system reboots:
```bash
sudo reboot
```
Upon reboot, systemd automatically restarts Docker Engine and all container instances (`protintel-redis`, `protintel-backend`, `protintel-frontend`) start automatically without manual intervention.

---

## 4. SSL & Domain Configuration (Optional)

If a domain name (e.g. `protintel.example.com`) is pointed to your VPS public IP:
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d protintel.example.com
```

*Note: If deploying via raw IP address without a domain, HTTPS is not supported by ACME/Let's Encrypt, and the application runs persistently over HTTP on port 3000 / port 80.*

---

## 5. Deployment Verification Checklist

Run these commands on the live server IP:

1. **Backend Health Check**:
   ```bash
   curl -i http://<SERVER_PUBLIC_IP>:8000/health
   ```
   *Expected Response (`200 OK`)*:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "device": "cpu",
     "redis_connected": true
   }
   ```

2. **Rate Limiting Check**:
   ```bash
   for i in {1..12}; do curl -s -o /dev/null -w "%{http_code}\n" -X POST http://<SERVER_PUBLIC_IP>:8000/predict -H "Content-Type: application/json" -d '{"sequence":"MKFLILLFN"}'; done
   ```
   *Expected Outcome*: First 10 requests return `200`, 11th and 12th return `429 Too Many Requests`.

3. **End-to-End Prediction Test**:
   - Navigate to `http://<SERVER_PUBLIC_IP>:3000` in browser.
   - Enter sequence `MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG`.
   - Verify async job polling transitions: `Job Queued...` $\rightarrow$ `Calculating XAI...` $\rightarrow$ `3D Aurora Ribbon & Secondary Structure Map`.
