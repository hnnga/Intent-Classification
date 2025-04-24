## Prerequisites
- **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS/Windows) or `docker.io` and `docker-compose` (Linux).

## Milvus Setup
1. Download and start Milvus:
   ```bash
   mkdir milvus && cd milvus
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
   docker-compose up -d
   ```
2. Check containers:
   ```bash
   docker ps
   ```

## Run Application
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run:
   ```bash
   python main.py
   ```

## WebUI (Optional)
Run Attu for Milvus management:
```bash
docker run -d \
  --name attu \
  -e MILVUS_URL="localhost:19530" \
  -p 3000:3000 \
  zilliz/attu:v2.2.17
```
Access: [http://localhost:3000](http://localhost:3000)
