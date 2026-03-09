#!/usr/bin/env bash
# Rezaa AI — Bootstrap an Amazon Linux 2023 EC2 instance.
# Run as ec2-user after SSH-ing in.
set -euo pipefail

REPO_URL="https://github.com/alamayaz/amz_hack2skill.git"   # <-- update if different
APP_DIR="$HOME/rezaa"

echo "==> Installing Docker..."
sudo dnf update -y -q
sudo dnf install -y -q docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

echo "==> Installing Docker Compose v2 plugin..."
COMPOSE_VERSION="v2.27.0"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Verify
docker compose version

echo "==> Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "    $APP_DIR exists — pulling latest..."
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

###############################################################################
# .env setup
###############################################################################
if [ ! -f .env ]; then
    echo ""
    read -rp "Enter your REZAA_OPENAI_API_KEY: " OPENAI_KEY
    echo "REZAA_OPENAI_API_KEY=$OPENAI_KEY" > .env
    echo "    .env created."
else
    echo "    .env already exists, skipping."
fi

###############################################################################
# Build & start
###############################################################################
echo "==> Building and starting containers..."
# newgrp docker is needed in the same session since we just added the user to
# the docker group — use sudo for the first run.
sudo docker compose up -d --build

echo ""
echo "==> Waiting for API to be healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/ >/dev/null 2>&1; then
        echo "    API is up!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "    WARNING: API did not respond after 30s. Check logs:"
        echo "    sudo docker compose logs api"
        exit 1
    fi
    sleep 1
done

echo ""
echo "============================================"
echo "  Rezaa AI is running!"
echo "============================================"
echo "  URL   : http://$(curl -s https://checkip.amazonaws.com):8000"
echo ""
echo "  Useful commands:"
echo "    sudo docker compose logs -f        # tail logs"
echo "    sudo docker stats                  # monitor RAM/CPU"
echo "    sudo docker compose down           # stop"
echo "    sudo docker compose up -d --build  # rebuild & restart"
echo "============================================"
