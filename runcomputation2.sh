#!/bin/bash
set -ex

# Detect init system
INIT_SYSTEM=$(ps -p 1 -o comm=)

# Function to start/stop Docker based on init system
docker_service() {
    local action=$1
    case $INIT_SYSTEM in
        systemd)
            sudo systemctl $action docker
            ;;
        init)
            sudo /etc/init.d/docker $action
            ;;
        *)
            echo "Unsupported init system: $INIT_SYSTEM"
            exit 1
            ;;
    esac
}

# Add Docker's official GPG key:
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Save Docker path to a bash variable
DOCKER_PATH=$(which docker)
echo "Docker path: $DOCKER_PATH"
$DOCKER_PATH run hello-world

filename=$(echo $HOME/bin/rootlesskit | sed -e s@^/@@ -e s@/@.@g)
cat <<EOF > ~/${filename}
abi <abi/4.0>,
include <tunables/global>
"$HOME/bin/rootlesskit" flags=(unconfined) {
  userns,
  include if exists <local/${filename}>
}
EOF
sudo mv ~/${filename} /etc/apparmor.d/${filename}
$DOCKER_PATH run hello-world

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

# Stop Docker service
docker_service stop

# Start Docker service
docker_service start

nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json

# Start Docker service again
docker_service start

sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

echo "done :)"