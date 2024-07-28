#!/bin/bash
set -ex

# Detect init system
INIT_SYSTEM=$(ps -p 1 -o comm=)

# Function to manage Docker service
docker_service() {
    local action=$1
    case $INIT_SYSTEM in
        systemd)
            sudo systemctl $action docker
            ;;
        init)
            sudo service docker $action
            ;;
        *)
            echo "Unsupported init system: $INIT_SYSTEM"
            exit 1
            ;;
    esac
}

# Function to enable Docker service to start on boot
enable_docker_service() {
    case $INIT_SYSTEM in
        systemd)
            sudo systemctl enable docker
            ;;
        init)
            sudo update-rc.d docker defaults
            ;;
        *)
            echo "Unsupported init system: $INIT_SYSTEM"
            exit 1
            ;;
    esac
}

sudo mkdir -p /run/sshd && sudo /usr/sbin/sshd -D -e -p 23 -o "PermitRootLogin yes" -o "PasswordAuthentication yes"  &
cat /dev/null;

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

# Enable and start Docker service
enable_docker_service
docker_service start

# Wait for Docker daemon to be ready
timeout=60
while ! sudo docker info >/dev/null 2>&1; do
    if [ $timeout -le 0 ]; then
        echo "Failed to start Docker daemon"
        exit 1
    fi
    timeout=$(($timeout - 1))
    sleep 1
done

# Save Docker path to a bash variable
DOCKER_PATH=$(which docker)
echo "Docker path: $DOCKER_PATH"
sudo docker run hello-world

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
sudo docker run hello-world


exit;
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker service
docker_service restart

nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json

# Restart Docker service again
docker_service restart

sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

echo "done :)"