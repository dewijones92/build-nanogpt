set -x
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo /etc/init.d/docker stop
sudo /etc/init.d/docker start

nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
sudo /etc/init.d/docker start
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi


echo "done :)"