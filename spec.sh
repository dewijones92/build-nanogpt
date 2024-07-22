#!/bin/bash

echo "System Information Script"
echo "========================="

# CPU Information
echo -e "\nCPU Information:"
lscpu | grep -E "^CPU\(s\):|^Thread\(s\) per core:|^Core\(s\) per socket:|^Model name:"

# RAM Information
echo -e "\nRAM Information:"
free -h | grep Mem:

# GPU Information (NVIDIA)
echo -e "\nGPU Information (NVIDIA):"
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "NVIDIA GPU not found or nvidia-smi not installed"
fi

ls -al /home/jupyter

# TPU Information
echo -e "\nTPU Information:"
if [ -d "/sys/class/tpu" ]; then
    echo "TPU found. Details:"
    ls -l /sys/class/tpu
else
    echo "No TPU found on this system"
fi

# Disk Information
echo -e "\nDisk Information:"
df -h | grep -E "^Filesystem|^/dev/"

# Network Information
echo -e "\nNetwork Information:"
ip addr | awk '/inet / {print $2}'

echo -e "\nScript completed."