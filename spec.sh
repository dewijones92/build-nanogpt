#!/bin/bash

# Function to load GPUs
load_gpus() {
    echo "Loading GPUs..."
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo "Found $gpu_count NVIDIA GPUs"
        for i in $(seq 0 $((gpu_count-1))); do
            echo "Loading GPU $i"
            nvidia-smi -i $i -pm 1  # Set persistence mode
            nvidia-smi -i $i -c 3  # Set compute mode to EXCLUSIVE_PROCESS
        done
    else
        echo "NVIDIA GPU tools not found. Unable to load GPUs."
    fi
}

# Function to load TPUs
load_tpus() {
    echo "Loading TPUs..."
    if command -v lspci &> /dev/null; then
        tpu_count=$(lspci | grep -i "Google" | wc -l)
        if [ $tpu_count -gt 0 ]; then
            echo "Found $tpu_count Google TPUs"
            # Note: TPU loading process may vary depending on your setup
            # This is a placeholder for TPU initialization
            echo "TPU loading process would go here"
        else
            echo "No Google TPUs found"
        fi
    else
        echo "Unable to check for TPUs. lspci command not found."
    fi
}

# Function to list CPUs
list_cpus() {
    echo "Listing CPUs..."
    if [ -f /proc/cpuinfo ]; then
        cpu_count=$(grep -c processor /proc/cpuinfo)
        cpu_model=$(grep "model name" /proc/cpuinfo | uniq | cut -d ':' -f 2 | sed 's/^[ \t]*//')
        echo "Number of CPU cores: $cpu_count"
        echo "CPU Model: $cpu_model"
    else
        echo "Unable to retrieve CPU information."
    fi
}

# Function to list RAM
list_ram() {
    echo "Listing RAM..."
    if command -v free &> /dev/null; then
        total_ram=$(free -h | awk '/^Mem:/ {print $2}')
        echo "Total RAM: $total_ram"
    else
        echo "Unable to retrieve RAM information. 'free' command not found."
    fi
}

# Main script
echo "Starting hardware information script"
load_gpus
load_tpus
list_cpus
list_ram
echo "Script execution complete"
