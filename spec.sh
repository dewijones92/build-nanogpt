#!/bin/bash -x

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

# Function to list detailed CPU information
list_cpu_details() {
    echo "Listing detailed CPU information..."
    if [ -f /proc/cpuinfo ]; then
        physical_cpus=$(lscpu | grep "Socket(s):" | awk '{print $2}')
        cores_per_cpu=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
        total_cores=$((physical_cpus * cores_per_cpu))
        threads_per_core=$(lscpu | grep "Thread(s) per core:" | awk '{print $4}')
        total_threads=$((total_cores * threads_per_core))
        cpu_model=$(grep "model name" /proc/cpuinfo | uniq | cut -d ':' -f 2 | sed 's/^[ \t]*//')

        echo "CPU Model: $cpu_model"
        echo "Number of Physical CPUs: $physical_cpus"
        echo "Cores per CPU: $cores_per_cpu"
        echo "Total CPU Cores: $total_cores"
        echo "Threads per Core: $threads_per_core"
        echo "Total CPU Threads: $total_threads"

        echo -e "\nDetailed Core/Thread Information:"
        awk '/^processor|^core id|^physical id/' /proc/cpuinfo | \
        paste - - - | \
        awk '{printf "Processor: %s, Physical ID: %s, Core ID: %s\n", $3, $9, $15}'
    else
        echo "Unable to retrieve CPU information."
    fi
}

# Function to list RAM
list_ram() {
    echo "Listing RAM..."
    if command -v free &> /dev/null; then
        total_ram=$(free -h | awk '/^Mem:/ {print $2}')
        used_ram=$(free -h | awk '/^Mem:/ {print $3}')
        free_ram=$(free -h | awk '/^Mem:/ {print $4}')
        echo "Total RAM: $total_ram"
        echo "Used RAM: $used_ram"
        echo "Free RAM: $free_ram"
    else
        echo "Unable to retrieve RAM information. 'free' command not found."
    fi
}

# Main script
echo "Starting hardware information script"
load_gpus
load_tpus
list_cpu_details
list_ram
echo "Script execution complete"