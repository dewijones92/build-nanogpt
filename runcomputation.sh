#!/bin/bash -xi
unset HISTFILE
# Configuration variables
export REMOTE_USER="wb1_user"
export REMOTE_HOST="dewijones92vultr.duckdns.org"
export REMOTE_PASSWORD="testpass"
export REMOTE_LOG_DIR="/home/wb1_user/code/build-nanogpt/run_logs"
export LOCAL_TRAIN_DATA_DIR="train_data"
export LOCAL_FINEWEB_DIR="edu_fineweb10B"
export DATA_SOURCE_LIST="data-source-list-books"
export FINEWEB_SCRIPT="fineweb.py"
export SPEC_SCRIPT="spec.sh"
export TRAIN_SCRIPT="train_gpt2.py"
export PROFILE_OUTPUT="profile_output.txt"

IP="$(curl -4 icanhazip.com)"
# Generate a consistent filename for logging
LOGFILE=$(date +%s%3N)_$(date +%Y-%m-%d_%H-%M-%S)_${IP}_$(hostname).goodlog

# Function to run a command with unbuffered output and log it
run_unbuffered() {
    set -x
    local cmd="$@"
    ubf $cmd |& \
    ubf ts |& \
    ubf tee >(ubf cat) >(sshpass -p "$REMOTE_PASSWORD" ssh -t "$REMOTE_USER@$REMOTE_HOST" \
        "bash -xc 'source ~/.bashrc && stdbuf -i0 -o0 -e0 bash -xc \"stdbuf -i0 -o0 -e0 cat >> $REMOTE_LOG_DIR/$LOGFILE\"'")
}
export -f run_unbuffered

ubf() {
    stdbuf -i0 -o0 -e0 "$@"
}
export -f ubf

# Function to download data sources
download_data_sources() {
    # Remove existing data
    rm -rf "$LOCAL_TRAIN_DATA_DIR"/*
   
    # Read each line from data-source-list-books
    while IFS= read -r line
    do
        # Skip lines starting with #
        [[ $line =~ ^#.*$ ]] && continue
       
        # Download the file
        wget -P "$LOCAL_TRAIN_DATA_DIR"/ "$line"
    done < "$DATA_SOURCE_LIST"
    rm -rf "$LOCAL_FINEWEB_DIR"/*
    $PYTHON_PATH "$FINEWEB_SCRIPT" --source 2
}
export -f download_data_sources

# Function to send profile output to server
send_profile_output() {
    if [ -f "$PROFILE_OUTPUT" ]; then
        sshpass -p "$REMOTE_PASSWORD" scp "$PROFILE_OUTPUT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_LOG_DIR/${LOGFILE}_$PROFILE_OUTPUT"
    fi
}

# Set up trap to send profile output on script exit
trap send_profile_output EXIT HUP INT QUIT TERM

# Main script function
main_script() {
    set -x
    get_command_path() {
        local cmd=$@
        local path=$(type -a "$cmd" | awk '/aliased|\/'"$cmd"'/ {print $NF; exit}' | tr -d "'")
        path=${path#\`}  # Remove leading backtick if present
        if [ -z "$path" ]; then
            echo "$cmd empty"
            path="$cmd"
        fi
        echo "$path"
    }
    PYTHON_PATH=$(get_command_path python)
    PIP_PATH=$(get_command_path pip)
    export PYTHON_PATH
    export PIP_PATH
    echo "Python path: $PYTHON_PATH"
    echo "Pip path: $PIP_PATH"

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install libcudnn9-dev-cuda-12

    (cd ~ && git clone -b main https://github.com/NVIDIA/cudnn-frontend.git)

  #  CUDNN_PATH=$(bash "get-latest-cudnn")
   # export CUDNN_PATH;

    echo "$PYTHON_PATH"
    "$PIP_PATH" show pip
    download_data_sources
    bash -x $SPEC_SCRIPT
    "$PIP_PATH" install line_profiler[all]
    "$PIP_PATH" install line_profiler
    "$PYTHON_PATH" -m pip install line_profiler
    $PYTHON_PATH -c "import sys; print(sys.path)"

    (repo_url="https://github.com/dewijones92/llm.c.git";
    branch="dewi-mod-shakespear";
    dir="llminc";
    mkdir -p "$dir" && cd "$dir";
    git clone -b "$branch" "$repo_url";
    cd llm.c;
    git fetch --all;
    git checkout "origin/$branch";
    ls -al;
    pwd;
    bash -x go.sh)

    echo "$PYTHON_PATH"
    #export LINE_PROFILE=1; "$PYTHON_PATH" $TRAIN_SCRIPT
}
export -f main_script

# Run the entire script through run_unbuffered
run_unbuffered "bash -c main_script"

# Optionally, you can also log the filename locally
echo "Log file: $LOGFILE"