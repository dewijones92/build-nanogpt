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
set -x

get_command_path() {
    local cmd=$1
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

# Function to send profile output to server
send_profile_output() {
    if [ -f "$PROFILE_OUTPUT" ]; then
        sshpass -p "$REMOTE_PASSWORD" scp "$PROFILE_OUTPUT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_LOG_DIR/${LOGFILE}_$PROFILE_OUTPUT"
    fi
}

# Set up trap to send profile output on script exit
trap send_profile_output EXIT HUP INT QUIT TERM

run_unbuffered 'echo $PYTHON_PATH'
run_unbuffered "$PIP_PATH show pip"


export -f download_data_sources
# Run the new download function
run_unbuffered "bash -xc download_data_sources"
# Run the original commands
run_unbuffered "bash -x $SPEC_SCRIPT"
run_unbuffered "$PIP_PATH install line_profiler[all]"
run_unbuffered "$PIP_PATH install line_profiler"


echo "$PYTHON_PATH"
# Run the profiler and stream the output to the server
export LINE_PROFILE=1; run_unbuffered "$PYTHON_PATH $TRAIN_SCRIPT"
# Optionally, you can also log the filename locally
echo "Log file: $LOGFILE"
