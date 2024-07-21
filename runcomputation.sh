#!/bin/bash

# Common variables
REMOTE_USER="wb1_user"
REMOTE_HOST="dewijones92vultr.duckdns.org"
REMOTE_PASSWORD="testpass"
REMOTE_LOG_DIR="/home/wb1_user/code/build-nanogpt/run_logs"

# Function to run a command with unbuffered output and log it
run_unbuffered() {
    local cmd="$1"
    local log_file="$2"
    
    set -x
    ub $cmd |& \
    ub ts |& \
    ub tee >(ub cat) >(sshpass -p "$REMOTE_PASSWORD" ssh -t "$REMOTE_USER@$REMOTE_HOST" \
        "bash -xc 'source ~/.bashrc && stdbuf -i0 -o0 -e0 bash -xc \"stdbuf -i0 -o0 -e0 cat >> $REMOTE_LOG_DIR/$log_file\"'")
}

# Function to transfer file to remote server
transfer_to_remote() {
    local local_file="$1"
    sshpass -p "$REMOTE_PASSWORD" scp "$local_file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_LOG_DIR/"
}

# Generate unique identifiers for log files
TIMESTAMP=$(date +%s%3N)
LOGFILE="run_log_$TIMESTAMP.txt"
PROFILE_OUTPUT="profile_output_$TIMESTAMP.pstats"

# Run the original commands
run_unbuffered "bash -x spec.sh" "$LOGFILE"

# Run train_gpt2.py with profiling
run_unbuffered "ub python -m cProfile -o $PROFILE_OUTPUT train_gpt2.py" "$LOGFILE"

# Analyze and stream the profile output
run_unbuffered "ub python -c 'import pstats; p = pstats.Stats(\"$PROFILE_OUTPUT\"); p.sort_stats(\"cumulative\").print_stats(50)'" "$PROFILE_OUTPUT"

# Transfer the .pstats file to the remote server
transfer_to_remote "$PROFILE_OUTPUT"

# Log the filenames locally
echo "Log file: $LOGFILE"
echo "Profile output: $PROFILE_OUTPUT"