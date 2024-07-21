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

# Function to stream file to remote server
stream_to_remote() {
    local pipe_file="$1"
    local remote_file="$2"
    
    cat "$pipe_file" | sshpass -p "$REMOTE_PASSWORD" ssh "$REMOTE_USER@$REMOTE_HOST" \
        "cat > $REMOTE_LOG_DIR/$remote_file" &
}

# Generate unique identifiers for log files
TIMESTAMP=$(date +%s%3N)
LOGFILE="run_log_$TIMESTAMP.txt"
PROFILE_OUTPUT="profile_output_$TIMESTAMP.pstats"

# Run the original commands
run_unbuffered "bash -x spec.sh" "$LOGFILE"

# Create a named pipe for the profile output
PROFILE_PIPE="/tmp/profile_pipe_$TIMESTAMP"
mkfifo "$PROFILE_PIPE"

# Start streaming the profile output to the remote server
stream_to_remote "$PROFILE_PIPE" "$PROFILE_OUTPUT"

# Run train_gpt2.py with profiling, outputting to the named pipe
run_unbuffered "ub python -m cProfile -o $PROFILE_PIPE train_gpt2.py" "$LOGFILE"

# Close the named pipe
rm "$PROFILE_PIPE"

# Analyze the profile output on the remote server
run_unbuffered "sshpass -p $REMOTE_PASSWORD ssh $REMOTE_USER@$REMOTE_HOST 'python -c \"import pstats; p = pstats.Stats('\"$REMOTE_LOG_DIR/$PROFILE_OUTPUT\"'); p.sort_stats(\\\"cumulative\\\").print_stats(50)\"'" "${PROFILE_OUTPUT}_analysis.txt"

# Log the filenames locally
echo "Log file: $LOGFILE"
echo "Profile output: $PROFILE_OUTPUT"
echo "Profile analysis: ${PROFILE_OUTPUT}_analysis.txt"