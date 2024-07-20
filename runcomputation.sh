#!/bin/bash
set -x

# Generate a consistent filename for logging
LOGFILE="$(date +%s%3N)_$(date +%Y-%m-%d_%H-%M-%S).goodlog"

# Function to run a command with unbuffered output and log it
run_unbuffered() {
    local cmd="$1"
    ub $cmd |& \
    ub ts |& \
    ub tee >(ub cat) >(sshpass -p testpass ssh -t wb1_user@dewijones92vultr.duckdns.org \
        "bash -c 'source ~/.bashrc && stdbuf -i0 -o0 -e0 bash -c \"stdbuf -i0 -o0 -e0 cat >> /home/wb1_user/code/build-nanogpt/run_logs/$LOGFILE\"'")
}

# Run the commands
run_unbuffered "bash spec.sh"
run_unbuffered "python train_gpt2.py"

# Optionally, you can also log the filename locally
echo "Log file: $LOGFILE"